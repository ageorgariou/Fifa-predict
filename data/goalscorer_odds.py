"""Per-player anytime-goalscorer odds from the-odds-api.com.

For each event we hit the per-event odds endpoint with
markets=player_goal_scorer_anytime. Each bookmaker's listed outcomes are
converted decimal → implied probability and then normalized inside that book
so the per-book vector sums to 1.0. Player-level estimates are aggregated
across books via median (robust to one stale book).

Normalization caveat
--------------------
Anytime-goalscorer is a YES/NO market, not a mutex set, so dividing by the
sum within a book is only an approximation of overround removal. It tends to
under-estimate p_score in matches where multiple goals are expected. We
compensate downstream in model/odds_blend.py (0.6 book / 0.4 stat blend) and
calibrate via the 2018+2022 backtest in step 16.

Cache: 6 hours per (sport_key, event_id, regions). Quota cost is one credit
per book per market per call.

Public API
----------
    fetch_event_goalscorer_probs(sport_key, event_id, regions='eu,uk,us')
        -> dict[str, float]            # {player_name: p_score}
    fetch_wc_goalscorer_probs(commence_window_days=10)
        -> dict[tuple[str, str], float]  # {(player_name, event_id): p_score}
    list_wc_events(commence_window_days=10) -> list[dict]
"""
from __future__ import annotations

import logging
import statistics
from datetime import datetime, timedelta, timezone

import httpx

from config import ODDS_API_KEY, TTL_GOALSCORER_ODDS
from data.cache import cache_get, cache_set
from data.elo import ScraperError

log = logging.getLogger(__name__)

WC_SPORT_KEY = "soccer_fifa_world_cup"
MARKET = "player_goal_scorer_anytime"
DEFAULT_REGIONS = "eu,uk,us"
BASE_URL = "https://api.the-odds-api.com/v4"


def _outcomes_to_probs(outcomes: list[dict]) -> dict[str, float]:
    """Per-book: raw 1/odds for each player, then divide by sum so the vector
    sums to 1.0 within the book. Returns {player_name: normalized_prob}."""
    raw: dict[str, float] = {}
    for o in outcomes:
        name = (o.get("description") or o.get("name") or "").strip()
        if not name:
            continue
        try:
            price = float(o["price"])
        except (KeyError, TypeError, ValueError):
            continue
        if price <= 1.0:
            continue
        raw[name] = raw.get(name, 0.0) + 1.0 / price
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {name: p / total for name, p in raw.items()}


def _aggregate_books(per_book: list[dict[str, float]]) -> dict[str, float]:
    """Median across books per player. Books that don't list a player don't
    contribute zeros to the median (they're treated as missing data)."""
    by_player: dict[str, list[float]] = {}
    for book in per_book:
        for name, p in book.items():
            by_player.setdefault(name, []).append(p)
    return {name: statistics.median(vals) for name, vals in by_player.items()}


def fetch_event_goalscorer_probs(
    sport_key: str,
    event_id: str,
    regions: str = DEFAULT_REGIONS,
) -> dict[str, float]:
    """Returns {player_name: P(scores anytime)} for one event.

    Empty dict means no bookmaker is currently pricing this market for this
    event (typical for matches >7 days out). Caller should fall back to the
    stat-only projection and flag low_confidence."""
    key = f"goalscorer:{sport_key}:{event_id}:{regions}"
    cached = cache_get(key, TTL_GOALSCORER_ODDS)
    if cached is not None:
        return cached

    if not ODDS_API_KEY:
        raise ScraperError("ODDS_API_KEY not set in environment / Streamlit secrets")

    url = f"{BASE_URL}/sports/{sport_key}/events/{event_id}/odds"
    with httpx.Client(timeout=20.0) as client:
        resp = client.get(url, params={
            "apiKey": ODDS_API_KEY,
            "regions": regions,
            "markets": MARKET,
            "oddsFormat": "decimal",
        })
    if resp.status_code != 200:
        raise ScraperError(
            f"the-odds-api {resp.status_code} for {sport_key}/{event_id}: {resp.text[:200]}"
        )
    quota = resp.headers.get("x-requests-remaining")
    if quota is not None:
        log.info("the-odds-api quota remaining: %s", quota)

    data = resp.json()
    per_book: list[dict[str, float]] = []
    for bk in data.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m.get("key") != MARKET:
                continue
            book_probs = _outcomes_to_probs(m.get("outcomes", []))
            if book_probs:
                per_book.append(book_probs)

    aggregated = _aggregate_books(per_book) if per_book else {}
    cache_set(key, aggregated)
    return aggregated


def list_wc_events(commence_window_days: int = 10) -> list[dict]:
    """Returns upcoming WC events whose commence_time is within the next
    `commence_window_days` days. Result is a list of dicts with keys
    `id`, `commence_time`, `home_team`, `away_team`."""
    if not ODDS_API_KEY:
        raise ScraperError("ODDS_API_KEY not set in environment / Streamlit secrets")
    cache_key = f"goalscorer:wc_events:{commence_window_days}"
    cached = cache_get(cache_key, TTL_GOALSCORER_ODDS)
    if cached is not None:
        return cached

    with httpx.Client(timeout=20.0) as client:
        resp = client.get(
            f"{BASE_URL}/sports/{WC_SPORT_KEY}/events",
            params={"apiKey": ODDS_API_KEY},
        )
        resp.raise_for_status()
    events = resp.json()

    cutoff = datetime.now(timezone.utc) + timedelta(days=commence_window_days)
    upcoming = [
        ev for ev in events
        if datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")) <= cutoff
    ]
    cache_set(cache_key, upcoming)
    return upcoming


def fetch_wc_goalscorer_probs(
    commence_window_days: int = 10,
) -> dict[tuple[str, str], float]:
    """For every WC match commencing within `commence_window_days`, fetch
    anytime-goalscorer odds. Returns {(player_name, event_id): p_score}.
    Events with no live markets contribute nothing and emit a debug log."""
    out: dict[tuple[str, str], float] = {}
    for ev in list_wc_events(commence_window_days):
        probs = fetch_event_goalscorer_probs(WC_SPORT_KEY, ev["id"])
        if not probs:
            log.debug("no goalscorer markets open yet for %s vs %s",
                      ev.get("home_team"), ev.get("away_team"))
            continue
        for player, p in probs.items():
            out[(player, ev["id"])] = p
    return out


if __name__ == "__main__":
    # ---- Verification fixture: 2026 UEFA Champions League Final ----
    # Used to prove parser/normalize/cache/median-aggregation logic against
    # real live data, since WC matches don't have player markets yet
    # (~15 days from MD1 as of run date).
    cl_event_id = "a54f22aca3be31d95f13eac0aeac62cf"
    probs = fetch_event_goalscorer_probs("soccer_uefa_champs_league", cl_event_id)
    print(f"=== CL Final (PSG vs Arsenal): {len(probs)} unique players priced ===")
    print(f"{'Rank':>4}  {'Player':<28}  P(score)   Median odds")
    for i, (name, p) in enumerate(
        sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:12], start=1
    ):
        dec_odds = 1.0 / p if p > 0 else float("inf")
        print(f"{i:>4}  {name:<28}  {p*100:5.2f}%    {dec_odds:.2f}")
    sanity = sum(probs.values())
    print(f"\nsum(probs) = {sanity:.4f} (~1.0 expected per mutex normalization)")

    # ---- WC probe: confirm we gracefully return empty until books open ----
    print("\n=== WC events within next 10 days ===")
    events = list_wc_events(10)
    print(f"in-window events: {len(events)}")
    wc_probs = fetch_wc_goalscorer_probs(10)
    print(f"player-event prob pairs returned: {len(wc_probs)}")
    if not wc_probs:
        print("(expected: WC bookmaker player markets open ~5-7 days before kickoff)")
