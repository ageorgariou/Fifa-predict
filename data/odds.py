"""FIFA World Cup 2026 tournament-winner odds from the-odds-api.com.

Sport key: `soccer_fifa_world_cup_winner` (futures market).
Per-bookmaker overround is removed by normalizing implied probabilities to
sum to 1, then we average across all available bookmakers.

Cached 24h via data.cache. On network failure, falls back to
data/odds_winner_snapshot.json. Quota is 500/month on the free tier — the
24h cache means at most ~30 calls per month under normal use.

Public API:
    fetch_winner_probs() -> dict[str, float]   # canonical team name → P(win tournament)
    top_n(n: int = 8) -> list[tuple[str, float]]
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict

import httpx

from config import DATA_DIR, ODDS_API_KEY, TTL_TOURNAMENT_ODDS
from data.cache import cache_get, cache_set
from data.elo import ScraperError  # reuse the same exception type
from model.groups_2026 import GROUPS_2026, NAME_ALIASES

log = logging.getLogger(__name__)

SPORT_KEY = "soccer_fifa_world_cup_winner"
ODDS_URL = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds"
REGIONS = "eu"           # sharpest exchange-driven prices for outrights
MARKET = "outrights"     # back side only; ignore `outrights_lay`
CACHE_KEY = "odds:wc2026:winner"
SNAPSHOT_PATH = DATA_DIR / "odds_winner_snapshot.json"


# Mapping from bookmaker-side names → canonical names in groups_2026.GROUPS_2026.
# Built once from NAME_ALIASES; extend here if a bookmaker uses a name we
# can't normalize automatically.
_BOOK_TO_CANONICAL: dict[str, str] = {}
for canonical, aliases in NAME_ALIASES.items():
    _BOOK_TO_CANONICAL[canonical.lower()] = canonical
    for alias in aliases:
        _BOOK_TO_CANONICAL[alias.lower()] = canonical
# Identity for everything already canonical
for group_teams in GROUPS_2026.values():
    for team in group_teams:
        _BOOK_TO_CANONICAL.setdefault(team.lower(), team)


def _canonicalize(name: str) -> str | None:
    """Map a bookmaker's team name to the canonical name used in groups_2026.
    Returns None if no match — caller can decide whether to drop or surface."""
    return _BOOK_TO_CANONICAL.get(name.strip().lower())


def _normalize_book(outcomes: list[dict]) -> dict[str, float]:
    """Convert one bookmaker's {name, price} list into {canonical_name:
    overround-normalized probability}. Drops outcomes we can't canonicalize."""
    raw: dict[str, float] = {}
    for o in outcomes:
        canon = _canonicalize(o["name"])
        if canon is None:
            continue
        price = float(o["price"])
        if price <= 1.0:
            continue
        raw[canon] = raw.get(canon, 0.0) + 1.0 / price
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {team: p / total for team, p in raw.items()}


def _save_snapshot(probs: dict[str, float]) -> None:
    SNAPSHOT_PATH.write_text(json.dumps(probs, indent=2, sort_keys=True))


def _load_snapshot() -> dict[str, float] | None:
    if not SNAPSHOT_PATH.exists():
        return None
    try:
        return json.loads(SNAPSHOT_PATH.read_text())
    except json.JSONDecodeError:
        log.warning("odds snapshot at %s is corrupt", SNAPSHOT_PATH)
        return None


def _fetch_remote() -> dict[str, float]:
    if not ODDS_API_KEY:
        raise ScraperError("ODDS_API_KEY not set in environment / Streamlit secrets")
    with httpx.Client(timeout=20.0) as client:
        resp = client.get(
            ODDS_URL,
            params={
                "apiKey": ODDS_API_KEY,
                "regions": REGIONS,
                "markets": MARKET,
                "oddsFormat": "decimal",
            },
        )
        resp.raise_for_status()
        quota_left = resp.headers.get("x-requests-remaining")
        if quota_left is not None:
            log.info("the-odds-api quota remaining: %s", quota_left)
    events = resp.json()
    if not events:
        raise ScraperError("the-odds-api returned no events for winner market")

    # Average normalized probabilities across all bookmakers for the (single)
    # outrights event. Each book gets equal weight; missing teams contribute 0
    # for that book, so the average is biased low for teams not listed by all
    # books — acceptable for our top-8 ranking use case.
    per_book: list[dict[str, float]] = []
    for ev in events:
        for bk in ev.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                if mkt.get("key") != MARKET:
                    continue
                normalized = _normalize_book(mkt.get("outcomes", []))
                if normalized:
                    per_book.append(normalized)
    if not per_book:
        raise ScraperError("no usable outrights markets parsed")

    summed: dict[str, float] = defaultdict(float)
    for book_probs in per_book:
        for team, p in book_probs.items():
            summed[team] += p
    n_books = len(per_book)
    avg = {team: total / n_books for team, total in summed.items()}
    # Re-normalize: the averaging step can leave the total slightly off 1.0
    # when some teams are missing from some books.
    s = sum(avg.values())
    if s > 0:
        avg = {team: p / s for team, p in avg.items()}
    return avg


def fetch_winner_probs() -> dict[str, float]:
    """Returns {canonical_team_name: normalized implied probability of winning
    the tournament}, averaged across all available bookmakers."""
    cached = cache_get(CACHE_KEY, TTL_TOURNAMENT_ODDS)
    if cached is not None:
        return cached

    try:
        probs = _fetch_remote()
    except Exception as exc:  # noqa: BLE001 — degrade gracefully
        log.warning("the-odds-api fetch failed: %s — falling back to snapshot", exc)
        snap = _load_snapshot()
        if snap is None:
            raise ScraperError(
                "Failed to fetch tournament-winner odds and no snapshot is bundled."
            ) from exc
        return snap

    cache_set(CACHE_KEY, probs)
    _save_snapshot(probs)
    return probs


def top_n(n: int = 8) -> list[tuple[str, float]]:
    probs = fetch_winner_probs()
    return sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:n]


if __name__ == "__main__":
    probs = fetch_winner_probs()
    print(f"{'Rank':>4}  {'Team':<24}  P(win)   Implied odds")
    for i, (team, p) in enumerate(top_n(8), start=1):
        dec_odds = 1.0 / p if p > 0 else float("inf")
        print(f"{i:>4}  {team:<24}  {p*100:5.2f}%   {dec_odds:.2f}")
    print(f"\nTotal teams priced: {len(probs)} | sum(p) = {sum(probs.values()):.4f}")
