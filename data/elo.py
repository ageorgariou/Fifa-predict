"""World Elo ratings from eloratings.net/World.tsv.

The TSV ships 2-letter codes only (mostly ISO 3166-1 alpha-2; eloratings.net
splits the UK into EN/SC/WL/NI and uses non-ISO codes for a few cases).
We map those codes onto the canonical team names used by model/groups_2026.py
so downstream code can join on a single naming convention.

Cached for 24h via data.cache. On network failure, falls back to the last
successful fetch persisted to data/elo_snapshot.json.

Public API:
    fetch_elo() -> dict[str, float]   # canonical team name → Elo rating
    top_n(n: int = 10) -> list[tuple[str, float]]
"""
from __future__ import annotations

import json
import logging

import httpx

from config import DATA_DIR, TTL_ELO
from data.cache import cache_get, cache_set

log = logging.getLogger(__name__)

ELO_URL = "https://www.eloratings.net/World.tsv"
CACHE_KEY = "elo:world"
SNAPSHOT_PATH = DATA_DIR / "elo_snapshot.json"
USER_AGENT = "Mozilla/5.0 (WC Fantasy Advisor)"


class ScraperError(RuntimeError):
    """Raised when neither network nor snapshot can produce data."""


# eloratings.net 2-letter code → canonical name (matches model/groups_2026.py).
# Covers all 48 teams in the 2026 World Cup draw. If a code is added later
# (e.g. a play-off winner not in the original draw), add it here.
ELO_CODE_TO_TEAM: dict[str, str] = {
    # Group A
    "MX": "Mexico", "KR": "South Korea", "ZA": "South Africa", "CZ": "Czechia",
    # Group B
    "CA": "Canada", "CH": "Switzerland", "QA": "Qatar", "BA": "Bosnia-Herzegovina",
    # Group C
    "BR": "Brazil", "MA": "Morocco", "SC": "Scotland", "HT": "Haiti",
    # Group D
    "US": "United States", "PY": "Paraguay", "AU": "Australia", "TR": "Turkiye",
    # Group E
    "DE": "Germany", "EC": "Ecuador", "CI": "Ivory Coast", "CW": "Curacao",
    # Group F
    "NL": "Netherlands", "JP": "Japan", "TN": "Tunisia", "SE": "Sweden",
    # Group G
    "BE": "Belgium", "IR": "Iran", "EG": "Egypt", "NZ": "New Zealand",
    # Group H
    "ES": "Spain", "UY": "Uruguay", "SA": "Saudi Arabia", "CV": "Cape Verde",
    # Group I
    "FR": "France", "SN": "Senegal", "NO": "Norway", "IQ": "Iraq",
    # Group J
    "AR": "Argentina", "AT": "Austria", "DZ": "Algeria", "JO": "Jordan",
    # Group K
    "PT": "Portugal", "CO": "Colombia", "UZ": "Uzbekistan", "CD": "DR Congo",
    # Group L
    "EN": "England", "HR": "Croatia", "PA": "Panama", "GH": "Ghana",
}


def _parse_tsv(text: str) -> dict[str, float]:
    """Returns {2-letter code: rating} parsed from World.tsv.

    Field layout (verified 2026-05-27): [0]=rank, [1]=numeric id, [2]=2-letter
    code, [3]=rating, then W/D/L/goal columns we don't use.
    """
    out: dict[str, float] = {}
    for raw in text.replace("\r", "").split("\n"):
        if not raw.strip():
            continue
        parts = raw.split("\t")
        if len(parts) < 4:
            continue
        code = parts[2].strip()
        if not code:
            continue
        try:
            rating = float(parts[3])
        except ValueError:
            continue
        out[code] = rating
    return out


def _map_to_canonical(by_code: dict[str, float]) -> dict[str, float]:
    """Convert {code: rating} to {canonical_team_name: rating} for the 48 WC
    teams. Logs any WC team whose code is missing from the feed."""
    by_team: dict[str, float] = {}
    for code, name in ELO_CODE_TO_TEAM.items():
        if code in by_code:
            by_team[name] = by_code[code]
        else:
            log.warning("Elo feed missing code %s (%s)", code, name)
    return by_team


def _save_snapshot(ratings: dict[str, float]) -> None:
    SNAPSHOT_PATH.write_text(json.dumps(ratings, indent=2, sort_keys=True))


def _load_snapshot() -> dict[str, float] | None:
    if not SNAPSHOT_PATH.exists():
        return None
    try:
        return json.loads(SNAPSHOT_PATH.read_text())
    except json.JSONDecodeError:
        log.warning("elo snapshot at %s is corrupt", SNAPSHOT_PATH)
        return None


def _fetch_remote() -> dict[str, float]:
    with httpx.Client(timeout=15.0, headers={"User-Agent": USER_AGENT}) as client:
        resp = client.get(ELO_URL)
        resp.raise_for_status()
    by_code = _parse_tsv(resp.text)
    if len(by_code) < 200:
        raise ScraperError(
            f"eloratings.net returned only {len(by_code)} parseable rows; "
            "format likely changed"
        )
    return _map_to_canonical(by_code)


def fetch_elo() -> dict[str, float]:
    """Return current Elo ratings as {canonical_team_name: rating} for the 48
    WC teams.

    Cache hit (≤24h)  → cached value
    Cache miss        → remote fetch; on success refresh cache + snapshot
    Remote failure    → bundled snapshot
    No snapshot       → ScraperError
    """
    cached = cache_get(CACHE_KEY, TTL_ELO)
    if cached is not None:
        return cached

    try:
        ratings = _fetch_remote()
    except Exception as exc:  # noqa: BLE001 — degrade gracefully on any fetch error
        log.warning("eloratings.net fetch failed: %s — falling back to snapshot", exc)
        snap = _load_snapshot()
        if snap is None:
            raise ScraperError(
                "Failed to fetch eloratings.net and no snapshot is bundled."
            ) from exc
        return snap

    cache_set(CACHE_KEY, ratings)
    _save_snapshot(ratings)
    return ratings


def top_n(n: int = 10) -> list[tuple[str, float]]:
    ratings = fetch_elo()
    return sorted(ratings.items(), key=lambda kv: kv[1], reverse=True)[:n]


if __name__ == "__main__":
    print(f"{'Rank':>4}  {'Team':<24}  Elo")
    for i, (team, rating) in enumerate(top_n(10), start=1):
        print(f"{i:>4}  {team:<24}  {rating:.1f}")
    print(f"\nTotal WC teams mapped from feed: {len(fetch_elo())} / 48")
