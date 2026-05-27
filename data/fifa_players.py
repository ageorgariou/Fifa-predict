"""FIFA World Cup 2026 Fantasy player list from play.fifa.com.

Primary source: the play.fifa.com static JSON endpoints
  https://play.fifa.com/json/fantasy/players.json   (~1450 players)
  https://play.fifa.com/json/fantasy/squads.json    (48 teams)

We join player.squadId → squad.name/group to expose the canonical team
name from each player's record. Cache TTL is 24h (TTL_FIFA_PLAYERS); the
list only changes when squads add/drop players before a deadline.

Manual override
---------------
data/manual_players.csv with columns
    player, team, position, price
acts as both fallback (when the endpoint is unreachable) and supplemental
source for players the official list omits (rare edge cases, late call-ups).
Manual rows are merged onto the API list on the (player, team) key — manual
takes precedence for `price` / `position` if both exist.

Public API
----------
    fetch_player_list() -> pd.DataFrame
        Columns: id, name, team, team_abbr, group, position, price,
        ownership_pct, status, source ('api' or 'manual')

Verification (__main__): prints counts by position, top-5 by ownership,
and a sample of each position.
"""
from __future__ import annotations

import logging

import httpx
import pandas as pd

from config import DATA_DIR, TTL_FIFA_PLAYERS
from data.cache import cache_get, cache_set
from data.elo import ScraperError
from model.groups_2026 import canonical_team

log = logging.getLogger(__name__)

PLAYERS_URL = "https://play.fifa.com/json/fantasy/players.json"
SQUADS_URL = "https://play.fifa.com/json/fantasy/squads.json"
CACHE_KEY = "fifa:players:wc2026"
USER_AGENT = "Mozilla/5.0 (WC Fantasy Advisor)"
MANUAL_CSV = DATA_DIR / "manual_players.csv"


def _player_display_name(p: dict) -> str:
    known = p.get("knownName")
    if known:
        return known.strip()
    parts = [p.get("firstName") or "", p.get("lastName") or ""]
    return " ".join(s.strip() for s in parts if s.strip())


def _fetch_remote() -> pd.DataFrame:
    with httpx.Client(timeout=20.0, headers={"User-Agent": USER_AGENT}) as client:
        players_resp = client.get(PLAYERS_URL)
        players_resp.raise_for_status()
        squads_resp = client.get(SQUADS_URL)
        squads_resp.raise_for_status()
    players = players_resp.json()
    squads = squads_resp.json()

    squad_map = {s["id"]: s for s in squads}
    rows = []
    for p in players:
        squad = squad_map.get(p.get("squadId"), {})
        raw_team = squad.get("name")
        # FIFA returns "USA", "Türkiye", etc. — normalize to canonical so
        # every downstream module joins on the same names that
        # model/groups_2026.GROUPS_2026 uses.
        team = canonical_team(raw_team) or raw_team
        rows.append({
            "id": p.get("id"),
            "name": _player_display_name(p),
            "team": team,
            "team_abbr": squad.get("abbr"),
            "group": squad.get("group"),
            "position": p.get("position"),
            "price": p.get("price"),
            "ownership_pct": p.get("percentSelected"),
            "status": p.get("status"),
            "source": "api",
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ScraperError("play.fifa.com returned no players")
    return df


def _load_manual() -> pd.DataFrame:
    if not MANUAL_CSV.exists():
        return pd.DataFrame(columns=["player", "team", "position", "price"])
    df = pd.read_csv(MANUAL_CSV).rename(columns={"player": "name"})
    df["source"] = "manual"
    return df


def _merge_manual(api_df: pd.DataFrame) -> pd.DataFrame:
    """Override price/position from manual CSV when (name, team) matches,
    and append any manual-only rows."""
    manual = _load_manual()
    if manual.empty:
        return api_df

    api_keyed = api_df.set_index(["name", "team"], drop=False)
    for _, m in manual.iterrows():
        idx = (m["name"], m["team"])
        if idx in api_keyed.index:
            if pd.notna(m.get("price")):
                api_keyed.loc[idx, "price"] = m["price"]
            if pd.notna(m.get("position")):
                api_keyed.loc[idx, "position"] = m["position"]
        else:
            api_keyed.loc[idx, :] = {
                "id": pd.NA,
                "name": m["name"],
                "team": m["team"],
                "team_abbr": pd.NA,
                "group": pd.NA,
                "position": m.get("position"),
                "price": m.get("price"),
                "ownership_pct": pd.NA,
                "status": "manual",
                "source": "manual",
            }
    return api_keyed.reset_index(drop=True)


def fetch_player_list() -> pd.DataFrame:
    """Return the merged FIFA + manual player DataFrame.

    Cache hit (≤24h)  → cached value
    Cache miss        → remote fetch; cache + return
    Remote failure    → manual CSV only (with warning); if also empty,
                        raises ScraperError.
    """
    cached = cache_get(CACHE_KEY, TTL_FIFA_PLAYERS)
    if cached is not None:
        return cached

    try:
        api_df = _fetch_remote()
    except Exception as exc:  # noqa: BLE001 — degrade gracefully
        log.warning("play.fifa.com fetch failed: %s — using manual CSV only", exc)
        manual = _load_manual()
        if manual.empty:
            raise ScraperError(
                "play.fifa.com unreachable and data/manual_players.csv is empty"
            ) from exc
        manual.rename(columns={"name": "name"}, inplace=True)
        for col in ("id", "team_abbr", "group", "ownership_pct", "status"):
            if col not in manual.columns:
                manual[col] = pd.NA
        return manual[
            ["id", "name", "team", "team_abbr", "group", "position",
             "price", "ownership_pct", "status", "source"]
        ]

    merged = _merge_manual(api_df)
    cache_set(CACHE_KEY, merged)
    return merged


if __name__ == "__main__":
    df = fetch_player_list()
    print(f"total players: {len(df)} ({df['source'].value_counts().to_dict()})")
    print(f"unique teams:   {df['team'].nunique()}")
    print()
    print("by position:")
    print(df["position"].value_counts().to_string())
    print()
    print("price range by position:")
    print(df.groupby("position")["price"].agg(["min", "median", "max"]).to_string())
    print()
    print("=== top 10 by ownership % (pre-tournament) ===")
    top_own = df.nlargest(10, "ownership_pct")[
        ["name", "team", "position", "price", "ownership_pct"]
    ]
    print(top_own.to_string(index=False))
    print()
    print("=== top 5 priciest per position ===")
    for pos in ["GK", "DEF", "MID", "FWD"]:
        print(f"\n--- {pos} ---")
        sub = df[df["position"] == pos].nlargest(5, "price")[
            ["name", "team", "price", "ownership_pct"]
        ]
        print(sub.to_string(index=False))
