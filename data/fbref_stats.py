"""FBref club-season stats.

Production path: load from static CSVs shipped in the repo at
`data/fbref_cache/big5_<season>.csv` (outfield) and
`big5_<season>_gk.csv`. Last-season stats don't change during the WC, so
the deployed app doesn't need to invoke `soccerdata` / chromedriver at all.

Refresh path (local-only): run `python -m data.fbref_stats refresh` to
re-scrape from FBref via soccerdata and rewrite the static CSVs. Do this
once before pushing changes to GitHub if a new club season has completed.

The original soccerdata pipeline used the Big 5 leagues' standard + shooting +
misc tables for outfielders and the keeper table for goalkeepers, merged
on (league, season, team, player). Limitations of the player-season feeds:

Soccerdata 1.9.0 limitation
---------------------------
At the player-season level, `read_player_season_stats` only accepts
['standard', 'keeper', 'shooting', 'playing_time', 'misc']. Passing and
defense tables aren't wrapped, so we don't get npxG, xAG, key passes, or
total tackles. Two consequences:

* npxG and xAG aren't available. The projection formula in model/projections.py
  treats per90_non_pen_goals as a noisy proxy for npxG_per90 and per90_assists
  as a noisy proxy for xAG_per90 — defensible for a tournament with a small
  number of matches per player, where realized recent form is similarly
  predictive to xG-based projections.
* Tackles uses TklW (tackles won) from `misc`; close to total tackles for top
  players but slightly biased low.

We DO get PKwon / PKcon at the season level from `misc`, which feed directly
into the scoring constants PEN_WON / PEN_CONCEDED.

Public API
----------
    fetch_outfield_stats(season=DEFAULT_SEASON) -> pd.DataFrame
    fetch_gk_stats(season=DEFAULT_SEASON) -> pd.DataFrame
    stats_for_player(name, df=None) -> dict | None  # convenience for verification

Players outside the Big 5 (Saudi PL, MLS, Süper Lig, South America) must be
supplied via data/manual_players.csv overrides (step 10).
"""
from __future__ import annotations

import logging
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR, TTL_FBREF
from data.cache import cache_get, cache_set

log = logging.getLogger(__name__)

DEFAULT_LEAGUES = "Big 5 European Leagues Combined"
DEFAULT_SEASON = "2025-2026"
_INDEX_COLS = ["league", "season", "team", "player"]

# Static CSV cache shipped in the repo. The deployed app reads from these
# directly — no soccerdata / chromedriver dependency in production.
STATIC_CACHE_DIR = DATA_DIR / "fbref_cache"


def _csv_path(season: str, gk: bool) -> Path:
    slug = season.replace("-", "_")
    suffix = "_gk" if gk else ""
    return STATIC_CACHE_DIR / f"big5_{slug}{suffix}.csv"


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case + ASCII-safe flatten of FBref's 2-level column MultiIndex."""
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        flat: list[str] = []
        for top, sub in out.columns.to_flat_index():
            joined = f"{top}_{sub}".strip("_").lower()
            for old, new in [
                (" ", "_"), ("+", "p"), ("-", "_"), ("%", "pct"), ("/", "_per_"),
            ]:
                joined = joined.replace(old, new)
            flat.append(joined)
        out.columns = flat
    return out.reset_index()


def _pick(df: pd.DataFrame, keep: dict[str, str]) -> pd.DataFrame:
    """Subset to columns listed in `keep` (src→dst). Missing src cols become NA."""
    out = pd.DataFrame(index=df.index)
    for src, dst in keep.items():
        out[dst] = df[src] if src in df.columns else pd.NA
    return out


def _read_stat_type(season: str, stat_type: str) -> pd.DataFrame:
    """Lazy-imports soccerdata so the deployed app (which never hits this
    path) doesn't carry the chromedriver dependency at import time."""
    import soccerdata as sd
    fbref = sd.FBref(leagues=DEFAULT_LEAGUES, seasons=season)
    return _flatten(fbref.read_player_season_stats(stat_type=stat_type))


def _scrape_outfield(season: str) -> pd.DataFrame:
    """Re-scrape outfield stats via soccerdata. Used by refresh_static_cache."""
    log.info("Scraping outfield stats for %s via soccerdata (~1-3 min)", season)
    std = _read_stat_type(season, "standard")
    sho = _read_stat_type(season, "shooting")
    msc = _read_stat_type(season, "misc")

    std_pick = _pick(std, {
        **{c: c for c in _INDEX_COLS},
        "nation": "nation",
        "pos": "pos",
        "age": "age",
        "playing_time_mp": "mp",
        "playing_time_starts": "starts",
        "playing_time_min": "minutes",
        "playing_time_90s": "nineties",
        "performance_gls": "goals",
        "performance_ast": "assists",
        "performance_g_pk": "non_pen_goals",
        "performance_pk": "pens_scored",
        "performance_pkatt": "pens_attempted",
        "performance_crdy": "yellow_cards",
        "performance_crdr": "red_cards",
        "per_90_minutes_gls": "per90_goals",
        "per_90_minutes_ast": "per90_assists",
        "per_90_minutes_g_pk": "per90_non_pen_goals",
    })
    sho_pick = _pick(sho, {
        **{c: c for c in _INDEX_COLS},
        "standard_sh": "shots",
        "standard_sot": "sot",
        "standard_sh_per_90": "per90_shots",
        "standard_sot_per_90": "per90_sot",
        "standard_sotpct": "sot_pct",
    })
    msc_pick = _pick(msc, {
        **{c: c for c in _INDEX_COLS},
        "performance_tklw": "tackles_won",
        "performance_int": "interceptions",
        "performance_pkwon": "pens_won",
        "performance_pkcon": "pens_conceded",
        "performance_fls": "fouls_committed",
        "performance_fld": "fouls_drawn",
        "performance_og": "own_goals",
    })

    merged = (
        std_pick
        .merge(sho_pick, on=_INDEX_COLS, how="left")
        .merge(msc_pick, on=_INDEX_COLS, how="left")
    )

    # Derive per-90 for columns FBref doesn't pre-compute.
    nineties = pd.to_numeric(merged["nineties"], errors="coerce").replace(0, np.nan)
    for total_col, p90_col in [
        ("tackles_won", "per90_tackles_won"),
        ("interceptions", "per90_interceptions"),
        ("pens_won", "per90_pens_won"),
        ("pens_conceded", "per90_pens_conceded"),
    ]:
        merged[p90_col] = pd.to_numeric(merged[total_col], errors="coerce") / nineties

    return merged


def _scrape_gk(season: str) -> pd.DataFrame:
    """Re-scrape GK stats via soccerdata. Used by refresh_static_cache."""
    log.info("Scraping GK stats for %s via soccerdata (~30-60s)", season)
    gk = _read_stat_type(season, "keeper")

    gk_pick = _pick(gk, {
        **{c: c for c in _INDEX_COLS},
        "nation": "nation",
        "pos": "pos",
        "age": "age",
        "playing_time_mp": "mp",
        "playing_time_starts": "starts",
        "playing_time_min": "minutes",
        # In the keeper table FBref renders 90s as a top-level column with an
        # empty sub-level — flat name is just "90s", not "playing_time_90s".
        "90s": "nineties",
        "performance_ga": "goals_against",
        "performance_ga90": "ga_per90",
        "performance_sota": "shots_on_target_against",
        "performance_saves": "saves",
        "performance_savepct": "save_pct",
        "performance_cs": "clean_sheets",
        "performance_cspct": "cs_pct",
        "penalty_kicks_pkatt": "pks_faced",
        "penalty_kicks_pka": "pks_allowed",
        "penalty_kicks_pksv": "pen_saves",
        "penalty_kicks_pkm": "pks_missed_by_opp",
    })
    nineties = pd.to_numeric(gk_pick["nineties"], errors="coerce").replace(0, np.nan)
    gk_pick["per90_saves"] = pd.to_numeric(gk_pick["saves"], errors="coerce") / nineties
    gk_pick["per90_cs"] = pd.to_numeric(gk_pick["clean_sheets"], errors="coerce") / nineties

    return gk_pick


# ---------------------------------------------------------------------------
# Public API: CSV-first loaders + refresh CLI
# ---------------------------------------------------------------------------

def _load_csv_or_scrape(season: str, gk: bool) -> pd.DataFrame:
    """Returns the requested stats frame. Tries CSV first, falls back to
    soccerdata scrape (with a warning), then writes the result back to CSV
    so future runs are fast and don't need chromedriver."""
    csv = _csv_path(season, gk)
    if csv.exists():
        return pd.read_csv(csv)
    log.warning(
        "%s missing — falling back to soccerdata scrape. "
        "Run `python -m data.fbref_stats refresh` to populate it.", csv
    )
    df = _scrape_gk(season) if gk else _scrape_outfield(season)
    csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)
    return df


def fetch_outfield_stats(season: str = DEFAULT_SEASON) -> pd.DataFrame:
    """One row per outfield player with per-90 metrics. Loads from the
    static CSV shipped in the repo; no chromedriver dependency in prod."""
    key = f"fbref:outfield:{DEFAULT_LEAGUES}:{season}"
    cached = cache_get(key, TTL_FBREF)
    if cached is not None:
        return cached
    df = _load_csv_or_scrape(season, gk=False)
    cache_set(key, df)
    return df


def fetch_gk_stats(season: str = DEFAULT_SEASON) -> pd.DataFrame:
    """One row per goalkeeper: saves, save%, CS, CS/90, GA, GA/90."""
    key = f"fbref:gk:{DEFAULT_LEAGUES}:{season}"
    cached = cache_get(key, TTL_FBREF)
    if cached is not None:
        return cached
    df = _load_csv_or_scrape(season, gk=True)
    cache_set(key, df)
    return df


def refresh_static_cache(season: str = DEFAULT_SEASON) -> None:
    """Local-only: re-scrape via soccerdata and rewrite the static CSVs.
    Requires Chrome / chromedriver on the host machine. Run before pushing
    when a new club season has completed."""
    STATIC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = _scrape_outfield(season)
    out_path = _csv_path(season, gk=False)
    out.to_csv(out_path, index=False)
    log.info("Wrote %d outfield rows to %s", len(out), out_path)

    gk = _scrape_gk(season)
    gk_path = _csv_path(season, gk=True)
    gk.to_csv(gk_path, index=False)
    log.info("Wrote %d GK rows to %s", len(gk), gk_path)


def _fold(s: str) -> str:
    """Strip accents + lowercase for accent-insensitive name matching."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def stats_for_player(name: str, df: pd.DataFrame | None = None) -> dict | None:
    """Accent-insensitive substring lookup. 'Vinícius' matches 'Vinicius Júnior';
    'Mbappe' matches 'Kylian Mbappé'. Returns first hit as a dict, or None."""
    if df is None:
        df = fetch_outfield_stats()
    needle = _fold(name)
    mask = df["player"].fillna("").map(lambda p: needle in _fold(p))
    hits = df[mask]
    if hits.empty:
        return None
    return hits.iloc[0].to_dict()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "refresh":
        logging.basicConfig(level=logging.INFO)
        refresh_static_cache()
        sys.exit(0)

    df = fetch_outfield_stats()
    print(f"outfield rows: {len(df)} | columns: {len(df.columns)}")
    csv = _csv_path(DEFAULT_SEASON, gk=False)
    print(f"loaded from: {csv}   exists={csv.exists()}")

    targets = ["Haaland", "Mbappé", "Bellingham", "Vinícius", "Yamal"]
    cols = [
        "team", "pos", "mp", "minutes", "goals", "assists",
        "per90_goals", "per90_assists", "per90_non_pen_goals",
        "per90_shots", "per90_sot", "per90_tackles_won",
        "pens_won", "pens_conceded",
    ]
    for name in targets:
        row = stats_for_player(name, df)
        if row is None:
            print(f"\n{name}: NOT FOUND")
            continue
        print(f"\n--- {row['player']} ---")
        for c in cols:
            val = row.get(c)
            if isinstance(val, float):
                print(f"  {c:<22} {val:.3f}")
            else:
                print(f"  {c:<22} {val}")
