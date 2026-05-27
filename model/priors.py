"""Per-player prior multipliers applied to stat-derived per-90 projections.

Five priors per spec:

* set_piece_taker_bonus  = 1.30 — FK/corner taker (data/set_piece_takers.json)
* penalty_taker_bonus    = 1.25 — designated PK taker (data/penalty_takers.json)
* host_nation_bonus      = 1.10 — USA/CAN/MEX during group stage only
* elite_team_defender    = 1.15 — DEF/GK whose national team has
                                   bookmaker decimal odds < 8.0 to win
                                   (≈ implied p_win > 12.5%)
* young_star_inflator    = 1.08 — under-23 AND ≥2700 club minutes last season

The set-piece and penalty lists are JSON arrays of {name, team} so the user
can edit them between deadlines without code changes.

Public API
----------
    apply_priors(players, stage='GROUP', book_probs=None) -> pd.Series
        Returns one combined multiplier per row of `players`.

    explain_priors(player_row, stage, book_probs) -> dict[str, bool]
        Same logic but returns which priors fired; for UI / debugging.

`players` must have columns: name, team, position, age (FBref 'YY-DDD' string
is fine), minutes (per-season club minutes).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Iterable

import pandas as pd

from config import DATA_DIR
from model.groups_2026 import HOST_NATIONS

log = logging.getLogger(__name__)

SET_PIECE_TAKER_BONUS = 1.30
PENALTY_TAKER_BONUS = 1.25
HOST_NATION_BONUS = 1.10
ELITE_TEAM_DEFENDER_BOOST = 1.15
YOUNG_STAR_INFLATOR = 1.08

ELITE_TEAM_MAX_DECIMAL_ODDS = 8.0      # equivalent to implied p_win > 12.5%
YOUNG_AGE_MAX_EXCLUSIVE = 23           # under-23
YOUNG_STAR_MIN_MINUTES = 2700          # ≈ 30 full club matches

PENALTY_TAKERS_PATH = DATA_DIR / "penalty_takers.json"
SET_PIECE_TAKERS_PATH = DATA_DIR / "set_piece_takers.json"

_AGE_RE = re.compile(r"^\s*(\d+)")


def _load_taker_list(path) -> set[tuple[str, str]]:
    """Returns {(name, team), ...}. Empty if file missing/corrupt."""
    if not path.exists():
        return set()
    try:
        items = json.loads(path.read_text())
    except json.JSONDecodeError:
        log.warning("%s is corrupt — treating as empty", path)
        return set()
    return {(it["name"], it["team"]) for it in items if "name" in it and "team" in it}


def _parse_age(value) -> int | None:
    """FBref ages are 'YY-DDD'; FIFA may pass a plain int. Both work."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int,)):
        return int(value)
    if isinstance(value, str):
        m = _AGE_RE.match(value)
        if m:
            return int(m.group(1))
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_elite_team(team: str, book_probs: dict[str, float] | None) -> bool:
    if not book_probs:
        return False
    p = book_probs.get(team)
    if p is None or p <= 0:
        return False
    return (1.0 / p) < ELITE_TEAM_MAX_DECIMAL_ODDS


def _is_young_star(age, minutes) -> bool:
    a = _parse_age(age)
    if a is None or a >= YOUNG_AGE_MAX_EXCLUSIVE:
        return False
    try:
        m = float(minutes)
    except (TypeError, ValueError):
        return False
    if pd.isna(m):
        return False
    return m >= YOUNG_STAR_MIN_MINUTES


def explain_priors(
    name: str,
    team: str,
    position: str,
    age,
    minutes,
    *,
    stage: str = "GROUP",
    book_probs: dict[str, float] | None = None,
    pk_takers: set[tuple[str, str]] | None = None,
    sp_takers: set[tuple[str, str]] | None = None,
) -> dict[str, bool]:
    """Boolean explanation of which priors fire for one player."""
    if pk_takers is None:
        pk_takers = _load_taker_list(PENALTY_TAKERS_PATH)
    if sp_takers is None:
        sp_takers = _load_taker_list(SET_PIECE_TAKERS_PATH)
    return {
        "set_piece_taker": (name, team) in sp_takers,
        "penalty_taker": (name, team) in pk_takers,
        "host_group": stage == "GROUP" and team in HOST_NATIONS,
        "elite_defender": position in ("DEF", "GK") and _is_elite_team(team, book_probs),
        "young_star": _is_young_star(age, minutes),
    }


def _combine(flags: dict[str, bool]) -> float:
    m = 1.0
    if flags["set_piece_taker"]:
        m *= SET_PIECE_TAKER_BONUS
    if flags["penalty_taker"]:
        m *= PENALTY_TAKER_BONUS
    if flags["host_group"]:
        m *= HOST_NATION_BONUS
    if flags["elite_defender"]:
        m *= ELITE_TEAM_DEFENDER_BOOST
    if flags["young_star"]:
        m *= YOUNG_STAR_INFLATOR
    return m


def apply_priors(
    players: pd.DataFrame,
    *,
    stage: str = "GROUP",
    book_probs: dict[str, float] | None = None,
) -> pd.Series:
    """Combined prior multiplier per row of `players`.

    Required columns: name, team, position, age, minutes."""
    pk_takers = _load_taker_list(PENALTY_TAKERS_PATH)
    sp_takers = _load_taker_list(SET_PIECE_TAKERS_PATH)

    out = []
    for _, r in players.iterrows():
        flags = explain_priors(
            name=r.get("name"),
            team=r.get("team"),
            position=r.get("position"),
            age=r.get("age"),
            minutes=r.get("minutes"),
            stage=stage,
            book_probs=book_probs,
            pk_takers=pk_takers,
            sp_takers=sp_takers,
        )
        out.append(_combine(flags))
    return pd.Series(out, index=players.index, name="prior_multiplier")


if __name__ == "__main__":
    from data.fbref_stats import fetch_outfield_stats, stats_for_player
    from data.fifa_players import fetch_player_list
    from data.odds import fetch_winner_probs

    fifa = fetch_player_list()
    fbref = fetch_outfield_stats()
    book = fetch_winner_probs()

    pk_takers = _load_taker_list(PENALTY_TAKERS_PATH)
    sp_takers = _load_taker_list(SET_PIECE_TAKERS_PATH)
    print(f"loaded {len(pk_takers)} penalty takers, {len(sp_takers)} set-piece takers")
    print(f"elite-team threshold: decimal odds < {ELITE_TEAM_MAX_DECIMAL_ODDS}  "
          f"(implied p_win > {1/ELITE_TEAM_MAX_DECIMAL_ODDS:.3f})")
    elite = [t for t, p in book.items() if p > 1.0 / ELITE_TEAM_MAX_DECIMAL_ODDS]
    print(f"elite teams under this rule: {sorted(elite)}")
    print()

    targets = [
        ("Erling Haaland", "Norway"),
        ("Kylian Mbappé", "France"),
        ("Jude Bellingham", "England"),
        ("Vinícius Júnior", "Brazil"),
        ("Lamine Yamal", "Spain"),
        ("William Saliba", "France"),  # elite-team defender example
        ("Christian Pulisic", "United States"),  # host-nation example
    ]
    print(f"{'Player':<22} {'Team':<14} {'Pos':<4} {'Age':<5} {'Mins':>6}  "
          f"{'Multiplier':>10}  Flags")
    for name, team in targets:
        fifa_row = fifa[(fifa["name"] == name) & (fifa["team"] == team)]
        if fifa_row.empty:
            print(f"{name:<22} {team:<14}  NOT FOUND IN FIFA LIST")
            continue
        position = fifa_row["position"].iloc[0]
        fb_row = stats_for_player(name, fbref)
        age = fb_row.get("age") if fb_row else None
        minutes = fb_row.get("minutes") if fb_row else None
        flags = explain_priors(
            name, team, position, age, minutes,
            stage="GROUP", book_probs=book,
            pk_takers=pk_takers, sp_takers=sp_takers,
        )
        mult = _combine(flags)
        fired = [k for k, v in flags.items() if v]
        age_str = str(_parse_age(age)) if age is not None else "?"
        mins_str = f"{minutes:.0f}" if isinstance(minutes, (int, float)) and not pd.isna(minutes) else "?"
        print(f"{name:<22} {team:<14} {position:<4} {age_str:<5} {mins_str:>6}  "
              f"{mult:>10.4f}  {', '.join(fired) or '—'}")
