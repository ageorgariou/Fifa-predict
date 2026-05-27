"""Stat-derived per-90 expected fantasy points (ep90).

Spec formula (with substitutions for the soccerdata data gaps documented in
data/fbref_stats.py):

    ep90 =  +2                                          # appearance (60+ mins)
          + per90_non_pen_goals × goal_points[pos]      # spec: npxG_per90
          + per90_assists × ASSIST_POINTS                # spec: xA_per90
          + cs_prob_match × cs_points[pos]               # from sim
          + per90_saves / GK_SAVES_PER_BONUS             # 0 for outfielders
          + per90_tackles_won / MID_TACKLES_PER_BONUS
          + per90_sot / FWD_SOT_PER_BONUS
          − card_expectation
          − goals_conceded_penalty                       # DEF/GK only
    ep90 *= prior_multiplier

Notes
-----
* `per90_non_pen_goals` proxies `npxG_per90` (FBref ships per-season goals but
  not npxG via the soccerdata wrapper). For top WC players in a small-sample
  tournament, realized recent form is similarly predictive.
* `per90_assists` proxies `xA_per90` for the same reason.
* `key_passes / 2` term is dropped (no source for key passes via soccerdata).
* `goals_conceded_penalty` uses a calibration based on the team's CS prob
  (more CS → lower expected GA penalty). See _ga_penalty.
* GKs come from a separate FBref table; this module merges both pools.

Public API
----------
    compute_ep90(stage='GROUP') -> pd.DataFrame
        One row per fantasy-eligible matched player with ep90 + intermediate
        components. Sorted descending by ep90.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from data.fbref_stats import fetch_gk_stats, fetch_outfield_stats
from data.fifa_players import fetch_player_list
from data.name_matcher import match_fifa_to_fbref
from data.odds import fetch_winner_probs
from model.priors import apply_priors
from model.tournament_sim import simulate
from scoring import rules as R

log = logging.getLogger(__name__)


GOAL_POINTS = R.GOAL_POINTS
ASSIST_POINTS = R.MID_ASSIST  # 3 for all positions
CS_POINTS = R.CLEAN_SHEET_POINTS

MIN_NINETIES = 5.0  # exclude club sample sizes under ~450 mins — per-90 is noise


def _safe(x, default=0.0) -> float:
    """Coerce to float, mapping NaN/None to default."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if np.isnan(v):
        return default
    return v


def _ga_penalty(cs_prob: float) -> float:
    """Expected per-match GA penalty in fantasy points for GK/DEF.

    Spec scoring: -1 per 2 goals conceded after the 1st = floor(GA/2) points.
    For Poisson GA with realistic WC rates (μ ≈ 1.0–1.6 vs typical opponents),
    E[floor(GA/2)] is well-approximated by 0.5 * P(GA ≥ 2) ≈ 0.5 * (1 - cs_prob)
    because most non-CS matches concede 1-2 goals. This is exact enough at
    the per-match level and avoids requiring a per-team μ from the simulator."""
    return 0.5 * (1.0 - cs_prob)


def _build_player_pool() -> pd.DataFrame:
    """Match FIFA outfielders → FBref outfield AND FIFA GKs → FBref keeper.
    Returns one DataFrame keyed on (fifa_name, fifa_team) with stats columns."""
    fifa = fetch_player_list()
    fbref_out = fetch_outfield_stats()
    fbref_gk = fetch_gk_stats()

    fifa_out = fifa[fifa["position"].isin(["DEF", "MID", "FWD"])]
    fifa_gk = fifa[fifa["position"] == "GK"]

    matched_out, _, _ = match_fifa_to_fbref(fifa_out, fbref_out, write_csvs=False)
    matched_gk, _, _ = match_fifa_to_fbref(fifa_gk, fbref_gk, write_csvs=False)

    if not matched_out.empty:
        matched_out = matched_out.copy()
        matched_out["pool"] = "outfield"
    if not matched_gk.empty:
        matched_gk = matched_gk.copy()
        matched_gk["pool"] = "gk"

    combined = pd.concat([matched_out, matched_gk], ignore_index=True)
    return combined


def compute_ep90(
    stage: str = "GROUP",
    n_sims: int | None = None,
) -> pd.DataFrame:
    """Compute ep90 for every matched FIFA player. See module docstring."""
    pool = _build_player_pool()
    book = fetch_winner_probs()
    sim = simulate(n_sims=n_sims) if n_sims else simulate()
    cs_by_team = (sim.set_index("team")["expected_clean_sheets"]
                  / sim.set_index("team")["expected_matches"]).to_dict()

    # Priors operate on (name, team, position, age, minutes)
    pool_for_priors = pool.rename(columns={
        "fifa_name": "name", "fifa_team": "team", "fifa_position": "position",
    })
    multiplier = apply_priors(pool_for_priors, stage=stage, book_probs=book)

    rows = []
    for i, r in pool.iterrows():
        pos = r["fifa_position"]
        team = r["fifa_team"]
        nineties = _safe(r.get("nineties"))
        if nineties < MIN_NINETIES:
            continue  # too small a sample — per-90 stats would be pure noise

        per90_goals_np = _safe(r.get("per90_non_pen_goals"))
        per90_assists_v = _safe(r.get("per90_assists"))
        per90_sot = _safe(r.get("per90_sot"))
        per90_tackles = _safe(r.get("per90_tackles_won"))
        per90_saves = _safe(r.get("per90_saves"))    # only nonzero for GKs

        # Card expectation: -1 per yellow, -2 per red, per 90 of play
        yellows = _safe(r.get("yellow_cards"))
        reds = _safe(r.get("red_cards"))
        cards_p90 = (yellows + 2.0 * reds) / nineties

        cs_prob = cs_by_team.get(team, 0.0)
        ga_penalty = _ga_penalty(cs_prob) if pos in ("GK", "DEF") else 0.0

        goal_pts = GOAL_POINTS.get(pos, 0)
        cs_pts = CS_POINTS.get(pos, 0)

        ep90 = (
            R.APPEARANCE_60_PLUS                                 # +2
            + per90_goals_np * goal_pts
            + per90_assists_v * ASSIST_POINTS
            + cs_prob * cs_pts
            + (per90_saves / R.GK_SAVES_PER_BONUS) * R.GK_SAVE_BONUS
            + (per90_tackles / R.MID_TACKLES_PER_BONUS) * R.MID_TACKLE_BONUS
            + (per90_sot / R.FWD_SOT_PER_BONUS) * R.FWD_SOT_BONUS
            - cards_p90
            - ga_penalty
        )
        ep90_priored = ep90 * multiplier.iloc[i]

        rows.append({
            "name": r["fifa_name"],
            "team": team,
            "position": pos,
            "price": r["fifa_price"],
            "ownership_pct": r["fifa_ownership_pct"],
            "club": r.get("fbref_club"),
            "minutes": r.get("minutes"),
            "cs_prob": round(cs_prob, 3),
            "ep90_raw": round(ep90, 3),
            "prior_mult": round(float(multiplier.iloc[i]), 3),
            "ep90": round(ep90_priored, 3),
        })

    df = pd.DataFrame(rows)
    return df.sort_values("ep90", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    df = compute_ep90(stage="GROUP")
    print(f"projections for {len(df)} matched players")
    print()
    print("=== top 30 by ep90 (group stage priors) ===")
    print(df.head(30).to_string(index=False))
    print()
    # Position breakdown of the top 30
    print("top-30 position breakdown:")
    print(df.head(30)["position"].value_counts().to_string())
