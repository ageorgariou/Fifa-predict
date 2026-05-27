"""Bookmaker-anchored projection blend.

For each player × upcoming match, the spec specifies:

    ep_from_goal_book    = p_score_book × goal_points[position]
    ep_from_goal_stat    = per_match_npxG × goal_points[position]
    ep_from_goal_blend   = 0.6 × ep_from_goal_book + 0.4 × ep_from_goal_stat
    ep_match             = ep_from_goal_blend + ep_other_stat

The blend is applied ONLY to the goal component. Assists, clean sheets,
tackles, saves, shots on target, cards — all the other ep components stay
purely stat-derived (bookmakers don't reliably price those).

If the bookmaker has no anytime-goalscorer line for a player in a match
(typical for matches >7 days out, fringe squad players, or markets not yet
open), the blend collapses to stat-only and the player's row carries
`low_confidence=True`.

Public API
----------
    blend_match(stat_ep90, book_p_score, position, opponent_cs_prob,
                minutes_share=1.0) -> dict
        Single (player, match) blend. Returns dict with ep_match, components,
        and low_confidence flag.

    blend_event_for_players(player_pool, event_id, sport_key) -> pd.DataFrame
        Convenience for verification — looks up book probs for the given
        event, blends every matched player in `player_pool` whose national
        team plays in that event.
"""
from __future__ import annotations

import logging

import pandas as pd

from data.goalscorer_odds import fetch_event_goalscorer_probs
from model.groups_2026 import _fold
from scoring import rules as R

log = logging.getLogger(__name__)

BLEND_BOOK_WEIGHT = 0.6
BLEND_STAT_WEIGHT = 0.4
assert abs(BLEND_BOOK_WEIGHT + BLEND_STAT_WEIGHT - 1.0) < 1e-9


def _goal_component_stat(per_match_np_goals: float, position: str) -> float:
    return per_match_np_goals * R.GOAL_POINTS.get(position, 0)


def _goal_component_book(p_score: float, position: str) -> float:
    return p_score * R.GOAL_POINTS.get(position, 0)


def blend_match(
    *,
    position: str,
    per_match_np_goals: float,
    per_match_assists: float,
    per_match_sot: float,
    per_match_tackles: float,
    per_match_saves: float,
    per_match_cards_pts: float,
    opponent_cs_prob: float,
    book_p_score: float | None,
) -> dict:
    """Compute ep_match for one player × one match.

    All per_match_* inputs are stat-derived per-match expectations (i.e.
    per-90 multiplied by an assumed minutes_share). Caller controls the
    minutes_share; we just sum the contributions."""
    stat_goal = _goal_component_stat(per_match_np_goals, position)
    if book_p_score is None:
        blended_goal = stat_goal
        low_confidence = True
    else:
        book_goal = _goal_component_book(book_p_score, position)
        blended_goal = BLEND_BOOK_WEIGHT * book_goal + BLEND_STAT_WEIGHT * stat_goal
        low_confidence = False

    cs_pts = opponent_cs_prob * R.CLEAN_SHEET_POINTS.get(position, 0)
    assist_pts = per_match_assists * R.MID_ASSIST  # 3 for all positions
    sot_pts = (per_match_sot / R.FWD_SOT_PER_BONUS) * R.FWD_SOT_BONUS
    tackle_pts = (per_match_tackles / R.MID_TACKLES_PER_BONUS) * R.MID_TACKLE_BONUS
    save_pts = (per_match_saves / R.GK_SAVES_PER_BONUS) * R.GK_SAVE_BONUS

    # GA penalty for GK/DEF: approximated from CS prob (see projections._ga_penalty)
    ga_penalty = 0.5 * (1.0 - opponent_cs_prob) if position in ("GK", "DEF") else 0.0

    ep_match = (
        R.APPEARANCE_60_PLUS
        + blended_goal
        + assist_pts
        + cs_pts
        + save_pts
        + tackle_pts
        + sot_pts
        - per_match_cards_pts
        - ga_penalty
    )

    return {
        "ep_match": ep_match,
        "ep_from_goal_blend": blended_goal,
        "ep_from_goal_stat": stat_goal,
        "ep_from_goal_book": _goal_component_book(book_p_score, position) if book_p_score is not None else None,
        "low_confidence": low_confidence,
    }


def blend_event_for_players(
    player_pool: pd.DataFrame,
    event_id: str,
    sport_key: str,
    home_team: str | None = None,
    away_team: str | None = None,
) -> pd.DataFrame:
    """Verification helper. Takes a player_pool DataFrame (columns include
    name, team, position, per90_non_pen_goals, per90_assists, per90_sot,
    per90_tackles_won, per90_saves, yellow_cards, red_cards, nineties),
    looks up bookmaker p_score for each player in (sport_key, event_id),
    and returns per-player blended goal-component for that match.

    If `home_team`/`away_team` are provided, the pool is restricted to those
    teams. Useful when running on the CL final test fixture where the pool
    is from FBref clubs (Real Madrid, Arsenal, etc.) rather than national
    teams — pass the club names through `team`."""
    book = fetch_event_goalscorer_probs(sport_key, event_id)
    # Build a folded-name index for tolerant lookup
    book_folded = {_fold(name): p for name, p in book.items()}

    rows = []
    pool = player_pool
    if home_team is not None or away_team is not None:
        keep = {t for t in (home_team, away_team) if t}
        pool = pool[pool["team"].isin(keep)]

    for _, r in pool.iterrows():
        position = r.get("position") or r.get("fifa_position")
        if position not in ("GK", "DEF", "MID", "FWD"):
            continue
        name = r.get("name") or r.get("fifa_name") or r.get("player")
        if not isinstance(name, str):
            continue
        p_score = book_folded.get(_fold(name))

        # Convert per-90 to per-match (assume full 90 mins for verification)
        per_match_np_goals = float(r.get("per90_non_pen_goals") or 0)
        per_match_assists = float(r.get("per90_assists") or 0)
        per_match_sot = float(r.get("per90_sot") or 0)
        per_match_tackles = float(r.get("per90_tackles_won") or 0)
        per_match_saves = float(r.get("per90_saves") or 0)
        nineties = float(r.get("nineties") or 0) or 1.0
        cards = (float(r.get("yellow_cards") or 0)
                 + 2.0 * float(r.get("red_cards") or 0)) / nineties

        blend = blend_match(
            position=position,
            per_match_np_goals=per_match_np_goals,
            per_match_assists=per_match_assists,
            per_match_sot=per_match_sot,
            per_match_tackles=per_match_tackles,
            per_match_saves=per_match_saves,
            per_match_cards_pts=cards,
            opponent_cs_prob=0.30,   # placeholder; downstream uses sim output
            book_p_score=p_score,
        )
        rows.append({
            "name": name,
            "team": r.get("team"),
            "position": position,
            "book_p_score": p_score,
            "stat_goal_pts": round(blend["ep_from_goal_stat"], 3),
            "book_goal_pts": round(blend["ep_from_goal_book"], 3)
                              if blend["ep_from_goal_book"] is not None else None,
            "blended_goal_pts": round(blend["ep_from_goal_blend"], 3),
            "ep_match": round(blend["ep_match"], 3),
            "low_confidence": blend["low_confidence"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("ep_match", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    # ---- Verification on the CL Final (PSG vs Arsenal, 2026-05-30) ----
    # WC goalscorer markets aren't live yet (~15 days from MD1). The blend
    # logic itself is dataset-agnostic, so we exercise it on a real fixture
    # where books ARE pricing player props.
    from data.fbref_stats import fetch_outfield_stats

    fbref = fetch_outfield_stats()
    # Restrict to PSG + Arsenal squads for the CL final
    cl_event_id = "a54f22aca3be31d95f13eac0aeac62cf"
    pool = fbref[fbref["team"].isin(["Paris Saint-Germain", "Arsenal"])].copy()
    # Default to FBref position character; FIFA Fantasy is what we'd use in WC
    pool["position"] = pool["pos"].str.split(",").str[0].map({
        "GK": "GK", "DF": "DEF", "MF": "MID", "FW": "FWD",
    })
    pool = pool[pool["position"].notna()]

    result = blend_event_for_players(
        pool, cl_event_id, "soccer_uefa_champs_league",
        home_team="Paris Saint-Germain", away_team="Arsenal",
    )
    print(f"=== CL Final blend ({len(result)} players) ===")
    cols = ["name", "team", "position", "book_p_score",
            "stat_goal_pts", "book_goal_pts", "blended_goal_pts",
            "ep_match", "low_confidence"]
    print(result[cols].head(20).to_string(index=False))

    # Comparison: book-priced vs not
    priced = result[result["low_confidence"] == False]
    not_priced = result[result["low_confidence"] == True]
    print(f"\nplayers priced by bookmaker:    {len(priced)}")
    print(f"players falling back to stat:   {len(not_priced)}")

    # How much does the blend reorder the goal component vs stat-only?
    if not priced.empty:
        priced = priced.copy()
        priced["delta_goal"] = priced["blended_goal_pts"] - priced["stat_goal_pts"]
        movers = priced.reindex(priced["delta_goal"].abs().sort_values(ascending=False).index)
        print("\n=== top 10 goal-component movers from blend ===")
        print(movers[["name", "team", "position", "stat_goal_pts",
                      "book_goal_pts", "blended_goal_pts", "delta_goal"]]
              .head(10).to_string(index=False))

    # ---- Spec-requested comparison: top players by blend vs stat-only ----
    # `ep_match` already uses the blended goal component; reconstruct a
    # stat-only ep_match by adding stat_goal_pts back to (ep_match − blended).
    full = result.copy()
    full["ep_match_stat_only"] = (
        full["ep_match"] - full["blended_goal_pts"] + full["stat_goal_pts"]
    )
    full["blend_rank"] = full["ep_match"].rank(ascending=False, method="min").astype(int)
    full["stat_rank"] = full["ep_match_stat_only"].rank(ascending=False, method="min").astype(int)
    full["rank_delta"] = full["blend_rank"] - full["stat_rank"]
    union = full[(full["blend_rank"] <= 15) | (full["stat_rank"] <= 15)]
    print("\n=== ep_match: blended vs stat-only (top 15 union) ===")
    print(union.sort_values("ep_match", ascending=False)[
        ["name", "team", "position",
         "ep_match", "ep_match_stat_only",
         "blend_rank", "stat_rank", "rank_delta", "low_confidence"]
    ].to_string(index=False))
