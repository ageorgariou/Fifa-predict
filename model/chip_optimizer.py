"""Chip-sequence optimizer.

For each remaining chip, score the EV lift from playing it in each of the
remaining matchdays and recommend a timeline.

Chips
-----
* Wildcard — full squad re-pick under that round's per-nation cap. EV lift
  is the gap between the round-optimal squad's matchday EV and our current
  squad's matchday EV. Biggest at R16 (cap goes 3 → 4 — a structural
  change) and at the start of new stages with fixture swings.
* 12th Man — turns the bench's top sub into an 11th-counting score for the
  round. EV lift ≈ EV of "best player not currently in our XI for that MD"
  (because we'd auto-sub them in anyway, but they'd replace a 0-from-a-DNP
  bench-line and now they count regardless).
* Maximum Captain — captain × 3 instead of × 2. EV lift = captain's MD EV
  (i.e. one extra captain bonus). Best on a high-variance MD with multiple
  candidates within 0.5 EV (a high ceiling), and NEVER on MD1 — at MD1 the
  field already triples up on Kane/Mbappé/Yamal, so the chip mostly raises
  the floor instead of moving you up the table.
* Qualification Booster (R32+ rounds) — +2 to every XI player whose nation
  advances past that round. EV lift = sum over XI of 2 × P(team advances).
  Best where XI has the highest aggregate progression probability.
* Mystery Booster — unrevealed. Always reported as "HOLD until revealed".

Output
------
recommend_chip_sequence(...) returns a list of dicts shaped like:
    {
        "matchday": int (1..8) or None (for Mystery / Hold),
        "chip": str,
        "expected_lift": float (in fantasy points),
        "reasoning": str,
        "extra": dict (e.g. {"best_alt_player": "..."} for 12th Man),
    }

Failure modes
-------------
* If `sim_summary` is missing per-team round probabilities, Qualification
  Booster falls back to a flat 0.5 for every team and logs a warning.
* If the EV table doesn't cover every player needed for the wildcard
  re-optimization, that round's wildcard lift is reported as 0 and a
  warning is logged. The recommender still completes.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from config import BUDGET_GROUP, BUDGET_R32, NATION_CAP
from model.optimizer import optimize_squad, pick_xi
from model.schedule import STAGE_BY_MD

log = logging.getLogger(__name__)

# Round-progression probability column names (from tournament_sim).
# E.g. for MD4 (R32) the relevant prob is "p_r16" — the chance the team
# survives past R32 and plays MD5.
NEXT_ROUND_PROB_COL = {
    4: "p_r16",   # boost applies if team advances out of R32
    5: "p_qf",    # advance out of R16
    6: "p_sf",    # advance out of QF
    7: "p_f",     # advance out of SF
    8: "p_win",   # advance past Final / win
}

ALL_CHIPS = ["Wildcard", "12th Man", "Maximum Captain",
             "Qualification Booster", "Mystery Booster"]
HIGH_VAR_TIEBREAK = 0.5     # see captain_optimizer.TIEBREAK_EV_WINDOW
KNOCKOUT_MDS = (4, 5, 6, 7, 8)
QUAL_BOOSTER_BONUS = 2.0    # +2 per XI player whose team advances

# --- Recommendation thresholds (added 2026-05) ----------------------------
# Wildcard, 12th Man, and Maximum Captain are season-defining one-shots: a
# single +30-50 pt window is worth far more than the sum of "fine" +5-10 pt
# uses. So we refuse to spend them on weak slots and prefer HOLD instead.
#
# MD1 is treated even more conservatively because everyone has full chip
# inventory, the field's captain choices are obvious (Kane/Mbappé/Yamal),
# and there's a full 7 matchdays of better options ahead.
HOLD_CHIPS = {"Wildcard", "12th Man", "Maximum Captain"}
MIN_LIFT_DEFAULT = 15.0
MIN_LIFT_MD1 = 20.0
N_ALTERNATIVES_TO_SHOW = 3

# Late-MD bias: 12th Man and Maximum Captain have higher ceilings in
# knockout rounds (deeper bench rotations + variance-friendly elimination
# matches), so we add a small per-MD lift to the *ranking* score. The
# reported `expected_lift` stays the raw model number — the bias only
# affects which MD is chosen.
LATE_MD_BIAS_CHIPS = {"12th Man", "Maximum Captain"}
LATE_MD_BIAS_PER_MD = 1.5


def _min_lift_for(md: int) -> float:
    return MIN_LIFT_MD1 if md == 1 else MIN_LIFT_DEFAULT


def _wildcard_lift_for_md(
    ev: pd.DataFrame,
    current_squad: pd.DataFrame,
    matchday: int,
) -> tuple[float, pd.DataFrame | None]:
    """EV lift of re-optimizing the squad under this MD's nation cap, vs
    keeping the current 15. Returns (lift, new_squad)."""
    md_col = f"ev_md{matchday}"
    stage = STAGE_BY_MD[matchday]
    budget = BUDGET_R32 if matchday >= 4 else BUDGET_GROUP
    nation_cap = NATION_CAP[stage]
    try:
        new_squad = optimize_squad(
            ev, budget=budget, nation_cap=nation_cap, objective_col=md_col,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("wildcard re-opt failed for MD%d: %s", matchday, exc)
        return 0.0, None

    new_xi, _, _ = pick_xi(new_squad, md_col)
    cur_xi, _, _ = pick_xi(current_squad, md_col)
    lift = float(new_xi[md_col].sum()) - float(cur_xi[md_col].sum())
    return max(0.0, round(lift, 2)), new_squad


def _twelfth_man_lift_for_md(
    ev: pd.DataFrame,
    current_squad: pd.DataFrame,
    matchday: int,
) -> tuple[float, str | None]:
    """EV lift = best non-squad player's MD EV (we'd pretend to add them
    as a 12th XI counter). Returns (lift, best_alt_name)."""
    md_col = f"ev_md{matchday}"
    squad_keys = set(zip(current_squad["name"], current_squad["team"]))
    out_of_squad = ev[
        ~ev.apply(lambda r: (r["name"], r["team"]) in squad_keys, axis=1)
    ]
    if out_of_squad.empty:
        return 0.0, None
    top = out_of_squad.sort_values(md_col, ascending=False).iloc[0]
    return round(float(top[md_col]), 2), str(top["name"])


def _max_captain_lift_for_md(
    current_squad: pd.DataFrame,
    matchday: int,
) -> tuple[float, int]:
    """EV lift = top XI player's MD EV (one extra captaincy bonus).
    Returns (lift, n_high_var_candidates) so we can score "variance" too."""
    md_col = f"ev_md{matchday}"
    xi, _, _ = pick_xi(current_squad, md_col)
    sorted_xi = xi.sort_values(md_col, ascending=False).reset_index(drop=True)
    if sorted_xi.empty:
        return 0.0, 0
    top_ev = float(sorted_xi.iloc[0][md_col])
    n_close = int((sorted_xi[md_col] >= top_ev - HIGH_VAR_TIEBREAK).sum())
    return round(top_ev, 2), n_close


def _qual_booster_lift_for_md(
    current_squad: pd.DataFrame,
    sim_summary: pd.DataFrame,
    matchday: int,
) -> float:
    """+2 per XI player whose nation progresses past this round.
    Only meaningful for knockout MDs (4-8)."""
    if matchday not in NEXT_ROUND_PROB_COL:
        return 0.0
    md_col = f"ev_md{matchday}"
    xi, _, _ = pick_xi(current_squad, md_col)
    prob_col = NEXT_ROUND_PROB_COL[matchday]
    if prob_col not in sim_summary.columns:
        log.warning("sim_summary missing %s; falling back to 0.5", prob_col)
        progress = {t: 0.5 for t in xi["team"].unique()}
    else:
        progress = sim_summary.set_index("team")[prob_col].to_dict()
    total = sum(QUAL_BOOSTER_BONUS * float(progress.get(t, 0.0))
                for t in xi["team"])
    return round(total, 2)


def recommend_chip_sequence(
    ev: pd.DataFrame,
    squad: pd.DataFrame,
    sim_summary: pd.DataFrame,
    *,
    mode: str = "floor",
    league_size: int = 15,
    chips_available: Optional[list[str]] = None,
    matchdays: Optional[list[int]] = None,
) -> list[dict]:
    """Return a chip plan covering every remaining matchday.

    Greedy assignment: for each chip we score every (chip, md) pair and
    pick the highest-lift slot, removing both the chip and the matchday
    from the candidate set. This is not optimal in a strict combinatorial
    sense but is robust to noise and matches how a human plans chips."""
    if chips_available is None:
        chips_available = list(ALL_CHIPS)
    if matchdays is None:
        matchdays = list(range(1, 9))

    plan: list[dict] = []
    md_used: set[int] = set()

    # Pre-score every (chip, md) combination so we can do greedy selection.
    # `lift` is the raw model number; `score` is `lift` plus the late-MD
    # bias when applicable, and is what we sort by.
    scores: dict[tuple[str, int], dict] = {}
    for md in matchdays:
        # Wildcard
        lift, _ = _wildcard_lift_for_md(ev, squad, md)
        scores[("Wildcard", md)] = {
            "lift": lift, "score": lift, "extra": {},
            "base_reasoning":
                f"re-optimization under nation cap "
                f"{NATION_CAP[STAGE_BY_MD[md]]} adds +{lift:.1f} pts",
        }

        # 12th Man — late-MD biased
        lift, alt = _twelfth_man_lift_for_md(ev, squad, md)
        scores[("12th Man", md)] = {
            "lift": lift,
            "score": lift + LATE_MD_BIAS_PER_MD * (md - 1),
            "base_reasoning": (
                f"best non-squad player is {alt or '(none)'} "
                f"(+{lift:.1f})"
            ),
            "extra": {"best_alt_player": alt},
        }

        # Max Captain — NEVER MD1, and late-MD biased
        if md == 1:
            scores[("Maximum Captain", md)] = {
                "lift": 0.0, "score": -1e9,    # ensure never picked
                "base_reasoning": (
                    "the field obviously triples Kane/Mbappé/Yamal; "
                    "the chip's edge collapses"
                ),
                "extra": {},
            }
        else:
            lift, n_close = _max_captain_lift_for_md(squad, md)
            var_bonus = max(0, n_close - 1) * 0.5    # variance bonus
            scores[("Maximum Captain", md)] = {
                "lift": round(lift + var_bonus, 2),
                "score": round(lift + var_bonus
                                + LATE_MD_BIAS_PER_MD * (md - 1), 2),
                "base_reasoning": (
                    f"top XI EV = {lift:.1f} pts, "
                    f"{n_close} captain candidates within "
                    f"{HIGH_VAR_TIEBREAK} EV"
                ),
                "extra": {},
            }

        # Qualification Booster — only meaningful R32 onward
        if md in KNOCKOUT_MDS:
            lift = _qual_booster_lift_for_md(squad, sim_summary, md)
            scores[("Qualification Booster", md)] = {
                "lift": lift, "score": lift, "extra": {},
                "base_reasoning":
                    f"aggregate XI progression bonus = +{lift:.1f} pts",
            }
        else:
            scores[("Qualification Booster", md)] = {
                "lift": 0.0, "score": -1e9,
                "base_reasoning": "not available in group stage",
                "extra": {},
            }

    def _alt_str(chip: str, exclude_md: int | None = None,
                  exclude_used: bool = True) -> str:
        """Compact ranked list of alternative MDs for a chip's reasoning."""
        rows = []
        for md in matchdays:
            if md == exclude_md:
                continue
            if exclude_used and md in md_used:
                continue
            s = scores.get((chip, md))
            if s is None or s["score"] < -1e8:
                continue
            rows.append((md, s["lift"]))
        rows.sort(key=lambda kv: kv[1], reverse=True)
        rows = rows[:N_ALTERNATIVES_TO_SHOW]
        if not rows:
            return "(no viable alternatives)"
        return ", ".join(f"MD{md}: +{lift:.1f}" for md, lift in rows)

    # Greedy assignment per chip.
    #
    # Two-step rule for HOLD_CHIPS:
    #   1. Threshold gate uses RAW lift — at least one slot must clear
    #      the per-MD threshold (15, or 20 for MD1) for the chip to be
    #      considered for assignment.
    #   2. Among the slots that clear the gate, pick the one with the
    #      highest BIASED score. The bias rewards knockout rounds for
    #      chips whose ceiling is higher there (12th Man, Max Captain).
    #
    # Other chips (Qualification Booster) bypass the threshold and use
    # raw lift directly.
    for chip in chips_available:
        if chip == "Mystery Booster":
            plan.append({
                "matchday": None, "chip": chip, "expected_lift": 0.0,
                "reasoning": "Hold until revealed by FIFA.", "extra": {},
            })
            continue

        candidates = [
            (md, scores[(chip, md)])
            for md in matchdays
            if md not in md_used and (chip, md) in scores
            and scores[(chip, md)]["score"] > -1e8
        ]
        if not candidates:
            plan.append({
                "matchday": None, "chip": chip, "expected_lift": 0.0,
                "reasoning": "Hold — no viable slot remaining.", "extra": {},
            })
            continue

        if chip in HOLD_CHIPS:
            # Step 1: threshold gate on raw lift.
            viable = [
                (md, s) for md, s in candidates
                if s["lift"] >= _min_lift_for(md)
            ]
            if not viable:
                # No slot crosses the bar — HOLD, and report the best raw-lift
                # slot so the user can see how far short we are.
                best_md, best = max(candidates, key=lambda kv: kv[1]["lift"])
                threshold = _min_lift_for(best_md)
                alts = _alt_str(chip, exclude_md=best_md)
                plan.append({
                    "matchday": None,
                    "chip": chip,
                    "expected_lift": best["lift"],
                    "reasoning": (
                        f"Hold — best slot is MD{best_md} at only "
                        f"+{best['lift']:.1f} pts (need ≥{threshold:.0f} pts; "
                        f"{best['base_reasoning']}). Other slots: {alts}."
                    ),
                    "extra": best.get("extra", {}),
                })
                continue
            # Step 2: among viable slots, pick highest BIASED score.
            best_md, best = max(viable, key=lambda kv: kv[1]["score"])
        else:
            # Non-hold chips (Qualification Booster) just take max raw lift.
            best_md, best = max(candidates, key=lambda kv: kv[1]["lift"])

        alts = _alt_str(chip, exclude_md=best_md)
        plan.append({
            "matchday": best_md,
            "chip": chip,
            "expected_lift": best["lift"],
            "reasoning": (
                f"MD{best_md} is the highest-lift slot at "
                f"+{best['lift']:.1f} pts ({best['base_reasoning']}). "
                f"Alternatives considered: {alts}."
            ),
            "extra": best.get("extra", {}),
        })
        md_used.add(best_md)

    # Sort by matchday (HOLDs / Mystery last).
    plan.sort(key=lambda p: (p["matchday"] is None, p["matchday"] or 99))
    return plan


if __name__ == "__main__":
    import logging as _log
    _log.basicConfig(level=_log.INFO)
    from model.ev import compute_ev
    from model.projections import compute_ep90
    from model.tournament_sim import simulate

    proj = compute_ep90()
    sim = simulate()
    ev = compute_ev(proj, sim)
    squad = optimize_squad(ev)
    plan = recommend_chip_sequence(ev=ev, squad=squad, sim_summary=sim)

    print("=== Chip plan ===")
    for p in plan:
        md = f"MD{p['matchday']}" if p["matchday"] else "—"
        print(f"  {md:>5}  {p['chip']:<22} +{p['expected_lift']:.1f} pts  "
              f"{p['reasoning']}")
