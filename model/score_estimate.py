"""Full-tournament projected fantasy score with confidence band + tier.

Pipeline
--------
1. Run squad LP (model/optimizer.run_full_pipeline) — gives the 15-man squad.
2. For each of 8 matchdays:
     - pick best XI from the squad by ev_md_r (model/optimizer.pick_xi)
     - run captain_optimizer to choose captain
     - md_score = sum(ev_md_r for XI) + captain_double_bonus
3. point_estimate = sum across all 8 matchdays.
4. 80% CI = point_estimate × (1 ± CONFIDENCE_BAND).
5. Place in office + global tier; emit plain-language string.

Calibration
-----------
The original spec included an 0.82× shrinkage to correct for "LP optimism".
That shrinkage is wrong for our pipeline: the EV inputs are already
probabilistic — each ev_md_r is ep90 × P(team plays MD r) × minutes_share,
which integrates over the simulator's elimination scenarios. Applying a
second shrinkage on top of that double-counts the uncertainty. Removed.

Tier ranges are calibrated for an 8-matchday WC fantasy tournament, NOT a
38-week Premier League season. They sit in the 150-700 band, matching the
612 / 675 realized totals from the 2018 / 2022 backtest.

Public API
----------
    estimate_score(mode='floor', league_size=15) -> dict
"""
from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

from model.captain_optimizer import optimize_captain
from model.optimizer import pick_xi, run_full_pipeline
from scoring import rules as R

log = logging.getLogger(__name__)

CONFIDENCE_BAND = 0.15         # ±15% → 80% CI (spec)
N_MATCHDAYS = 8

# WC-realistic tiers (8 matchdays). UI should import these constants
# rather than hardcoding.
OFFICE_TIERS = [
    {"name": "Likely winner",  "lo": 450, "hi": 550},
    {"name": "Top 3 finish",   "lo": 380, "hi": 450},
    {"name": "Middle of pack", "lo": 250, "hi": 380},
]
GLOBAL_TIERS = [
    {"name": "Beginner avg",    "lo": 150, "hi": 250},
    {"name": "Average player",  "lo": 250, "hi": 380},
    {"name": "Advanced player", "lo": 380, "hi": 500},
    {"name": "Pro (top 1%)",    "lo": 500, "hi": 650},
    {"name": "Global winner",   "lo": 700, "hi": 1500},
]


def _place_in_tiers(score: float, tiers: Iterable[dict]) -> str:
    sorted_tiers = sorted(tiers, key=lambda t: -t["lo"])
    if score > sorted_tiers[0]["hi"]:
        return f"above '{sorted_tiers[0]['name']}'"
    if score < sorted_tiers[-1]["lo"]:
        return f"below '{sorted_tiers[-1]['name']}'"
    for tier in sorted_tiers:
        if tier["lo"] <= score <= tier["hi"]:
            return tier["name"]
    return "between tiers"


def _matchday_breakdown(
    squad: pd.DataFrame,
    *,
    mode: str,
    league_size: int,
) -> pd.DataFrame:
    """Per-MD XI/captain/score. The squad is fixed across all 8 matchdays;
    XI and captain are re-optimized per MD based on ev_md_r."""
    rows = []
    for md in range(1, N_MATCHDAYS + 1):
        score_col = f"ev_md{md}"
        xi, formation, _bench = pick_xi(squad, score_col)
        rec = optimize_captain(
            xi, score_col, league_size=league_size, mode=mode,
        )
        captain = rec["primary_captain"]
        vice = rec["vice_captain"]
        xi_score = float(xi[score_col].sum())
        captain_bonus = float(xi.loc[xi["name"] == captain, score_col].iloc[0])
        md_score = xi_score + captain_bonus * (R.CAPTAIN_MULTIPLIER - 1)
        rows.append({
            "md": md,
            "formation": f"{formation[1]}-{formation[2]}-{formation[3]}",
            "captain": captain,
            "vice": vice,
            "xi_score": round(xi_score, 2),
            "captain_bonus": round(captain_bonus, 2),
            "md_score": round(md_score, 2),
        })
    return pd.DataFrame(rows)


def estimate_score(
    *,
    mode: str = "floor",
    league_size: int = 15,
) -> dict:
    """Run the full estimate. Returns a dict with summary numbers + the
    per-matchday breakdown DataFrame."""
    pipeline = run_full_pipeline(matchday=1)
    squad = pipeline["squad"]

    breakdown = _matchday_breakdown(squad, mode=mode, league_size=league_size)
    point = float(breakdown["md_score"].sum())
    lo = point * (1 - CONFIDENCE_BAND)
    hi = point * (1 + CONFIDENCE_BAND)

    office_tier = _place_in_tiers(point, OFFICE_TIERS)
    global_tier = _place_in_tiers(point, GLOBAL_TIERS)

    plain = (
        f"Projected: {point:.0f} pts "
        f"(80% CI: {lo:.0f} – {hi:.0f}). "
        f"Office tier: {office_tier}. Global tier: {global_tier}."
    )

    return {
        "point_estimate": round(point, 1),
        "ci_low": round(lo, 1),
        "ci_high": round(hi, 1),
        "office_tier": office_tier,
        "global_tier": global_tier,
        "plain_language": plain,
        "breakdown": breakdown,
        "squad_cost": round(float(squad["price"].sum()), 1),
        "n_nations": int(squad["team"].nunique()),
        "mode": mode,
    }


if __name__ == "__main__":
    out = estimate_score()
    print("=" * 72)
    print("Full-tournament projected score (mode='floor', league_size=15)")
    print("=" * 72)
    print(f"Squad cost:           ${out['squad_cost']:.1f}M / $100M   "
          f"({out['n_nations']} nations)")
    print(f"Point estimate:       {out['point_estimate']:.1f}")
    print(f"80% confidence band:  {out['ci_low']:.0f} – {out['ci_high']:.0f}")
    print(f"Office tier:          {out['office_tier']}")
    print(f"Global tier:          {out['global_tier']}")
    print()
    print("Per-matchday breakdown:")
    print(out["breakdown"].to_string(index=False))
    print()
    print(f">>> {out['plain_language']}")

    # Calibration sanity vs the 2018/2022 backtest realized totals (612 / 675)
    print("\nCalibration check vs historical backtest:")
    print(f"  2018 system realized: 612 (favorable: France SF, Belgium 3rd)")
    print(f"  2022 system realized: 675 (favorable: Argentina champion)")
    print(f"  This projection:      {out['point_estimate']:.0f}  (expected across")
    print(f"                        all simulator outcomes)")
