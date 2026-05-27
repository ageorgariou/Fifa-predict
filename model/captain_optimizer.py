"""Ownership-aware captain choice.

The captain double is the single highest-variance decision in FIFA Fantasy:
+8 pts of expected captain bonus per matchday × 8 matchdays = ~64 pts at
stake for the season — comparable to the gap between a winning office-
league squad and an average one. So picking the captain by raw EV alone is
NOT enough; you have to think about *what your opponents will captain too*.

Algorithm (per spec)
--------------------
1. Model the field of `n_sharp_opponents` "sharp" managers. Without explicit
   opponent captain info, assume each sharp manager captains:
      60% — the highest ep_round player in their XI (top-1)
      30% — the second-highest (top-2)
      10% — the third-highest (top-3)
   With opponent_captains provided, use the empirical distribution.

2. For each candidate captain c in our XI, Monte-Carlo their match score
   from a Gamma distribution with mean = ep_round and σ = 0.7 × ep_round
   (right-skewed to match observed fantasy scoring).

3. For each sim, also draw "a random opponent's captain score" from the
   weighted field distribution.

4. P(beat field) = P(our captain score > opponent's captain score). Note
   the captain doubling factor cancels on both sides.

5. Expected rank gain = n_sharp_opponents × (P(beat) − 0.5). Positive means
   "expected to move up the table"; negative means "expected to slide".

6. Recommend the captain with the highest expected rank gain. The
   reasoning text explains whether this is a Floor pick (highest-EV AND
   consensus) or a Ceiling pick (contrarian / similar-EV).

Public API
----------
    optimize_captain(xi, score_col, league_size=15, ...) -> dict
        Returns {primary_captain, vice_captain, captain_reasoning,
                 captain_ev_table}.
"""
from __future__ import annotations

from collections import Counter
import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FIELD_CAPTAIN_WEIGHTS = [0.60, 0.30, 0.10]   # spec model on top-3 XI EV
SIGMA_RATIO = 0.7                              # σ / mean for the score distribution

# Ceiling-mode parameters:
CEILING_OWNERSHIP_MAX = 0.30                   # contrarian if est_ownership < this
CEILING_EV_THRESHOLD = 0.85                    # candidate EV must be ≥ this × top EV


def _sample_score(rng: np.random.Generator, ep: float, n_sims: int) -> np.ndarray:
    """Gamma(mean=ep, σ=SIGMA_RATIO·ep), shape inferred from those moments.
    Gamma is right-skewed which matches real fantasy outcomes (most matches
    around the mean, occasional double-figure hauls)."""
    ep = max(ep, 1e-4)
    sigma = SIGMA_RATIO * ep
    theta = sigma * sigma / ep       # var / mean
    k = ep / theta                   # shape
    return rng.gamma(k, theta, size=n_sims)


def optimize_captain(
    xi: pd.DataFrame,
    score_col: str,
    *,
    league_size: int = 15,
    n_sharp_opponents: int | None = None,
    opponent_captains: list[str] | None = None,
    mode: str = "floor",
    n_sims: int = 10_000,
    seed: int = 42,
) -> dict:
    """See module docstring."""
    if "name" not in xi.columns or score_col not in xi.columns:
        raise ValueError("xi must have 'name' and score_col columns")
    n_sharp = n_sharp_opponents if n_sharp_opponents is not None else league_size - 1
    n_sharp = max(1, n_sharp)

    rng = np.random.default_rng(seed)
    xi_sorted = xi.sort_values(score_col, ascending=False).reset_index(drop=True)

    # Pre-sample each XI player's score
    name_to_ep = dict(zip(xi_sorted["name"], xi_sorted[score_col]))
    name_to_samples = {n: _sample_score(rng, float(ep), n_sims)
                       for n, ep in name_to_ep.items()}

    # Build field captain distribution
    if opponent_captains:
        counts = Counter(opponent_captains)
        total = sum(counts.values())
        field_names = list(counts.keys())
        field_weights = [counts[n] / total for n in field_names]
    else:
        n_field = min(3, len(xi_sorted))
        field_names = xi_sorted["name"].head(n_field).tolist()
        weights = FIELD_CAPTAIN_WEIGHTS[:n_field]
        wsum = sum(weights)
        field_weights = [w / wsum for w in weights]

    # Per-sim: which field captain did "the average opponent" choose?
    field_choice = rng.choice(len(field_names), size=n_sims, p=field_weights)
    field_samples = np.zeros(n_sims)
    for i, name in enumerate(field_names):
        mask = (field_choice == i)
        # If a named opponent captain isn't in our XI (e.g. user named a
        # different team's player), we fall back to a low-EP draw so that
        # information still degrades sensibly.
        if name in name_to_samples:
            field_samples[mask] = name_to_samples[name][mask]
        else:
            field_samples[mask] = _sample_score(rng, 4.0, mask.sum())

    rows = []
    for _, p in xi_sorted.iterrows():
        c_name = p["name"]
        c_samples = name_to_samples[c_name]
        wins = (c_samples > field_samples).sum()
        ties = (c_samples == field_samples).sum()
        p_beat = (wins + 0.5 * ties) / n_sims

        est_own = 0.0
        if c_name in field_names:
            est_own = field_weights[field_names.index(c_name)]

        rank_gain = n_sharp * (p_beat - 0.5)
        rows.append({
            "name": c_name,
            "ep_round": round(float(p[score_col]), 3),
            "est_ownership": round(est_own, 3),
            "p_beat_field": round(float(p_beat), 4),
            "expected_rank_gain": round(float(rank_gain), 3),
        })

    table = pd.DataFrame(rows)
    top_ep_value = float(table["ep_round"].max())

    # Primary selection depends on mode.
    if mode == "ceiling":
        # Among candidates whose EV is within 15% of the top, prefer the most
        # contrarian (lowest est_ownership). Tie-break by expected_rank_gain.
        eligible = table[
            (table["ep_round"] >= CEILING_EV_THRESHOLD * top_ep_value)
            & (table["est_ownership"] < CEILING_OWNERSHIP_MAX)
        ]
        if not eligible.empty:
            primary = eligible.sort_values(
                ["est_ownership", "expected_rank_gain"],
                ascending=[True, False],
            ).iloc[0]["name"]
        else:
            primary = table.loc[table["expected_rank_gain"].idxmax(), "name"]
    else:  # floor (default)
        primary = table.loc[table["expected_rank_gain"].idxmax(), "name"]

    primary_row = table[table["name"] == primary].iloc[0]
    others = table[table["name"] != primary]
    vice = others.loc[others["expected_rank_gain"].idxmax(), "name"]

    top_ep_name = xi_sorted.iloc[0]["name"]
    top_ep = float(xi_sorted.iloc[0][score_col])
    top_own = float(table[table["name"] == top_ep_name]["est_ownership"].iloc[0])

    if primary == top_ep_name and primary_row["est_ownership"] >= 0.5:
        reasoning = (
            f"{primary} is both highest-EV ({primary_row['ep_round']:.2f}) "
            f"and the consensus captain (~{primary_row['est_ownership']*100:.0f}% "
            f"of sharp managers). Floor logic: no rank loss vs the field; "
            f"P(beat field) = {primary_row['p_beat_field']*100:.1f}%."
        )
    elif primary == top_ep_name:
        reasoning = (
            f"{primary} has the highest EV ({primary_row['ep_round']:.2f}) and "
            f"is moderately owned (~{primary_row['est_ownership']*100:.0f}%). "
            f"Expected rank gain: {primary_row['expected_rank_gain']:+.2f}."
        )
    else:
        ep_gap = primary_row["ep_round"] - top_ep
        own_gap = primary_row["est_ownership"] - top_own
        gap_dir = "below" if ep_gap < 0 else "above"
        reasoning = (
            f"{primary} is a contrarian pick over {top_ep_name} "
            f"(EV {primary_row['ep_round']:.2f} vs {top_ep:.2f} — "
            f"{abs(ep_gap):.2f} {gap_dir}; "
            f"{primary_row['est_ownership']*100:.0f}% vs {top_own*100:.0f}% owned). "
            f"Expected rank gain: {primary_row['expected_rank_gain']:+.2f}."
        )

    return {
        "primary_captain": primary,
        "vice_captain": vice,
        "captain_reasoning": reasoning,
        "captain_ev_table": table.sort_values("expected_rank_gain", ascending=False)
                                  .to_dict(orient="records"),
    }


if __name__ == "__main__":
    # Verification: pick captain for MD1 XI from the live optimizer pipeline.
    from model.optimizer import run_full_pipeline

    pipeline = run_full_pipeline(matchday=1)
    xi = pipeline["xi"]
    md = pipeline["matchday"]
    score_col = f"ev_md{md}"

    print(f"=== Captain optimization for MD{md} ===\n")
    print(f"XI ep_round candidates (top-5):")
    print(xi.sort_values(score_col, ascending=False).head()[["name", "team", "position", score_col]].to_string(index=False))

    # --- Scenario 1: small office league (15 players, all sharp) ---
    print("\n\n--- Scenario A: 15-player office league, all sharps, no opponent info ---")
    rec = optimize_captain(xi, score_col, league_size=15)
    print(f"Primary:    {rec['primary_captain']}")
    print(f"Vice:       {rec['vice_captain']}")
    print(f"Reasoning:  {rec['captain_reasoning']}")
    print("\ncaptain_ev_table:")
    table = pd.DataFrame(rec["captain_ev_table"])
    print(table.to_string(index=False))

    # --- Scenario 2: larger league (100 sharps) — does the contrarian get any more attractive? ---
    print("\n\n--- Scenario B: 100-player large league, all sharps ---")
    rec_b = optimize_captain(xi, score_col, league_size=100)
    print(f"Primary:    {rec_b['primary_captain']}")
    print(f"Reasoning:  {rec_b['captain_reasoning']}")
    table_b = pd.DataFrame(rec_b["captain_ev_table"])
    print(table_b.head(5).to_string(index=False))

    # --- Scenario 3: explicit opponent info ("everyone in my league captains Kane") ---
    print("\n\n--- Scenario C: 14 opponents all captain Harry Kane (Floor mode) ---")
    opp = ["Harry Kane"] * 14
    rec_c = optimize_captain(xi, score_col, league_size=15, opponent_captains=opp)
    print(f"Primary:    {rec_c['primary_captain']}")
    print(f"Reasoning:  {rec_c['captain_reasoning']}")
    table_c = pd.DataFrame(rec_c["captain_ev_table"])
    print(table_c.head(5).to_string(index=False))

    # --- Scenario 4: same as C but with Ceiling mode ---
    print("\n\n--- Scenario D: same as C but mode='ceiling' ---")
    rec_d = optimize_captain(
        xi, score_col, league_size=15, opponent_captains=opp, mode="ceiling",
    )
    print(f"Primary:    {rec_d['primary_captain']}")
    print(f"Reasoning:  {rec_d['captain_reasoning']}")
    table_d = pd.DataFrame(rec_d["captain_ev_table"])
    print(table_d.head(5).to_string(index=False))
