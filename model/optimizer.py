"""Squad + XI optimizers (PuLP linear programs).

Squad LP (full tournament)
--------------------------
Pick 15 players maximizing sum(ev_total). Constraints:
    * Exactly 2 GK + 5 DEF + 5 MID + 3 FWD
    * Sum(price) ≤ BUDGET_GROUP (=100 by default)
    * ≤ NATION_CAP per country
The LP is exact and small (≈500 binary vars, a few hundred constraints) —
CBC solves it in <100 ms.

XI selector (single matchday)
-----------------------------
Given a 15-man squad and a per-matchday score column (e.g. `ev_md4`), pick
the 11 players maximizing total score, subject to one of seven valid
formations: 4-4-2, 4-3-3, 4-5-1, 3-4-3, 3-5-2, 5-4-1, 5-3-2.

We don't run an LP for the XI — there are only seven formations and within
each one the optimal pick is just `top-N-per-position`, so we enumerate.

Captain selector (placeholder)
------------------------------
For step 17 the captain is simply the highest-EV player in the XI. Step 18
adds ownership-aware captaincy logic (model/captain_optimizer.py).

Public API
----------
    optimize_squad(ev_df, budget=100.0, nation_cap=3) -> pd.DataFrame
    pick_xi(squad, score_col) -> (xi_df, formation, bench_df)
    pick_captain(xi_df, score_col) -> (captain_name, vice_name)
    run_full_pipeline(stage, matchday) -> dict  (verification helper)
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
import pulp

from config import BUDGET_GROUP, NATION_CAP, SQUAD_SPLIT

log = logging.getLogger(__name__)

VALID_FORMATIONS = [
    (1, 4, 4, 2),
    (1, 4, 3, 3),
    (1, 4, 5, 1),
    (1, 3, 4, 3),
    (1, 3, 5, 2),
    (1, 5, 4, 1),
    (1, 5, 3, 2),
]


def optimize_squad(
    ev_df: pd.DataFrame,
    *,
    budget: float = BUDGET_GROUP,
    nation_cap: int = NATION_CAP["GROUP"],
    objective_col: str = "ev_total",
) -> pd.DataFrame:
    """Squad LP: pick 15 players maximizing `objective_col` subject to
    budget + position split + nation cap. Returns the chosen rows.

    `ev_df` requires columns: name, team, position, price, plus
    `objective_col`."""
    df = ev_df.dropna(subset=["price", objective_col]).reset_index(drop=True)
    n = len(df)
    if n == 0:
        raise ValueError("empty ev_df")

    prob = pulp.LpProblem("squad", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]
    prob += pulp.lpSum(df.iloc[i][objective_col] * x[i] for i in range(n))

    for pos, required in SQUAD_SPLIT.items():
        prob += (
            pulp.lpSum(x[i] for i in range(n) if df.iloc[i]["position"] == pos)
            == required
        ), f"pos_{pos}"

    prob += (
        pulp.lpSum(df.iloc[i]["price"] * x[i] for i in range(n)) <= budget,
        "budget",
    )

    for nation in df["team"].unique():
        prob += (
            pulp.lpSum(x[i] for i in range(n) if df.iloc[i]["team"] == nation)
            <= nation_cap
        ), f"nation_{nation}"

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"squad LP infeasible/non-optimal: {pulp.LpStatus[status]}")

    chosen = [i for i in range(n) if x[i].value() and x[i].value() > 0.5]
    return df.iloc[chosen].reset_index(drop=True)


def pick_xi(
    squad: pd.DataFrame, score_col: str
) -> tuple[pd.DataFrame, tuple[int, int, int, int], pd.DataFrame]:
    """Best XI from a 15-player squad. Enumerates seven valid formations,
    within each picks top-N per position by `score_col`. Returns
    (xi_df, formation, bench_df) where bench_df is squad − xi sorted by
    score_col descending (first row is the auto-sub / 12th-man candidate)."""
    if len(squad) != 15:
        raise ValueError(f"squad has {len(squad)} players, need 15")
    sorted_pos = {
        pos: squad[squad["position"] == pos]
                  .sort_values(score_col, ascending=False)
                  .reset_index(drop=True)
        for pos in ("GK", "DEF", "MID", "FWD")
    }
    best_xi = None
    best_form = None
    best_total = -np.inf
    for form in VALID_FORMATIONS:
        n_gk, n_def, n_mid, n_fwd = form
        if (len(sorted_pos["GK"]) < n_gk or
                len(sorted_pos["DEF"]) < n_def or
                len(sorted_pos["MID"]) < n_mid or
                len(sorted_pos["FWD"]) < n_fwd):
            continue
        xi = pd.concat([
            sorted_pos["GK"].head(n_gk),
            sorted_pos["DEF"].head(n_def),
            sorted_pos["MID"].head(n_mid),
            sorted_pos["FWD"].head(n_fwd),
        ])
        total = xi[score_col].sum()
        if total > best_total:
            best_total = total
            best_xi = xi
            best_form = form
    if best_xi is None:
        raise RuntimeError("no valid formation could be formed from this squad")
    bench = squad[~squad["name"].isin(best_xi["name"])].sort_values(
        score_col, ascending=False
    ).reset_index(drop=True)
    return best_xi.reset_index(drop=True), best_form, bench


def pick_captain(xi: pd.DataFrame, score_col: str) -> tuple[str, str]:
    """Placeholder: highest score → captain; second highest → vice. Step 18
    replaces this with ownership-aware logic."""
    ordered = xi.sort_values(score_col, ascending=False)
    return ordered.iloc[0]["name"], ordered.iloc[1]["name"]


def run_full_pipeline(matchday: int = 1) -> dict:
    """Convenience: build EV table → optimize squad → pick XI → pick captain.
    Returns intermediate artifacts for inspection in the UI / __main__."""
    # Import lazily to avoid forcing every consumer to pull the full graph
    from model.ev import compute_ev
    from model.projections import compute_ep90
    from model.tournament_sim import simulate

    proj = compute_ep90()
    sim = simulate()
    ev = compute_ev(proj, sim)

    squad = optimize_squad(ev)
    score_col = f"ev_md{matchday}"
    xi, form, bench = pick_xi(squad, score_col)
    captain, vice = pick_captain(xi, score_col)

    return {
        "ev": ev,
        "sim": sim,
        "squad": squad,
        "xi": xi,
        "formation": form,
        "bench": bench,
        "captain": captain,
        "vice": vice,
        "matchday": matchday,
    }


if __name__ == "__main__":
    out = run_full_pipeline(matchday=1)
    squad = out["squad"]
    xi = out["xi"]
    bench = out["bench"]
    form = out["formation"]
    matchday = out["matchday"]

    fmt_str = f"{form[1]}-{form[2]}-{form[3]}"
    cost = squad["price"].sum()
    n_nations = squad["team"].nunique()

    print(f"=== Optimal 15-man squad (budget ${cost:.1f}M / $100M, {n_nations} nations) ===")
    cols = ["name", "team", "position", "price", "ep90", "ev_total"]
    print(squad.sort_values("ev_total", ascending=False)[cols].to_string(index=False))

    print(f"\n=== Starting XI for MD{matchday}  (formation {fmt_str}) ===")
    print(f"     Captain:  {out['captain']}")
    print(f"     Vice:     {out['vice']}")
    xi_cols = ["name", "team", "position", "price", "ep90", f"ev_md{matchday}"]
    xi_print = xi.copy()
    xi_print["role"] = xi_print["name"].map(
        lambda n: "(C)" if n == out["captain"] else "(VC)" if n == out["vice"] else ""
    )
    print(xi_print[xi_cols + ["role"]].sort_values(f"ev_md{matchday}", ascending=False)
          .to_string(index=False))

    print(f"\n=== Bench (sub priority order) ===")
    print(bench[xi_cols].to_string(index=False))

    # Headline numbers
    xi_score = xi[f"ev_md{matchday}"].sum()
    captain_bonus = xi.loc[xi["name"] == out["captain"], f"ev_md{matchday}"].iloc[0]
    print(f"\nMD{matchday} expected XI EV:   {xi_score:.2f}")
    print(f"MD{matchday} captain bonus:   +{captain_bonus:.2f}")
    print(f"MD{matchday} predicted score: {xi_score + captain_bonus:.2f}")
    print(f"\nFull-tournament projected total (squad sum of ev_total): "
          f"{squad['ev_total'].sum():.1f}")
