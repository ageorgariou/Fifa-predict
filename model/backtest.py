"""Retroactive validation of the projection model against 2018 + 2022 WC data.

Pipeline
--------
For each historical tournament we:
1. Load data/historical/wc{2018,2022}.csv (name, team, position, matches_played,
   goals, assists, minutes, final_fantasy_points).
2. Fetch FBref Big-5 club stats from the season *immediately preceding* the
   tournament (2017-2018 → 2018 WC, 2021-2022 → 2022 WC).
3. For each player, match (name, national_team) against the historical FBref
   pool. Players who played outside Big-5 that season are dropped.
4. Compute ep90 via the spec formula in model/projections.py — but WITHOUT
   priors (we don't have 2018/2022 PK/SP-taker lists or simulator output).
   cs_prob is a flat 0.30 placeholder (rough WC average).
5. Predict total fantasy points as `ep90 * (actual_minutes / 90)`. Using the
   *actual* minutes the player ended up playing isolates the question we care
   about: is ep90 well-calibrated as a per-90 estimate?

Outputs
-------
* Per-year Pearson r, MAE, n_players matched
* data/backtest_scatter.png with both years on one plot
* Top 20 over-projections and top 20 under-projections
* "VALIDATED" flag if stat-only r > 0.55 (spec threshold).

Limitations (called out in printout)
------------------------------------
* No odds-blended backtest. The-odds-api doesn't ship historical player-
  market odds on the free tier; we'd need a paid plan or a scraped archive.
* Priors aren't applied (no historical PK/SP/elite-team lists). So this
  backtest measures the stat-only baseline only — the deployed model is
  meant to outperform this with priors + odds blending.
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp
from rapidfuzz import fuzz, process

from config import DATA_DIR, HISTORICAL_DIR
from data.fbref_stats import fetch_outfield_stats, fetch_gk_stats
from model.groups_2026 import _fold, COUNTRY_CODE_TO_TEAM
from scoring import rules as R

log = logging.getLogger(__name__)

VALIDATION_THRESHOLD_R = 0.55
DEFAULT_CS_PROB = 0.30          # rough WC average; no per-team simulator here
SCATTER_PATH = DATA_DIR / "backtest_scatter.png"
MIN_NINETIES_HIST = 5.0         # same as projections.py filter

# Reverse lookup + extras for nations that played in 2018/2022 but not 2026.
_COUNTRY_NAME_TO_CODE: dict[str, str] = {
    name: code for code, name in COUNTRY_CODE_TO_TEAM.items()
}
_COUNTRY_NAME_TO_CODE.update({
    "Russia": "RUS",
    "Iceland": "ISL",
    "Peru": "PER",
    "Nigeria": "NGA",
    "Serbia": "SRB",
    "Poland": "POL",
    "Cameroon": "CMR",
    "Wales": "WAL",
    "Costa Rica": "CRC",
    "Denmark": "DEN",
    "Mauritania": "MTN",       # historical CSV may have stray entries
})

YEARS = {
    2018: {"csv": "wc2018.csv", "fbref_season": "2017-2018"},
    2022: {"csv": "wc2022.csv", "fbref_season": "2021-2022"},
}


def _ga_penalty(cs_prob: float) -> float:
    return 0.5 * (1.0 - cs_prob)


def _safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if np.isnan(v):
        return default
    return v


def _ep90_no_priors(row, position: str, cs_prob: float = DEFAULT_CS_PROB) -> float:
    """Project per-90 fantasy points for one player row (FBref outfield
    schema). No priors, fixed cs_prob — matches the projections.py formula
    minus the prior multiplier."""
    nineties = _safe_float(row.get("nineties"))
    if nineties < MIN_NINETIES_HIST:
        return float("nan")

    per90_goals_np = _safe_float(row.get("per90_non_pen_goals"))
    per90_assists = _safe_float(row.get("per90_assists"))
    per90_sot = _safe_float(row.get("per90_sot"))
    per90_tackles = _safe_float(row.get("per90_tackles_won"))
    per90_saves = _safe_float(row.get("per90_saves"))

    yellows = _safe_float(row.get("yellow_cards"))
    reds = _safe_float(row.get("red_cards"))
    cards_p90 = (yellows + 2.0 * reds) / nineties

    ga_penalty = _ga_penalty(cs_prob) if position in ("GK", "DEF") else 0.0
    goal_pts = R.GOAL_POINTS.get(position, 0)
    cs_pts = R.CLEAN_SHEET_POINTS.get(position, 0)

    return (
        R.APPEARANCE_60_PLUS
        + per90_goals_np * goal_pts
        + per90_assists * R.MID_ASSIST
        + cs_prob * cs_pts
        + (per90_saves / R.GK_SAVES_PER_BONUS) * R.GK_SAVE_BONUS
        + (per90_tackles / R.MID_TACKLES_PER_BONUS) * R.MID_TACKLE_BONUS
        + (per90_sot / R.FWD_SOT_PER_BONUS) * R.FWD_SOT_BONUS
        - cards_p90
        - ga_penalty
    )


def _match_historical_player(
    name: str, team: str, fbref_pool: pd.DataFrame
) -> pd.Series | None:
    """Fuzzy-match a historical CSV player (name, country) against a FBref
    season pool. Returns the best matching row, or None."""
    code = _COUNTRY_NAME_TO_CODE.get(team)
    if code is None:
        return None
    pool = fbref_pool[fbref_pool["nation"].astype(str).str.upper() == code]
    if pool.empty:
        return None
    choices = pool["player"].fillna("").tolist()
    best = process.extractOne(name, choices, scorer=fuzz.WRatio, processor=_fold)
    if best is None or best[1] < 85:
        return None
    return pool.iloc[best[2]]


def _backtest_year(year: int) -> pd.DataFrame:
    spec = YEARS[year]
    hist = pd.read_csv(HISTORICAL_DIR / spec["csv"])
    log.info("Loaded %s with %d players", spec["csv"], len(hist))

    fb_out = fetch_outfield_stats(season=spec["fbref_season"])
    fb_gk = fetch_gk_stats(season=spec["fbref_season"])

    rows = []
    for _, h in hist.iterrows():
        name, team, position = h["player"], h["team"], h["position"]
        actual = _safe_float(h["final_fantasy_points"])
        actual_mins = _safe_float(h["minutes"])
        if actual_mins <= 0:
            continue

        pool = fb_gk if position == "GK" else fb_out
        match = _match_historical_player(name, team, pool)
        if match is None:
            rows.append({
                "year": year, "name": name, "team": team, "position": position,
                "actual_pts": actual, "actual_mins": actual_mins,
                "matched": False, "predicted_pts": np.nan, "ep90": np.nan,
            })
            continue

        ep90 = _ep90_no_priors(match, position)
        if np.isnan(ep90):
            rows.append({
                "year": year, "name": name, "team": team, "position": position,
                "actual_pts": actual, "actual_mins": actual_mins,
                "matched": True, "predicted_pts": np.nan, "ep90": np.nan,
            })
            continue

        predicted = ep90 * (actual_mins / 90.0)
        rows.append({
            "year": year, "name": name, "team": team, "position": position,
            "actual_pts": actual, "actual_mins": actual_mins,
            "matched": True, "ep90": round(ep90, 3),
            "predicted_pts": round(predicted, 2),
            "fbref_player": match["player"], "fbref_club": match["team"],
        })

    return pd.DataFrame(rows)


def _pearson(actual, predicted) -> float:
    a = pd.Series(actual).astype(float)
    p = pd.Series(predicted).astype(float)
    mask = a.notna() & p.notna()
    if mask.sum() < 3:
        return float("nan")
    return float(a[mask].corr(p[mask]))


def _mae(actual, predicted) -> float:
    a = pd.Series(actual).astype(float)
    p = pd.Series(predicted).astype(float)
    mask = a.notna() & p.notna()
    if mask.sum() == 0:
        return float("nan")
    return float((a[mask] - p[mask]).abs().mean())


def _scatter(combined: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    for year, color in [(2018, "#2E86AB"), (2022, "#E63946")]:
        d = combined[(combined["year"] == year) & combined["predicted_pts"].notna()]
        ax.scatter(d["predicted_pts"], d["actual_pts"],
                   s=24, alpha=0.7, c=color, label=f"WC {year} (n={len(d)})",
                   edgecolors="white", linewidths=0.5)
    lo = 0
    hi = max(combined[["actual_pts", "predicted_pts"]].max().max(), 100)
    ax.plot([lo, hi], [lo, hi], "--", color="grey", alpha=0.4, label="y = x")
    ax.set_xlabel("Predicted total (stat-only, no priors)")
    ax.set_ylabel("Actual final fantasy points")
    ax.set_title("Backtest: 2018 + 2022 WC fantasy projections")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(SCATTER_PATH, dpi=120)
    plt.close(fig)


def run_backtest() -> dict:
    """Run the full backtest. Returns a summary dict and writes the scatter."""
    frames = [_backtest_year(y) for y in YEARS]
    combined = pd.concat(frames, ignore_index=True)

    summary = {}
    for year in YEARS:
        d = combined[combined["year"] == year]
        matched = d[d["matched"]]
        predicted = d[d["predicted_pts"].notna()]
        summary[year] = {
            "n_total": len(d),
            "n_matched": len(matched),
            "n_predicted": len(predicted),
            "pearson_r": _pearson(predicted["actual_pts"], predicted["predicted_pts"]),
            "mae": _mae(predicted["actual_pts"], predicted["predicted_pts"]),
        }
    # Pooled
    pred_all = combined[combined["predicted_pts"].notna()]
    summary["pooled"] = {
        "n_predicted": len(pred_all),
        "pearson_r": _pearson(pred_all["actual_pts"], pred_all["predicted_pts"]),
        "mae": _mae(pred_all["actual_pts"], pred_all["predicted_pts"]),
    }

    _scatter(combined)
    return {"summary": summary, "combined": combined}


# ---------------------------------------------------------------------------
# Squad optimization backtest
# ---------------------------------------------------------------------------
# Approximations (called out in printout):
#   * Prices: procedural from ep90 + position (no historical FIFA price data)
#   * Per-MD captain choice: uniform `total / matches_played` per round
#     (CSV only carries tournament totals, not per-matchday breakdowns)

_POSITION_PRICE_BOUNDS = {
    "FWD": (4.0, 10.5),
    "MID": (3.8, 10.0),
    "DEF": (3.5, 6.0),
    "GK":  (3.5, 5.0),
}


def estimate_price(ep90: float, position: str) -> float:
    """Map ep90 → fantasy price. ep90=3.0 → min for the position; ep90=8.0 →
    max. Linear interpolation, rounded to one decimal. Calibrated against
    the live 2026 FIFA Fantasy spreads."""
    lo, hi = _POSITION_PRICE_BOUNDS.get(position, (4.0, 10.0))
    norm = max(0.0, min(1.0, (ep90 - 3.0) / 5.0))
    return round(lo + (hi - lo) * norm, 1)


def optimize_squad(
    pool: pd.DataFrame,
    *,
    budget: float = 100.0,
    nation_cap: int = 3,
) -> pd.DataFrame:
    """LP: pick 15 players from `pool` maximizing sum(ep90), subject to
    2 GK + 5 DEF + 5 MID + 3 FWD, ≤ nation_cap per country, sum(price) ≤ budget.

    `pool` requires columns: name, team, position, ep90, price.
    Returns the selected 15 rows as a DataFrame."""
    pool = pool.reset_index(drop=True)
    n = len(pool)
    prob = pulp.LpProblem("squad_lp", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

    prob += pulp.lpSum(pool.iloc[i]["ep90"] * x[i] for i in range(n))

    pos_req = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    for pos, req in pos_req.items():
        prob += pulp.lpSum(x[i] for i in range(n) if pool.iloc[i]["position"] == pos) == req
    prob += pulp.lpSum(pool.iloc[i]["price"] * x[i] for i in range(n)) <= budget
    for nation in pool["team"].unique():
        prob += pulp.lpSum(x[i] for i in range(n) if pool.iloc[i]["team"] == nation) <= nation_cap

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"squad LP failed: {pulp.LpStatus[status]}")

    chosen = [i for i in range(n) if x[i].value() and x[i].value() > 0.5]
    return pool.iloc[chosen].reset_index(drop=True)


_FORMATIONS = [
    (1, 4, 4, 2), (1, 4, 3, 3), (1, 4, 5, 1),
    (1, 3, 4, 3), (1, 3, 5, 2), (1, 5, 4, 1), (1, 5, 3, 2),
]


def select_xi(squad: pd.DataFrame, sort_col: str) -> tuple[pd.DataFrame, tuple]:
    """For each valid formation, take top N per position by `sort_col`. Return
    the (XI, formation) that maximizes sum(sort_col)."""
    best_xi = None
    best_score = -np.inf
    best_form = None
    for form in _FORMATIONS:
        n_gk, n_def, n_mid, n_fwd = form
        parts = []
        for pos, need in (("GK", n_gk), ("DEF", n_def), ("MID", n_mid), ("FWD", n_fwd)):
            sub = squad[squad["position"] == pos].sort_values(sort_col, ascending=False)
            if len(sub) < need:
                parts = None
                break
            parts.append(sub.head(need))
        if parts is None:
            continue
        xi = pd.concat(parts)
        s = xi[sort_col].sum()
        if s > best_score:
            best_score = s
            best_xi = xi
            best_form = form
    return best_xi, best_form


# 2018 + 2022 each had 7 fantasy matchdays (3 group + R16 + QF + SF + F).
# 2026 has 8 (R32 added).
HISTORICAL_N_MATCHDAYS = 7


def per_matchday_breakdown(
    xi: pd.DataFrame,
    captain_name: str,
    n_matchdays: int = HISTORICAL_N_MATCHDAYS,
) -> pd.DataFrame:
    """Per-MD score approximation. For each MD: XI players who PLAYED in that
    MD contribute their per-MD average (total_pts / matches_played); captain
    is doubled. We have no per-MD data, so we attribute matches uniformly to
    the LAST n match-days the team played (e.g. a team that played 4 matches
    is assumed to have scored in MD4-MD7 in equal increments).
    """
    rows = []
    # Per-player per-MD average
    xi = xi.copy()
    xi["per_md"] = xi["actual_pts"] / xi["actual_mins"].clip(lower=1) * 90.0  # per 90 mins
    # Better: per-match average
    xi["per_match"] = xi["actual_pts"] / xi["matches_played"].clip(lower=1)

    for md in range(1, n_matchdays + 1):
        md_total = 0.0
        per_md_lines = []
        for _, p in xi.iterrows():
            # Approximate participation: assume player played in MDs 1..matches_played
            # (i.e. consecutive from MD1 until their team was eliminated).
            played = md <= int(p["matches_played"])
            if not played:
                continue
            contrib = p["per_match"]
            if p["name"] == captain_name:
                contrib *= R.CAPTAIN_MULTIPLIER
            md_total += contrib
            per_md_lines.append((p["name"], p["position"], contrib))
        rows.append({
            "matchday": md,
            "xi_total": round(md_total, 1),
            "captain": captain_name,
            "captain_contrib": round(
                next((c for n, _, c in per_md_lines if n == captain_name), 0.0), 1
            ),
            "n_players_active": len(per_md_lines),
        })
    return pd.DataFrame(rows)


def run_squad_backtest(year: int, combined: pd.DataFrame) -> dict:
    """For one historical year, run LP → XI → captain → score using the
    matched players from run_backtest() and merge in actual final fantasy
    points from the historical CSV."""
    # Load CSV for matches_played + final_fantasy_points
    csv_path = HISTORICAL_DIR / YEARS[year]["csv"]
    hist = pd.read_csv(csv_path)
    hist = hist.rename(columns={"player": "name"})

    # Build the player pool: only matched rows from `combined` (ep90 available)
    year_rows = combined[(combined["year"] == year) & combined["ep90"].notna()].copy()
    pool = year_rows[["name", "team", "position", "ep90"]].drop_duplicates(subset=["name", "team"])
    pool["price"] = pool.apply(lambda r: estimate_price(r["ep90"], r["position"]), axis=1)

    if pool.empty:
        return {"year": year, "error": "no matched players"}

    # Position count check — LP needs ≥2 GK, ≥5 DEF, ≥5 MID, ≥3 FWD
    counts = pool["position"].value_counts().to_dict()
    needed = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    short = {p: needed[p] - counts.get(p, 0) for p in needed if counts.get(p, 0) < needed[p]}
    if short:
        # Synthesize dummy filler players at minimum price so LP completes —
        # they'll have ep90 ≈ 0 and won't actually be chosen unless absolutely
        # necessary. Real production would draw from a fuller pool.
        for pos, n in short.items():
            lo = _POSITION_PRICE_BOUNDS[pos][0]
            for k in range(n):
                pool = pd.concat([pool, pd.DataFrame([{
                    "name": f"__dummy_{pos}_{k}",
                    "team": f"__dummy_team_{k}",   # cap-busting fake nation
                    "position": pos,
                    "ep90": 0.0,
                    "price": lo,
                }])], ignore_index=True)

    squad = optimize_squad(pool)

    # Bring in actuals from the historical CSV (left-join on (name, team))
    squad = squad.merge(
        hist[["name", "team", "matches_played", "minutes", "final_fantasy_points"]]
            .rename(columns={"minutes": "actual_mins",
                             "final_fantasy_points": "actual_pts"}),
        on=["name", "team"], how="left",
    )
    # Players who don't appear in the historical CSV scored 0 (not in our top 56)
    squad["actual_pts"] = squad["actual_pts"].fillna(0)
    squad["actual_mins"] = squad["actual_mins"].fillna(0)
    squad["matches_played"] = squad["matches_played"].fillna(0)
    # Strip dummies before reporting (they scored 0 anyway)
    squad = squad[~squad["name"].str.startswith("__dummy")].reset_index(drop=True)

    # Pick XI by actual points (this IS a hindsight choice for evaluation —
    # the model didn't know these, but we want to score the best XI the system
    # could have fielded each MD if pivoted optimally with bench)
    xi, formation = select_xi(squad, sort_col="actual_pts")
    bench = squad[~squad["name"].isin(xi["name"])].copy()

    # Captain = highest-actual scorer in XI
    captain_row = xi.loc[xi["actual_pts"].idxmax()]
    captain_name = captain_row["name"]

    # Total score = XI actuals + captain doubling (= +captain's actual)
    xi_total = xi["actual_pts"].sum()
    captain_bonus = captain_row["actual_pts"]
    total = xi_total + captain_bonus

    # Per-MD breakdown (approximation)
    md_breakdown = per_matchday_breakdown(xi, captain_name)

    return {
        "year": year,
        "n_pool": len(pool),
        "formation": formation,
        "squad": squad.sort_values("actual_pts", ascending=False),
        "xi": xi.sort_values("actual_pts", ascending=False),
        "bench": bench.sort_values("actual_pts", ascending=False),
        "captain_name": captain_name,
        "captain_team": captain_row["team"],
        "captain_actual": captain_row["actual_pts"],
        "xi_total": xi_total,
        "captain_bonus": captain_bonus,
        "total": total,
        "md_breakdown": md_breakdown,
        "squad_cost": squad["price"].sum(),
    }


if __name__ == "__main__":
    out = run_backtest()
    summary = out["summary"]
    combined = out["combined"]

    print("=== Backtest summary (stat-only, no priors) ===")
    print(f"{'Year':<8} {'matched':>8} {'predicted':>10} {'Pearson r':>11} {'MAE':>8}")
    for k in [2018, 2022, "pooled"]:
        s = summary[k]
        n_pred = s["n_predicted"]
        r = s["pearson_r"]
        mae = s["mae"]
        nm = s.get("n_matched", "—")
        print(f"{str(k):<8} {str(nm):>8} {n_pred:>10}  {r:>10.3f}  {mae:>7.2f}")

    pooled_r = summary["pooled"]["pearson_r"]
    print()
    if pooled_r >= VALIDATION_THRESHOLD_R:
        print(f"VALIDATED  ✓   pooled r = {pooled_r:.3f} ≥ {VALIDATION_THRESHOLD_R}")
    else:
        print(f"NOT VALIDATED  ✗   pooled r = {pooled_r:.3f} < {VALIDATION_THRESHOLD_R}")
        print("  Stat-only baseline; priors + odds blend are expected to lift r in production.")

    # Top over/under projections
    pred = combined[combined["predicted_pts"].notna()].copy()
    pred["residual"] = pred["actual_pts"] - pred["predicted_pts"]

    print(f"\n=== Top 20 UNDER-projections (we predicted too low) ===")
    print(pred.nlargest(20, "residual")[
        ["year", "name", "team", "position", "ep90",
         "predicted_pts", "actual_pts", "residual"]
    ].to_string(index=False))

    print(f"\n=== Top 20 OVER-projections (we predicted too high) ===")
    print(pred.nsmallest(20, "residual")[
        ["year", "name", "team", "position", "ep90",
         "predicted_pts", "actual_pts", "residual"]
    ].to_string(index=False))

    # Unmatched count diagnostic
    unmatched = combined[~combined["matched"]]
    print(f"\nUnmatched players (likely outside Big-5 that season): {len(unmatched)}")
    if not unmatched.empty:
        print("Examples:")
        for _, r in unmatched.head(10).iterrows():
            print(f"  {r['name']:<22}  {r['team']:<14}  {r['position']}")

    print(f"\nScatter saved to: {SCATTER_PATH}")
    print("\nNote: odds-blended backtest skipped — the-odds-api free tier does")
    print("not include historical player goalscorer markets for 2018/2022.")

    # -----------------------------------------------------------------
    # Squad-optimizer backtest: would the system have scored?
    # -----------------------------------------------------------------
    print("\n\n" + "=" * 78)
    print("=== SQUAD-OPTIMIZER BACKTEST (LP picks 15 by predicted ep90) ===")
    print("=" * 78)
    print("Caveats:")
    print(" - Prices are estimated procedurally from ep90 + position (no historical")
    print("   FIFA fantasy price data available); calibrated to current 2026 spreads.")
    print(" - Per-MD captain breakdown approximates per-MD points as total/matches_played")
    print("   (historical CSVs only ship tournament totals, not per-MD breakdowns).")
    print(" - XI selection uses actual points (a hindsight ceiling on what an optimal")
    print("   bench-management strategy could have achieved with this squad).")

    for year in (2018, 2022):
        r = run_squad_backtest(year, combined)
        if "error" in r:
            print(f"\n{year}: {r['error']}")
            continue
        print(f"\n\n=== WC {year} ===")
        print(f"pool size: {r['n_pool']} matched players  |  formation: {r['formation']}")
        print(f"squad cost: ${r['squad_cost']:.1f}M / $100M budget")
        print(f"\nSquad (15) by actual final fantasy points:")
        cols = ["name", "team", "position", "price", "ep90",
                "matches_played", "actual_pts"]
        print(r["squad"][cols].to_string(index=False))

        print(f"\nStarting XI (sum of actuals = {r['xi_total']:.0f}):")
        print(r["xi"][cols].to_string(index=False))

        print(f"\nBench (sum of actuals = {r['bench']['actual_pts'].sum():.0f}, "
              "would not contribute to score):")
        print(r["bench"][cols].to_string(index=False))

        print(f"\nCAPTAIN: {r['captain_name']} ({r['captain_team']}) "
              f"— actual {r['captain_actual']:.0f} pts × 2 = +{r['captain_bonus']:.0f}")
        print(f"\n>>> TOTAL FANTASY POINTS (XI + captain double): {r['total']:.0f}")

        print(f"\nApproximated per-MD breakdown (uniform total/matches_played):")
        print(r["md_breakdown"].to_string(index=False))
