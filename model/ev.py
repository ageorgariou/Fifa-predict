"""Per-player tournament EV aggregation.

For each player × matchday r:

    EV_r = ep90 × P(team plays in MD r) × minutes_share

Then EV_total = sum_r EV_r.

P(team plays MD r) comes from the tournament simulator:
    MD1, MD2, MD3 = 1.0   (every team plays all 3 group games)
    MD4           = p_r32 (qualified to round of 32)
    MD5           = p_r16
    MD6           = p_qf
    MD7           = p_sf
    MD8           = p_sf  (both finalists and 3rd-place teams reached SF)

minutes_share is derived from FBref club data: `minutes / (matches_played × 90)`,
clamped to [0, 1]. If the player has missing/zero matches, we use a default
of 0.85 (≈ a regular international starter).

Public API
----------
    compute_ev(player_pool, sim_summary) -> pd.DataFrame
        Columns: name, team, position, price, ep90, minutes_share,
                 ev_md1..ev_md8, ev_total.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

N_MATCHDAYS = 8
DEFAULT_MINUTES_SHARE = 0.85


def _p_play_for_md(sim_row, md: int) -> float:
    if md in (1, 2, 3):
        return 1.0
    if md == 4:
        return float(sim_row["p_r32"])
    if md == 5:
        return float(sim_row["p_r16"])
    if md == 6:
        return float(sim_row["p_qf"])
    if md == 7 or md == 8:   # SF round + (final / 3rd-place playoff)
        return float(sim_row["p_sf"])
    raise ValueError(f"invalid matchday {md}")


def _minutes_share(row) -> float:
    """Club minutes per 90 expected at international level. We assume regular
    club starters keep their minutes role for their national team — true for
    star players, slightly noisy for fringe ones."""
    mins = row.get("minutes")
    matches = row.get("matches_played", None)
    if matches is None:
        matches = row.get("mp")
    try:
        mins = float(mins)
        matches = float(matches)
    except (TypeError, ValueError):
        return DEFAULT_MINUTES_SHARE
    if np.isnan(mins) or np.isnan(matches) or matches <= 0:
        return DEFAULT_MINUTES_SHARE
    share = mins / (matches * 90.0)
    return float(min(1.0, max(0.0, share)))


def compute_ev(
    player_pool: pd.DataFrame,
    sim_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate per-matchday EV.

    `player_pool` needs columns: name, team, position, price, ep90,
        minutes, matches_played (or mp). Players whose team isn't in
        `sim_summary` are dropped.
    `sim_summary` is the DataFrame returned by tournament_sim.simulate().
    """
    sim_by_team = sim_summary.set_index("team")

    rows = []
    for _, p in player_pool.iterrows():
        team = p.get("team")
        if team is None or team not in sim_by_team.index:
            continue
        sim_row = sim_by_team.loc[team]
        ep90 = float(p.get("ep90") or 0.0)
        if ep90 <= 0:
            continue
        ms = _minutes_share(p)

        ev_per_md = {}
        ev_total = 0.0
        for md in range(1, N_MATCHDAYS + 1):
            ev_r = ep90 * _p_play_for_md(sim_row, md) * ms
            ev_per_md[f"ev_md{md}"] = round(ev_r, 3)
            ev_total += ev_r

        rows.append({
            "name": p["name"],
            "team": team,
            "position": p["position"],
            "price": p.get("price"),
            "ep90": round(ep90, 3),
            "minutes_share": round(ms, 3),
            **ev_per_md,
            "ev_total": round(ev_total, 3),
        })
    return pd.DataFrame(rows).sort_values("ev_total", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from data.fifa_players import fetch_player_list
    from model.projections import compute_ep90
    from model.tournament_sim import simulate

    proj = compute_ep90()
    sim = simulate(n_sims=5000)
    fifa = fetch_player_list()
    # Merge prices into the projection table (proj already has them, but
    # confirm we're not missing minutes/matches_played for ev compute)
    proj_with_mp = proj.copy()
    if "matches_played" not in proj_with_mp.columns:
        proj_with_mp["matches_played"] = pd.NA
    if "minutes" not in proj_with_mp.columns:
        proj_with_mp["minutes"] = pd.NA

    ev = compute_ev(proj_with_mp, sim)
    print(f"EV table: {len(ev)} players")
    print("\n=== Top 20 by ev_total ===")
    cols = ["name", "team", "position", "price", "ep90",
            "minutes_share", "ev_md1", "ev_md4", "ev_md7", "ev_md8",
            "ev_total"]
    print(ev.head(20)[cols].to_string(index=False))

    # Bottom check: ev_total / ep90 should ≈ 8 × minutes_share for top-team
    # players (who play every MD) and proportionally less for weaker teams.
    print("\n=== EV multiplier check (ev_total / ep90) ===")
    ev["ev_mult"] = ev["ev_total"] / ev["ep90"]
    sample = ev.groupby("team")["ev_mult"].mean().sort_values(ascending=False)
    print(sample.head(10).to_string())
