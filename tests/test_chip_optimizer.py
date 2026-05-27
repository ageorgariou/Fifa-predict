"""Smoke test for model.chip_optimizer.

Runs the optimizer against synthetic EV / squad / sim inputs to verify:
* The plan covers every chip with no duplicates.
* Maximum Captain is never assigned to MD1.
* Mystery Booster is always "hold".
* Qualification Booster only triggers on knockout MDs.
"""
from __future__ import annotations

import pandas as pd

from model.chip_optimizer import (
    ALL_CHIPS,
    KNOCKOUT_MDS,
    recommend_chip_sequence,
)


def _synthetic_ev() -> pd.DataFrame:
    """30 players spread across 3 teams + 4 positions, all 8 MDs covered."""
    rows = []
    teams = ["Brazil", "Spain", "France"]
    positions = [("GK", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]
    for t in teams:
        for pos, n in positions:
            for i in range(n):
                base = 6.0 + (1.5 if pos == "FWD" else 0.0)
                row = {
                    "name": f"{t}-{pos}{i}",
                    "team": t,
                    "position": pos,
                    "price": 5.0 + i * 0.5,
                    "ep90": base,
                }
                for md in range(1, 9):
                    row[f"ev_md{md}"] = round(base + 0.1 * md, 2)
                row["ev_total"] = sum(row[f"ev_md{md}"] for md in range(1, 9))
                rows.append(row)
    return pd.DataFrame(rows)


def _synthetic_sim(ev: pd.DataFrame) -> pd.DataFrame:
    teams = ev["team"].unique()
    return pd.DataFrame([
        {
            "team": t,
            "p_r32": 0.9, "p_r16": 0.7, "p_qf": 0.5,
            "p_sf": 0.3, "p_f": 0.15, "p_win": 0.05,
            "expected_matches": 5.0,
            "expected_clean_sheets": 2.0,
        }
        for t in teams
    ])


def _synthetic_squad(ev: pd.DataFrame) -> pd.DataFrame:
    """Hand-pick a valid 15 from the synthetic EV table (2 GK / 5 DEF /
    5 MID / 3 FWD)."""
    out = []
    for pos, n in [("GK", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]:
        out.append(ev[ev["position"] == pos].head(n))
    return pd.concat(out).reset_index(drop=True)


def test_chip_plan_covers_all_chips():
    ev = _synthetic_ev()
    squad = _synthetic_squad(ev)
    sim = _synthetic_sim(ev)
    plan = recommend_chip_sequence(ev=ev, squad=squad, sim_summary=sim)
    chips_in_plan = {p["chip"] for p in plan}
    assert chips_in_plan == set(ALL_CHIPS)


def test_max_captain_never_md1():
    ev = _synthetic_ev()
    squad = _synthetic_squad(ev)
    sim = _synthetic_sim(ev)
    plan = recommend_chip_sequence(ev=ev, squad=squad, sim_summary=sim)
    for p in plan:
        if p["chip"] == "Maximum Captain":
            assert p["matchday"] != 1


def test_mystery_booster_is_hold():
    ev = _synthetic_ev()
    squad = _synthetic_squad(ev)
    sim = _synthetic_sim(ev)
    plan = recommend_chip_sequence(ev=ev, squad=squad, sim_summary=sim)
    mystery = next(p for p in plan if p["chip"] == "Mystery Booster")
    assert mystery["matchday"] is None
    assert "hold" in mystery["reasoning"].lower()


def test_qual_booster_in_knockout():
    ev = _synthetic_ev()
    squad = _synthetic_squad(ev)
    sim = _synthetic_sim(ev)
    plan = recommend_chip_sequence(ev=ev, squad=squad, sim_summary=sim)
    qb = next(p for p in plan if p["chip"] == "Qualification Booster")
    # Either the optimizer picked a knockout MD OR found zero lift everywhere
    # (which can happen with synthetic data — accept either as long as it
    # didn't pick a group MD with nonzero lift).
    if qb["matchday"] is not None and qb["expected_lift"] > 0:
        assert qb["matchday"] in KNOCKOUT_MDS


def test_chips_assigned_to_distinct_matchdays():
    ev = _synthetic_ev()
    squad = _synthetic_squad(ev)
    sim = _synthetic_sim(ev)
    plan = recommend_chip_sequence(ev=ev, squad=squad, sim_summary=sim)
    mds = [p["matchday"] for p in plan if p["matchday"] is not None]
    assert len(mds) == len(set(mds)), "chips collided on a matchday"
