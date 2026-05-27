"""Smoke tests for the shot-volume captain tiebreaker."""
from __future__ import annotations

import pandas as pd

from model.captain_optimizer import (
    TIEBREAK_EV_WINDOW,
    _apply_tiebreak,
    optimize_captain,
)


def _synthetic_xi() -> pd.DataFrame:
    """11-row XI with two close-EV forwards. Player A is the EV leader."""
    return pd.DataFrame([
        {"name": "Player A", "team": "X", "position": "FWD", "ev_md1": 8.0},
        {"name": "Player B", "team": "Y", "position": "FWD", "ev_md1": 7.7},
        {"name": "Player C", "team": "Z", "position": "MID", "ev_md1": 7.5},
        {"name": "Player D", "team": "W", "position": "DEF", "ev_md1": 6.0},
    ] + [
        {"name": f"F{i}", "team": "F", "position": "DEF", "ev_md1": 3.0}
        for i in range(7)
    ])


def test_tiebreak_swaps_when_close_ev_and_higher_sot():
    """A and B are within 0.5 EV; B has higher per90 SoT → B wins."""
    xi = _synthetic_xi()
    rec = optimize_captain(
        xi, "ev_md1", league_size=15,
        shot_volume={"Player A": 1.0, "Player B": 2.5},
    )
    assert rec["primary_captain"] == "Player B"
    assert "tiebreaker" in rec["captain_reasoning"].lower()


def test_tiebreak_keeps_leader_when_he_has_higher_sot():
    xi = _synthetic_xi()
    rec = optimize_captain(
        xi, "ev_md1", league_size=15,
        shot_volume={"Player A": 3.0, "Player B": 1.0},
    )
    assert rec["primary_captain"] == "Player A"
    # No tiebreak applied → reasoning shouldn't mention it.
    assert "tiebreaker" not in rec["captain_reasoning"].lower()


def test_tiebreak_disabled_when_shot_volume_empty():
    xi = _synthetic_xi()
    rec = optimize_captain(xi, "ev_md1", league_size=15, shot_volume={})
    assert rec["primary_captain"] == "Player A"


def test_tiebreak_ignored_outside_ev_window():
    """Player D has higher SoT but EV is well below Player A (8.0 vs 6.0,
    gap = 2.0, much larger than TIEBREAK_EV_WINDOW=0.5) → no swap."""
    xi = _synthetic_xi()
    rec = optimize_captain(
        xi, "ev_md1", league_size=15,
        shot_volume={"Player A": 1.0, "Player D": 5.0},
    )
    assert rec["primary_captain"] == "Player A"


def test_tiebreak_missing_data_falls_back_to_raw_ev():
    """If the leader has no SoT data, the tiebreaker can't act safely."""
    xi = _synthetic_xi()
    rec = optimize_captain(
        xi, "ev_md1", league_size=15,
        shot_volume={"Player B": 3.0},  # only B has data; A missing
    )
    # Leader missing → no swap.
    assert rec["primary_captain"] == "Player A"


def test_window_constant():
    assert TIEBREAK_EV_WINDOW == 0.5


def test_apply_tiebreak_unit():
    """Direct unit test on the helper."""
    table = pd.DataFrame([
        {"name": "A", "ep_round": 8.0},
        {"name": "B", "ep_round": 7.8},
        {"name": "C", "ep_round": 6.0},
    ])
    chosen, note = _apply_tiebreak(table, "A", {"A": 1.0, "B": 2.0, "C": 5.0})
    assert chosen == "B"
    assert note is not None
    # C beats B on SoT but is outside the EV window.
