"""Smoke tests for model.schedule deadline helpers."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from model.schedule import (
    MD_DEADLINES_UTC,
    current_matchday,
    md_info,
    next_deadline,
    within_window,
)


def test_all_deadlines_are_utc_aware():
    for md, dt in MD_DEADLINES_UTC.items():
        assert dt.tzinfo is not None, f"MD{md} deadline must be tz-aware"
        assert 1 <= md <= 8


def test_deadlines_are_monotonic():
    deadlines = [MD_DEADLINES_UTC[md] for md in sorted(MD_DEADLINES_UTC)]
    for a, b in zip(deadlines, deadlines[1:]):
        assert a < b


def test_next_deadline_before_tournament():
    before = datetime(2026, 1, 1, tzinfo=timezone.utc)
    info = next_deadline(before)
    assert info is not None
    assert info.matchday == 1


def test_next_deadline_mid_tournament():
    mid = datetime(2026, 6, 20, tzinfo=timezone.utc)
    info = next_deadline(mid)
    assert info is not None
    assert info.matchday == 3


def test_next_deadline_after_tournament():
    after = datetime(2027, 1, 1, tzinfo=timezone.utc)
    assert next_deadline(after) is None


def test_current_matchday_before():
    assert current_matchday(datetime(2026, 1, 1, tzinfo=timezone.utc)) == 1


def test_current_matchday_after():
    assert current_matchday(datetime(2027, 1, 1, tzinfo=timezone.utc)) == 8


def test_within_window_hits_deadline():
    md3 = MD_DEADLINES_UTC[3]
    # 90 minutes before MD3 — email window
    now = md3 - timedelta(minutes=90)
    info = within_window(target_minutes_before=90, tolerance_min=15, now=now)
    assert info is not None
    assert info.matchday == 3


def test_within_window_misses_when_outside_tolerance():
    md3 = MD_DEADLINES_UTC[3]
    # 6 hours before MD3 — well outside the 90min ± 15 window
    now = md3 - timedelta(hours=6)
    assert within_window(90, 15, now) is None


def test_md_info_invalid():
    with pytest.raises(ValueError):
        md_info(99)
