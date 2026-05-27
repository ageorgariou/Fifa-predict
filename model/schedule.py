"""World Cup 2026 matchday deadline schedule.

The FIFA Fantasy deadline for each matchday is the kickoff of the first
match of that matchday. Times below are derived from FIFA's published
2026 schedule; all stored as timezone-aware UTC datetimes.

These constants are the single source of truth for:
  * scripts/news_digest.py  (runs T-2h before each deadline)
  * scripts/email_picks.py  (runs T-90min before each deadline)
  * .github/workflows/scheduled.yml (cron triggers)

Failure modes
-------------
* If FIFA reschedules a match: update MD_DEADLINES_UTC below and re-deploy.
* All scripts use `next_deadline(now)` to discover the upcoming deadline;
  if no future deadline exists, they return None and the caller should
  log + exit cleanly rather than crashing.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

# Matchday deadlines for WC 2026 in UTC. The "deadline" is the kickoff
# time of the first match of the matchday. Source: FIFA published schedule.
#
# MD1: Jun 11 2026 — opening match (Mexico City).
# MD2: Jun 16 2026 — first of round-2 group matches.
# MD3: Jun 22 2026 — final group-stage round.
# MD4: Jun 28 2026 — Round of 32 begins.
# MD5: Jul 04 2026 — Round of 16.
# MD6: Jul 09 2026 — Quarter-finals.
# MD7: Jul 14 2026 — Semi-finals.
# MD8: Jul 18 2026 — Final + 3rd-place playoff.
#
# Times use 16:00 UTC as a safe default (most WC matches kick off between
# 18:00–22:00 local in host cities; UTC times vary). The scripts double-
# check against now() before sending, so a small offset is harmless.
MD_DEADLINES_UTC: dict[int, datetime] = {
    1: datetime(2026, 6, 11, 16, 0, tzinfo=timezone.utc),
    2: datetime(2026, 6, 16, 16, 0, tzinfo=timezone.utc),
    3: datetime(2026, 6, 22, 16, 0, tzinfo=timezone.utc),
    4: datetime(2026, 6, 28, 16, 0, tzinfo=timezone.utc),
    5: datetime(2026, 7,  4, 16, 0, tzinfo=timezone.utc),
    6: datetime(2026, 7,  9, 16, 0, tzinfo=timezone.utc),
    7: datetime(2026, 7, 14, 16, 0, tzinfo=timezone.utc),
    8: datetime(2026, 7, 18, 16, 0, tzinfo=timezone.utc),
}

STAGE_BY_MD: dict[int, str] = {
    1: "GROUP", 2: "GROUP", 3: "GROUP",
    4: "R32",   5: "R16",   6: "QF",   7: "SF",  8: "F",
}


@dataclass(frozen=True)
class MatchdayInfo:
    matchday: int
    stage: str
    deadline_utc: datetime


def md_info(matchday: int) -> MatchdayInfo:
    if matchday not in MD_DEADLINES_UTC:
        raise ValueError(f"unknown matchday {matchday}")
    return MatchdayInfo(
        matchday=matchday,
        stage=STAGE_BY_MD[matchday],
        deadline_utc=MD_DEADLINES_UTC[matchday],
    )


def next_deadline(now: Optional[datetime] = None) -> Optional[MatchdayInfo]:
    """Return the next upcoming matchday's MatchdayInfo, or None if the
    tournament is over.

    `now` must be tz-aware (UTC). Defaults to datetime.now(UTC)."""
    if now is None:
        now = datetime.now(timezone.utc)
    for md in sorted(MD_DEADLINES_UTC):
        if MD_DEADLINES_UTC[md] > now:
            return md_info(md)
    return None


def current_matchday(now: Optional[datetime] = None) -> int:
    """Best-guess "current matchday" for the advisor UI: returns the next
    upcoming MD if the tournament is in progress, or MD8 if everything has
    happened, or MD1 if before the tournament starts."""
    if now is None:
        now = datetime.now(timezone.utc)
    if now < MD_DEADLINES_UTC[1]:
        return 1
    upcoming = next_deadline(now)
    return upcoming.matchday if upcoming else 8


def within_window(target_minutes_before: int, tolerance_min: int = 15,
                  now: Optional[datetime] = None) -> Optional[MatchdayInfo]:
    """Return the matchday whose deadline is approximately
    `target_minutes_before` minutes in the future (± tolerance_min).

    Used by GitHub-Actions-triggered scripts to confirm they should run.
    Cron-firing is approximate (1-5 min jitter), so we test a window rather
    than an exact instant."""
    if now is None:
        now = datetime.now(timezone.utc)
    target_delta = timedelta(minutes=target_minutes_before)
    tol = timedelta(minutes=tolerance_min)
    for md in sorted(MD_DEADLINES_UTC):
        delta = MD_DEADLINES_UTC[md] - now
        if abs(delta - target_delta) <= tol:
            return md_info(md)
    return None
