"""FIFA World Cup 2026 Fantasy scoring constants.

Every projection / EV / optimizer call must reference these constants — never
inline a points value at the call site. Sources:

- Spec tables provided by the user.
- play.fifa.com/fantasy/help/rules (JS-rendered; could not be auto-verified;
  spec values trusted unless flagged below).
- Differential bonus values (+2 / <5% / >4) cross-checked against public
  game guides as of 2026-05-26.
- PEN_WON / PEN_CONCEDED were not auto-verifiable; user accepted spec
  defaults (+2 / -1) on 2026-05-27.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# All players
# ---------------------------------------------------------------------------
APPEARANCE_UP_TO_60 = 1
APPEARANCE_60_PLUS = 2

YELLOW_CARD = -1
RED_CARD = -2
OWN_GOAL = -2
MISSED_PENALTY = -2

GOAL_OUTSIDE_BOX_BONUS = 1
DIRECT_FREE_KICK_GOAL_BONUS = 1

# Spec defaults adopted by user decision (2026-05-27) — could not auto-verify
# on play.fifa.com (JS-rendered) or via reachable secondary sources. Penalty
# events are rare enough that small mis-specification here won't materially
# shift projections; revisit if a confirmed source surfaces.
PEN_WON = 2
PEN_CONCEDED = -1

# ---------------------------------------------------------------------------
# Goalkeepers
# ---------------------------------------------------------------------------
GK_GOAL = 6
GK_ASSIST = 3
GK_CLEAN_SHEET = 5            # awarded only if player played 60+ minutes
GK_CONCEDED_DIVISOR = 2       # -1 per N goals conceded after the first
GK_CONCEDED_PENALTY = -1
GK_SAVES_PER_BONUS = 3        # +1 per N saves
GK_SAVE_BONUS = 1
GK_PEN_SAVE = 5

# ---------------------------------------------------------------------------
# Defenders
# ---------------------------------------------------------------------------
DEF_GOAL = 6
DEF_ASSIST = 3
DEF_CLEAN_SHEET = 5
DEF_CONCEDED_DIVISOR = 2
DEF_CONCEDED_PENALTY = -1

# ---------------------------------------------------------------------------
# Midfielders
# ---------------------------------------------------------------------------
MID_GOAL = 5
MID_ASSIST = 3
MID_CLEAN_SHEET = 1
MID_TACKLES_PER_BONUS = 3      # +1 per N tackles
MID_TACKLE_BONUS = 1
MID_CHANCES_PER_BONUS = 2      # +1 per N chances created (key passes)
MID_CHANCE_BONUS = 1

# ---------------------------------------------------------------------------
# Forwards
# ---------------------------------------------------------------------------
FWD_GOAL = 4
FWD_ASSIST = 3
FWD_SOT_PER_BONUS = 2          # +1 per N shots on target
FWD_SOT_BONUS = 1

# ---------------------------------------------------------------------------
# Differential bonus (CONFIRMED via published 2026 game guides)
# +2 to a player who scored more than 4 points AND is owned by fewer than 5%
# of all teams for that matchday.
# ---------------------------------------------------------------------------
DIFFERENTIAL_BONUS = 2
DIFFERENTIAL_OWNERSHIP_MAX = 0.05   # exclusive: owned < 5%
DIFFERENTIAL_MIN_POINTS = 4         # exclusive: scored > 4 in the matchday

# ---------------------------------------------------------------------------
# Captaincy
# ---------------------------------------------------------------------------
CAPTAIN_MULTIPLIER = 2

# ---------------------------------------------------------------------------
# Position-keyed helper maps. Use these in projections / odds_blend so position
# branching stays out of the call site.
# ---------------------------------------------------------------------------
GOAL_POINTS = {
    "GK": GK_GOAL,
    "DEF": DEF_GOAL,
    "MID": MID_GOAL,
    "FWD": FWD_GOAL,
}
ASSIST_POINTS = {
    "GK": GK_ASSIST,
    "DEF": DEF_ASSIST,
    "MID": MID_ASSIST,
    "FWD": FWD_ASSIST,
}
CLEAN_SHEET_POINTS = {
    "GK": GK_CLEAN_SHEET,
    "DEF": DEF_CLEAN_SHEET,
    "MID": MID_CLEAN_SHEET,
    "FWD": 0,
}


