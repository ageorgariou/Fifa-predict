"""
2026 FIFA World Cup — Official Group Draw

Source: FIFA / NBC Sports, confirmed after UEFA play-off winners were determined.
All 48 teams placed in 12 groups of 4. Top 2 + 8 best 3rd-placed teams → R32.

Tournament: 11 June – 19 July 2026 (USA / Canada / Mexico)
Total matches: 104. Fantasy matchdays: 8.

Use this dict in model/tournament_sim.py. Order in each group matters:
position [0] is the seeded host or Pot 1 team (predetermined slot).
"""

GROUPS_2026 = {
    "A": ["Mexico", "South Korea", "South Africa", "Czechia"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia-Herzegovina"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["United States", "Paraguay", "Australia", "Turkiye"],
    "E": ["Germany", "Ecuador", "Ivory Coast", "Curacao"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Iran", "Egypt", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "DR Congo"],
    "L": ["England", "Croatia", "Panama", "Ghana"],
}

# Host nations — apply host_nation_bonus in priors.py
HOST_NATIONS = {"Mexico", "Canada", "United States"}

# Predetermined positional slots (from FIFA draw rules):
# Mexico = A1, Canada = B1, USA = D1
HOST_POSITIONS = {"Mexico": ("A", 0), "Canada": ("B", 0), "United States": ("D", 0)}


def all_teams() -> list[str]:
    """Returns flat list of all 48 teams."""
    teams = []
    for group_teams in GROUPS_2026.values():
        teams.extend(group_teams)
    assert len(teams) == 48, f"Expected 48 teams, got {len(teams)}"
    return teams


def team_to_group(team: str) -> str:
    """Returns the group letter for a given team."""
    for letter, teams in GROUPS_2026.items():
        if team in teams:
            return letter
    raise ValueError(f"Team not found in 2026 draw: {team}")


def group_opponents(team: str) -> list[str]:
    """Returns the 3 group-stage opponents for a given team."""
    group = team_to_group(team)
    return [t for t in GROUPS_2026[group] if t != team]


# Standard ISO country names for joining with FBref / Elo / odds sources.
# Mappings needed where name conventions differ (rapidfuzz fallback in data/name_matcher.py).
NAME_ALIASES = {
    "United States": ["USA", "USMNT", "US"],
    "South Korea": ["Korea Republic", "Republic of Korea"],
    "Czechia": ["Czech Republic"],
    "Bosnia-Herzegovina": ["Bosnia and Herzegovina", "Bosnia & Herzegovina", "Bosnia"],
    "Ivory Coast": ["Côte d'Ivoire", "Cote d'Ivoire"],
    "Curacao": ["Curaçao"],
    "Turkiye": ["Turkey", "Türkiye"],
    "DR Congo": ["Democratic Republic of the Congo", "DR Congo", "DRC"],
    "Cape Verde": ["Cabo Verde"],
}


# FBref / FIFA Fantasy / odds-source nation codes (3-letter) → canonical name.
# FBref's `nation` column uses FIFA-style codes (ENG not GBR, GER not DEU,
# POR not PRT, ...). Verified against actual rows pulled in step 7.
COUNTRY_CODE_TO_TEAM: dict[str, str] = {
    # Group A
    "MEX": "Mexico", "KOR": "South Korea", "RSA": "South Africa", "CZE": "Czechia",
    # Group B
    "CAN": "Canada", "SUI": "Switzerland", "QAT": "Qatar", "BIH": "Bosnia-Herzegovina",
    # Group C
    "BRA": "Brazil", "MAR": "Morocco", "SCO": "Scotland", "HAI": "Haiti",
    # Group D
    "USA": "United States", "PAR": "Paraguay", "AUS": "Australia", "TUR": "Turkiye",
    # Group E
    "GER": "Germany", "ECU": "Ecuador", "CIV": "Ivory Coast", "CUW": "Curacao",
    # Group F
    "NED": "Netherlands", "JPN": "Japan", "TUN": "Tunisia", "SWE": "Sweden",
    # Group G
    "BEL": "Belgium", "IRN": "Iran", "EGY": "Egypt", "NZL": "New Zealand",
    # Group H
    "ESP": "Spain", "URU": "Uruguay", "KSA": "Saudi Arabia", "CPV": "Cape Verde",
    # Group I
    "FRA": "France", "SEN": "Senegal", "NOR": "Norway", "IRQ": "Iraq",
    # Group J
    "ARG": "Argentina", "AUT": "Austria", "ALG": "Algeria", "JOR": "Jordan",
    # Group K
    "POR": "Portugal", "COL": "Colombia", "UZB": "Uzbekistan", "COD": "DR Congo",
    # Group L
    "ENG": "England", "CRO": "Croatia", "PAN": "Panama", "GHA": "Ghana",
}


def country_to_team(code: str) -> str | None:
    """3-letter nation code → canonical team name, or None if unknown."""
    if not isinstance(code, str):
        return None
    code = code.strip().upper()
    if not code:
        return None
    return COUNTRY_CODE_TO_TEAM.get(code)


# ---- Team-name alias normalization (FIFA squads.json may use "USA",
# odds API uses "Bosnia & Herzegovina", FIFA may use "Türkiye" with the umlaut,
# etc.) ------------------------------------------------------------------------
import unicodedata as _unicodedata

_SMART_QUOTES_TO_ASCII = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def _fold(s: str) -> str:
    """Accent-strip + smart-quote-normalize + lowercase."""
    if not isinstance(s, str):
        return ""
    s = s.translate(_SMART_QUOTES_TO_ASCII)
    nfkd = _unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not _unicodedata.combining(c)).lower().strip()


_TEAM_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canon, _aliases in NAME_ALIASES.items():
    _TEAM_ALIAS_TO_CANONICAL[_fold(_canon)] = _canon
    for _alias in _aliases:
        _TEAM_ALIAS_TO_CANONICAL[_fold(_alias)] = _canon
for _group_teams in GROUPS_2026.values():
    for _team in _group_teams:
        _TEAM_ALIAS_TO_CANONICAL.setdefault(_fold(_team), _team)


def canonical_team(name: str) -> str | None:
    """Normalize any team-name variant to its canonical form, or None.

    Handles "USA"→"United States", "Türkiye"→"Turkiye", "Bosnia &
    Herzegovina"→"Bosnia-Herzegovina", smart-quote variants, etc."""
    return _TEAM_ALIAS_TO_CANONICAL.get(_fold(name))


if __name__ == "__main__":
    # Sanity check
    teams = all_teams()
    print(f"Total teams: {len(teams)}")
    print(f"Total groups: {len(GROUPS_2026)}")
    for letter, group in GROUPS_2026.items():
        print(f"Group {letter}: {', '.join(group)}")
    # Confirm code map is 1:1 with the 48 teams
    mapped = set(COUNTRY_CODE_TO_TEAM.values())
    missing = set(teams) - mapped
    extra = mapped - set(teams)
    if missing or extra:
        print(f"\nWARN  missing from code map: {missing}")
        print(f"WARN  extra in code map:     {extra}")
    else:
        print("\nCOUNTRY_CODE_TO_TEAM covers all 48 teams ✓")
