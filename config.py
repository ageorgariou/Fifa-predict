"""Central config: paths, API keys, cache TTLs, tournament constants.

Reads from env vars (loaded via python-dotenv) with a Streamlit secrets fallback
so the same code works locally and on Streamlit Cloud.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
HISTORICAL_DIR = DATA_DIR / "historical"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _secret(name: str) -> str | None:
    val = os.environ.get(name)
    if val:
        return val
    try:
        import streamlit as st  # type: ignore

        return st.secrets.get(name)  # type: ignore[no-any-return]
    except Exception:
        return None


ANTHROPIC_API_KEY = _secret("ANTHROPIC_API_KEY")
ODDS_API_KEY = _secret("ODDS_API_KEY")

CLAUDE_MODEL = "claude-opus-4-7"

# Cache TTLs (seconds)
TTL_FBREF = 7 * 24 * 3600
TTL_TOURNAMENT_ODDS = 24 * 3600
TTL_GOALSCORER_ODDS = 6 * 3600
TTL_ELO = 24 * 3600
TTL_FIFA_PLAYERS = 24 * 3600

# Tournament constants
SQUAD_SIZE = 15
SQUAD_SPLIT = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
BUDGET_GROUP = 100.0
BUDGET_R32 = 105.0
NATION_CAP = {
    "GROUP": 3,
    "R32": 3,
    "R16": 4,
    "QF": 5,
    "SF": 6,
    "F": 8,
}
N_SIMS = 10_000
