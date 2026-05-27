"""Headless full-pipeline refresh.

Runs every 6h via GitHub Actions. Loads all data sources, runs the full
projection → simulation → EV → squad LP → XI → captain → claude_advisor
chain, and writes the result to `data/latest_recommendation.json` for the
Streamlit app to pick up on next page load.

Per-source resilience
---------------------
Each data source is wrapped in try/except and logged. The pipeline NEVER
crashes the whole run for a single missing source — the projection layer
already degrades gracefully (e.g. odds_blend returns the raw projection
when the goalscorer API is empty; priors fall back to position averages
when FBref is incomplete).

Failure modes
-------------
* FBref cache missing  → log, abort: the projection pipeline cannot run
  without club stats. Recovery: run `python -m data.fbref_stats refresh`
  locally and commit the new CSVs.
* Tournament-winner odds empty (typical 5+ days before MD1) → log, continue
  using stat-only projection (priors degrade to non-odds mode).
* Goalscorer odds empty → log, continue.
* Claude advisor key missing or call fails → log, write recommendation
  with `advice=None` so the UI/email can still render the model-only
  picks.

Output
------
data/latest_recommendation.json — see `_write_snapshot` for shape.
data/refresh_log.jsonl — appended one line per refresh with per-source
    status. Used by ops monitoring and as input to the email script's
    "data freshness" footer.

CLI
---
    python -m scripts.refresh_pipeline           # refresh current MD
    python -m scripts.refresh_pipeline --md 4    # explicit matchday
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config import DATA_DIR, NATION_CAP
from model.schedule import STAGE_BY_MD, current_matchday

log = logging.getLogger(__name__)

LATEST_JSON = DATA_DIR / "latest_recommendation.json"
REFRESH_LOG = DATA_DIR / "refresh_log.jsonl"
FBREF_CSV = DATA_DIR / "fbref_cache" / "big5_2025_2026.csv"


# ---------------------------------------------------------------------------
# Per-source guarded loaders
# ---------------------------------------------------------------------------
def _guarded(name: str, fn, statuses: dict, *args, **kwargs):
    """Run `fn(*args, **kwargs)`. Record success/failure in `statuses`.
    On failure, return None so the caller can keep going."""
    t0 = time.time()
    try:
        out = fn(*args, **kwargs)
        statuses[name] = {
            "ok": True,
            "elapsed_s": round(time.time() - t0, 2),
            "shape": (
                list(out.shape) if isinstance(out, pd.DataFrame)
                else (len(out) if hasattr(out, "__len__") else None)
            ),
        }
        return out
    except Exception as exc:  # noqa: BLE001
        log.exception("source %s failed", name)
        statuses[name] = {
            "ok": False,
            "elapsed_s": round(time.time() - t0, 2),
            "error": f"{type(exc).__name__}: {exc}",
        }
        return None


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------
def _df_records(df: pd.DataFrame | None) -> list[dict]:
    if df is None or df.empty:
        return []
    safe = df.where(pd.notna(df), None)
    return safe.to_dict(orient="records")


def _advice_dict(advice) -> dict | None:
    if advice is None:
        return None
    if hasattr(advice, "model_dump"):
        return advice.model_dump()
    return dict(advice)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------
def run_refresh(matchday: int, call_advisor: bool = True,
                mode: str = "floor", league_size: int = 15) -> dict[str, Any]:
    """Run the full advisor pipeline for `matchday` and return the snapshot
    dict that will be written to disk. Never raises for missing data
    sources (only for genuine bugs)."""
    statuses: dict[str, dict] = {}

    # Guard: FBref cache must exist. Without it, the whole projection
    # falls apart, so abort rather than producing garbage picks.
    if not FBREF_CSV.exists():
        raise FileNotFoundError(
            f"FBref cache missing at {FBREF_CSV}. "
            "Run `python -m data.fbref_stats refresh` locally and commit."
        )

    # Import lazily so a missing optional dep (e.g. anthropic) for a
    # specific source doesn't kill the whole module at import time.
    from data.fbref_stats import fetch_outfield_stats, fetch_gk_stats
    from data.elo import fetch_elo
    from data.odds import fetch_winner_probs
    from data.fifa_players import fetch_player_list
    from data.goalscorer_odds import fetch_wc_goalscorer_probs
    from model.projections import compute_ep90
    from model.tournament_sim import simulate
    from model.ev import compute_ev
    from model.optimizer import optimize_squad, pick_xi
    from model.captain_optimizer import optimize_captain

    log.info("loading data sources …")
    _guarded("fbref_outfield", fetch_outfield_stats, statuses)
    _guarded("fbref_gk", fetch_gk_stats, statuses)
    _guarded("elo", fetch_elo, statuses)
    _guarded("odds_winner", fetch_winner_probs, statuses)
    fifa_df = _guarded("fifa_players", fetch_player_list, statuses)
    _guarded("goalscorer_odds", fetch_wc_goalscorer_probs, statuses)

    # ep90 / sim / ev are the heaviest part of the pipeline. If any of
    # these fail we cannot produce picks — propagate the exception so
    # GitHub Actions surfaces the failure.
    log.info("computing projections …")
    proj = compute_ep90()
    statuses["projections"] = {"ok": True, "shape": list(proj.shape)}

    log.info("running tournament simulator …")
    sim = simulate()
    statuses["tournament_sim"] = {"ok": True, "shape": list(sim.shape)}

    log.info("computing EV …")
    ev = compute_ev(proj, sim)
    statuses["ev"] = {"ok": True, "shape": list(ev.shape)}

    stage = STAGE_BY_MD[matchday]
    log.info("solving squad LP for stage %s …", stage)
    squad = optimize_squad(ev, nation_cap=NATION_CAP[stage])

    # Attach ownership
    if fifa_df is not None and not fifa_df.empty:
        own_map = fifa_df.set_index(["name", "team"])["ownership_pct"].to_dict()
        squad = squad.copy()
        squad["ownership_pct"] = squad.apply(
            lambda r: own_map.get((r["name"], r["team"]), pd.NA), axis=1,
        )
    else:
        own_map = {}

    md_col = f"ev_md{matchday}"
    xi, formation, bench = pick_xi(squad, md_col)
    captain_rec = optimize_captain(xi, md_col, league_size=league_size,
                                   mode=mode.lower())

    # Differentials
    squad_keys = set(zip(squad["name"], squad["team"]))
    ev_own = ev.copy()
    ev_own["ownership_pct"] = ev_own.apply(
        lambda r: own_map.get((r["name"], r["team"]), pd.NA), axis=1,
    )
    differentials = ev_own[
        ev_own.apply(lambda r: (r["name"], r["team"]) not in squad_keys, axis=1)
        & (ev_own["ownership_pct"].fillna(100) < 5.0)
    ].head(10)

    # Per-MD breakdown
    md_breakdown = []
    point_estimate = 0.0
    for md in range(1, 9):
        col = f"ev_md{md}"
        m_xi, _, _ = pick_xi(squad, col)
        m_rec = optimize_captain(m_xi, col, league_size=league_size,
                                 mode=mode.lower())
        xi_score = float(m_xi[col].sum())
        cb = float(m_xi.loc[m_xi["name"] == m_rec["primary_captain"], col].iloc[0])
        md_breakdown.append({
            "md": md,
            "stage": STAGE_BY_MD[md],
            "captain": m_rec["primary_captain"],
            "score": round(xi_score + cb, 1),
        })
        point_estimate += xi_score + cb

    # Chip optimization (if available — module is built in step 4)
    chip_plan: list[dict] = []
    try:
        from model.chip_optimizer import recommend_chip_sequence
        chip_plan = recommend_chip_sequence(
            ev=ev, squad=squad, sim_summary=sim,
            mode=mode.lower(), league_size=league_size,
        )
        statuses["chip_optimizer"] = {"ok": True, "n_recs": len(chip_plan)}
    except ImportError:
        statuses["chip_optimizer"] = {"ok": False,
                                       "error": "module not built yet"}
    except Exception as exc:  # noqa: BLE001
        log.exception("chip_optimizer failed")
        statuses["chip_optimizer"] = {"ok": False, "error": str(exc)}

    # News digest (if available)
    news_context: dict[str, list[str]] = {}
    news_path = DATA_DIR / "news_context.json"
    if news_path.exists():
        try:
            news_context = json.loads(news_path.read_text())
            statuses["news_context"] = {"ok": True,
                                         "n_players": len(news_context)}
        except Exception as exc:  # noqa: BLE001
            log.exception("news_context load failed")
            statuses["news_context"] = {"ok": False, "error": str(exc)}
    else:
        statuses["news_context"] = {"ok": False, "error": "no news file"}

    # Claude advisor
    advice = None
    if call_advisor:
        try:
            from advisor.claude_client import get_advice
            user_news_blob = _news_to_text(news_context, squad)
            advice = get_advice(
                matchday=matchday, stage=stage, mode=mode.lower(),
                league_size=league_size,
                free_transfers=999 if matchday in (1, 4) else 2,
                chips_remaining=["Wildcard", "12th Man", "Maximum Captain",
                                  "Qualification Booster", "Mystery Booster"],
                squad=squad, xi=xi, formation=formation, bench=bench,
                captain_rec=captain_rec, differentials=differentials,
                user_news=user_news_blob,
                chip_plan=chip_plan,
            )
            statuses["claude_advisor"] = {"ok": True}
        except Exception as exc:  # noqa: BLE001
            log.exception("claude advisor failed")
            statuses["claude_advisor"] = {"ok": False,
                                          "error": f"{type(exc).__name__}: {exc}"}

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "matchday": matchday,
        "stage": stage,
        "mode": mode.lower(),
        "league_size": league_size,
        "data_sources": statuses,
        "formation": list(formation),
        "squad": _df_records(squad),
        "xi": _df_records(xi),
        "bench": _df_records(bench),
        "captain_rec": captain_rec,
        "differentials": _df_records(differentials),
        "md_breakdown": md_breakdown,
        "point_estimate": round(point_estimate, 1),
        "chip_plan": chip_plan,
        "advice": _advice_dict(advice),
    }
    return snapshot


def _news_to_text(news_context: dict[str, list[str]],
                  squad: pd.DataFrame) -> str:
    """Render the news_context dict to free-form text for the advisor
    prompt, limited to players in our squad."""
    if not news_context:
        return ""
    squad_names = set(squad["name"])
    lines = []
    for player, snippets in news_context.items():
        if player not in squad_names:
            continue
        for s in snippets:
            lines.append(f"- {player}: {s}")
    if not lines:
        return ""
    return "Scraped news for squad members:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _write_snapshot(snapshot: dict) -> Path:
    LATEST_JSON.parent.mkdir(parents=True, exist_ok=True)
    LATEST_JSON.write_text(json.dumps(snapshot, indent=2, default=str))
    return LATEST_JSON


def _append_log(snapshot: dict, ok: bool, error: str | None = None) -> None:
    REFRESH_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "matchday": snapshot.get("matchday") if snapshot else None,
        "ok": ok,
        "error": error,
        "data_sources": snapshot.get("data_sources") if snapshot else None,
    }
    with REFRESH_LOG.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--md", type=int, default=None,
                        help="matchday to refresh (default: auto-detect)")
    parser.add_argument("--no-advisor", action="store_true",
                        help="skip the Claude advisor call")
    parser.add_argument("--mode", default="floor", choices=["floor", "ceiling"])
    parser.add_argument("--league-size", type=int, default=15)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    matchday = args.md if args.md is not None else current_matchday()
    log.info("refreshing pipeline for MD%d", matchday)
    snapshot: dict | None = None
    try:
        snapshot = run_refresh(
            matchday=matchday,
            call_advisor=not args.no_advisor,
            mode=args.mode,
            league_size=args.league_size,
        )
        _write_snapshot(snapshot)
        _append_log(snapshot, ok=True)
        log.info("wrote snapshot to %s", LATEST_JSON)
        log.info("point estimate (full tournament): %.1f",
                 snapshot["point_estimate"])
        return 0
    except Exception as exc:  # noqa: BLE001
        log.exception("refresh failed")
        _append_log(snapshot or {"matchday": matchday}, ok=False,
                    error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
