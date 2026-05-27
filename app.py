"""Streamlit UI for the WC Fantasy Advisor.

Run locally:  .venv/bin/streamlit run app.py
"""
from __future__ import annotations

from datetime import date
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from advisor.claude_client import get_advice
from config import NATION_CAP
from data.cache import cache_age
from data.cache import clear as clear_cache_db
from data.fifa_players import fetch_player_list
from model.captain_optimizer import optimize_captain
from model.ev import compute_ev
from model.optimizer import optimize_squad, pick_xi
from model.projections import compute_ep90
from model.score_estimate import (
    CONFIDENCE_BAND, GLOBAL_TIERS, OFFICE_TIERS, _place_in_tiers,
)
from model.tournament_sim import simulate

# ---------------------------------------------------------------------------
# Matchday calendar (auto-detect)
# ---------------------------------------------------------------------------
MD_DATES = {
    1: date(2026, 6, 11),
    2: date(2026, 6, 16),
    3: date(2026, 6, 22),
    4: date(2026, 6, 28),
    5: date(2026, 7, 4),
    6: date(2026, 7, 9),
    7: date(2026, 7, 14),
    8: date(2026, 7, 19),
}
STAGE_BY_MD = {1: "GROUP", 2: "GROUP", 3: "GROUP", 4: "R32",
               5: "R16", 6: "QF", 7: "SF", 8: "F"}
DEFAULT_TRANSFERS = {1: 999, 2: 2, 3: 2, 4: 999, 5: 4, 6: 4, 7: 5, 8: 6}
ALL_CHIPS = ["Wildcard", "12th Man", "Maximum Captain",
             "Qualification Booster", "Mystery Booster"]


def auto_detect_md() -> int:
    today = date.today()
    if today < MD_DATES[1]:
        return 1
    for md in range(8, 0, -1):
        if today >= MD_DATES[md]:
            return md
    return 1


# ---------------------------------------------------------------------------
# Cached pipeline pieces (Streamlit caches across reruns)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=60 * 60)
def cached_ev_table() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Projection + simulator + EV. Returned tuple is the heaviest part of
    the pipeline; cached for 1h."""
    proj = compute_ep90()
    sim = simulate()
    ev = compute_ev(proj, sim)
    return proj, sim, ev


@st.cache_data(show_spinner=False, ttl=60 * 60)
def cached_squad(stage: str) -> pd.DataFrame:
    _, _, ev = cached_ev_table()
    return optimize_squad(ev, nation_cap=NATION_CAP[stage])


@st.cache_data(show_spinner=False, ttl=60 * 60)
def cached_fifa() -> pd.DataFrame:
    return fetch_player_list()


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
PITCH_GREEN = "#0e3a1f"


def draw_pitch(xi: pd.DataFrame, captain: str, vice: str,
               formation: tuple[int, int, int, int]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 10), facecolor=PITCH_GREEN)
    ax.set_facecolor(PITCH_GREEN)

    # Pitch markings
    ax.add_patch(mpatches.Rectangle((0, 0), 100, 100, ec="white",
                                    fc=PITCH_GREEN, lw=2))
    ax.plot([0, 100], [50, 50], color="white", lw=1.5)
    ax.add_patch(mpatches.Circle((50, 50), 10, ec="white", fc="none", lw=1.5))
    ax.add_patch(mpatches.Rectangle((25, 0), 50, 16, ec="white", fc="none", lw=1.5))
    ax.add_patch(mpatches.Rectangle((25, 84), 50, 16, ec="white", fc="none", lw=1.5))
    ax.add_patch(mpatches.Rectangle((40, 0), 20, 6, ec="white", fc="none", lw=1.5))
    ax.add_patch(mpatches.Rectangle((40, 94), 20, 6, ec="white", fc="none", lw=1.5))

    # Row layout — formation = (GK, DEF, MID, FWD)
    rows = [
        ("GK", formation[0], 10),
        ("DEF", formation[1], 30),
        ("MID", formation[2], 55),
        ("FWD", formation[3], 80),
    ]
    by_pos = {pos: xi[xi["position"] == pos].reset_index(drop=True)
              for pos in ("GK", "DEF", "MID", "FWD")}

    def _xs(n: int) -> list[float]:
        return [(i + 1) * 100 / (n + 1) for i in range(n)]

    for pos, n, y in rows:
        for i, x in enumerate(_xs(n)):
            if i >= len(by_pos[pos]):
                continue
            p = by_pos[pos].iloc[i]
            name = p["name"]
            is_capt = (name == captain)
            is_vice = (name == vice)
            fill = "#FFD700" if is_capt else "#C0C0C0" if is_vice else "white"
            ax.scatter(x, y, s=900, c=fill, edgecolors="black",
                       linewidths=1.5, zorder=10)
            badge = "C" if is_capt else "VC" if is_vice else ""
            if badge:
                ax.annotate(badge, (x, y), ha="center", va="center",
                            fontsize=10, fontweight="bold",
                            color="black", zorder=11)
            # Player name below
            ax.annotate(name, (x, y - 5), ha="center", va="top",
                        fontsize=9, color="white", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  fc="black", ec="none", alpha=0.6),
                        zorder=12)

    ax.set_xlim(-3, 103)
    ax.set_ylim(-8, 105)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


def draw_tier_bar(point: float, lo: float, hi: float,
                  tiers: list[dict], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 1.6))
    tiers_sorted = sorted(tiers, key=lambda t: t["lo"])
    palette = ["#fa5252", "#ff922b", "#fcc419", "#94d82d", "#37b24d"]
    for t, color in zip(tiers_sorted, palette):
        ax.axvspan(t["lo"], t["hi"], alpha=0.55, color=color,
                   label=f"{t['name']}  ({t['lo']:.0f}-{t['hi']:.0f})")

    # CI band + point marker
    ax.axvspan(lo, hi, color="black", alpha=0.18, zorder=8)
    ax.axvline(point, color="black", lw=3, zorder=10)
    ax.annotate(f"  {point:.0f}", (point, 0.5), xycoords=("data", "axes fraction"),
                ha="left", va="center", fontsize=12, fontweight="bold")

    overall_lo = min(t["lo"] for t in tiers_sorted)
    overall_hi = max(t["hi"] for t in tiers_sorted)
    ax.set_xlim(overall_lo, overall_hi)
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3),
              ncol=min(len(tiers_sorted), 3), fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-click pipeline runner
# ---------------------------------------------------------------------------
def run_full_pipeline(
    *,
    matchday: int,
    stage: str,
    mode: str,
    league_size: int,
    n_sharp: int,
    opponent_captains: list[str] | None,
    user_news: str,
    free_transfers: int,
    chips: list[str],
    call_advisor: bool,
) -> dict:
    proj, sim, ev = cached_ev_table()
    fifa = cached_fifa()

    squad = cached_squad(stage).copy()
    own_map = fifa.set_index(["name", "team"])["ownership_pct"].to_dict()
    squad["ownership_pct"] = squad.apply(
        lambda r: own_map.get((r["name"], r["team"]), pd.NA), axis=1
    )

    md_col = f"ev_md{matchday}"
    xi, formation, bench = pick_xi(squad, md_col)
    captain_rec = optimize_captain(
        xi, md_col,
        league_size=league_size,
        n_sharp_opponents=n_sharp,
        opponent_captains=opponent_captains,
        mode=mode.lower(),
    )

    # Differentials: high-EV, low-owned, not already in squad
    squad_keys = set(zip(squad["name"], squad["team"]))
    ev_with_own = ev.copy()
    ev_with_own["ownership_pct"] = ev_with_own.apply(
        lambda r: own_map.get((r["name"], r["team"]), pd.NA), axis=1
    )
    differentials = ev_with_own[
        ev_with_own.apply(lambda r: (r["name"], r["team"]) not in squad_keys, axis=1)
        & (ev_with_own["ownership_pct"].fillna(100) < 5.0)
    ].head(10)

    # Per-MD score (the system's projection for THIS matchday)
    xi_score = float(xi[md_col].sum())
    captain_bonus = float(xi.loc[xi["name"] == captain_rec["primary_captain"], md_col].iloc[0])
    md_predicted = xi_score + captain_bonus

    # Full-tournament projection
    breakdown_rows = []
    point_estimate = 0.0
    for md in range(1, 9):
        col = f"ev_md{md}"
        md_xi, md_form, _ = pick_xi(squad, col)
        md_rec = optimize_captain(
            md_xi, col, league_size=league_size, mode=mode.lower(),
        )
        s = float(md_xi[col].sum())
        cb = float(md_xi.loc[md_xi["name"] == md_rec["primary_captain"], col].iloc[0])
        breakdown_rows.append({"md": md, "captain": md_rec["primary_captain"],
                               "score": round(s + cb, 1)})
        point_estimate += s + cb
    ci_lo = point_estimate * (1 - CONFIDENCE_BAND)
    ci_hi = point_estimate * (1 + CONFIDENCE_BAND)
    office_tier = _place_in_tiers(point_estimate, OFFICE_TIERS)
    global_tier = _place_in_tiers(point_estimate, GLOBAL_TIERS)

    out = {
        "matchday": matchday,
        "stage": stage,
        "squad": squad,
        "xi": xi,
        "formation": formation,
        "bench": bench,
        "captain_rec": captain_rec,
        "differentials": differentials,
        "md_predicted": md_predicted,
        "xi_score": xi_score,
        "captain_bonus": captain_bonus,
        "point_estimate": point_estimate,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "office_tier": office_tier,
        "global_tier": global_tier,
        "md_breakdown": pd.DataFrame(breakdown_rows),
        "advice": None,
        "advice_error": None,
    }

    if call_advisor:
        try:
            out["advice"] = get_advice(
                matchday=matchday, stage=stage, mode=mode.lower(),
                league_size=league_size, free_transfers=free_transfers,
                chips_remaining=chips, squad=squad, xi=xi, formation=formation,
                bench=bench, captain_rec=captain_rec, differentials=differentials,
                user_news=user_news, opponent_captains=opponent_captains,
            )
        except Exception as exc:  # noqa: BLE001 — advisor failures shouldn't crash UI
            out["advice_error"] = str(exc)
    return out


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="WC Fantasy Advisor", page_icon=None,
                   layout="wide", initial_sidebar_state="expanded")
st.title("World Cup Fantasy Advisor")

# === Always-visible pre-deadline checklist ===
st.warning(
    "**Pre-deadline checklist — 30 minutes before kickoff:**\n\n"
    "1. **sofascore.com** — open predicted lineups for every match in this matchday.\n"
    "2. **Twitter/X** — quick scan of @FabrizioRomano, @David_Ornstein, and the "
    "beat reporters for each national team in your XI.\n"
    "3. **Paste any new news** into the **Injuries / news** box in the sidebar.\n"
    "4. **Click _Get this round's picks_ again** — the advisor uses your news to "
    "validate or override the model.",
    icon=None,
)

# === Sidebar ===
with st.sidebar:
    st.header("Settings")

    mode = st.radio(
        "Mode",
        ["Floor", "Ceiling"],
        index=0,
        help=("Floor = safer; match the field's likely captain.\n"
              "Ceiling = swing for the league win; favor contrarian picks."),
    )

    matchday = st.number_input(
        "Current matchday", min_value=1, max_value=8,
        value=auto_detect_md(),
        help="Auto-detected from today's date; override if needed.",
    )
    stage = STAGE_BY_MD[matchday]
    st.caption(f"Stage: **{stage}**")

    league_size = st.number_input("League size", min_value=2, max_value=10_000,
                                  value=15)
    n_sharp = st.number_input(
        "Sharp opponents", min_value=1, max_value=int(league_size),
        value=int(league_size) - 1,
        help="How many opponents you assume are also running optimization-style picks.",
    )

    opp_captains_raw = st.text_input(
        "Known opponent captains (comma-separated)",
        help="Leave blank if you don't know. Used to refine contrarian recommendations.",
    )
    opp_captains: list[str] | None = None
    if opp_captains_raw.strip():
        opp_captains = [s.strip() for s in opp_captains_raw.split(",") if s.strip()]

    free_transfers = st.number_input(
        "Free transfers", min_value=0, max_value=999,
        value=DEFAULT_TRANSFERS.get(matchday, 2),
    )
    chips = st.multiselect("Chips remaining", ALL_CHIPS, default=ALL_CHIPS)

    user_news = st.text_area(
        "Injuries / news (paste before deadline)",
        height=140,
        placeholder="Example:\n- Kane carrying a knock, 50/50 to start\n"
                    "- Argentina expected to rest Lautaro in MD3",
    )

    st.divider()
    if st.button("Refresh data"):
        cleared = clear_cache_db()
        st.cache_data.clear()
        st.success(f"Cleared {cleared} cache rows.")

    with st.expander("Model accuracy"):
        st.markdown(
            "Validated on **2018 + 2022 WC** via the projection pipeline:\n\n"
            "- Pooled Pearson r = **0.691**\n"
            "- 2018: r = 0.749 (46 matched players)\n"
            "- 2022: r = 0.630 (42 matched players)\n\n"
            "Stat-only baseline; live deployment also uses priors + "
            "odds-blending when bookmakers open WC player markets "
            "(~5 days before MD1)."
        )

# === Main run button ===
run_clicked = st.button("Get this round's picks", type="primary",
                        use_container_width=True)

if run_clicked:
    with st.spinner("Thinking…"):
        try:
            st.session_state.results = run_full_pipeline(
                matchday=int(matchday), stage=stage, mode=mode,
                league_size=int(league_size), n_sharp=int(n_sharp),
                opponent_captains=opp_captains, user_news=user_news,
                free_transfers=int(free_transfers), chips=chips,
                call_advisor=True,
            )
        except Exception as exc:  # noqa: BLE001
            elo_age = cache_age("elo:world")
            cache_msg = (f" Last cached data ≈ {elo_age/3600:.1f}h old."
                         if elo_age else "")
            st.error(
                f"Pipeline failed: {exc}.{cache_msg} "
                f"Try **Refresh data** in the sidebar, or check your "
                f"ANTHROPIC_API_KEY / ODDS_API_KEY."
            )

# === Render results ===
results: dict[str, Any] | None = st.session_state.get("results")
if results:
    # --- 1. Projected score card ---
    st.subheader("Projected final-tournament score")
    score_col, _ = st.columns([3, 1])
    with score_col:
        st.metric(
            label="Point estimate (full tournament)",
            value=f"{results['point_estimate']:.0f}",
            delta=f"80% CI: {results['ci_lo']:.0f} – {results['ci_hi']:.0f}",
            delta_color="off",
        )
    st.caption(
        f"Office league: **{results['office_tier']}**  |  "
        f"Global: **{results['global_tier']}**"
    )
    fig = draw_tier_bar(
        results["point_estimate"], results["ci_lo"], results["ci_hi"],
        OFFICE_TIERS, "Office league benchmark",
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    fig = draw_tier_bar(
        results["point_estimate"], results["ci_lo"], results["ci_hi"],
        GLOBAL_TIERS, "Global benchmark",
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # --- 2. Football pitch with XI ---
    st.divider()
    formation = results["formation"]
    fmt_str = f"{formation[1]}-{formation[2]}-{formation[3]}"
    st.subheader(f"MD{results['matchday']} XI — {fmt_str}")
    pitch_col, info_col = st.columns([3, 2])
    with pitch_col:
        fig = draw_pitch(
            results["xi"],
            results["captain_rec"]["primary_captain"],
            results["captain_rec"]["vice_captain"],
            results["formation"],
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with info_col:
        st.markdown(f"**Captain:** {results['captain_rec']['primary_captain']}  ")
        st.markdown(f"**Vice:** {results['captain_rec']['vice_captain']}  ")
        st.markdown(
            f"**Squad cost:** ${results['squad']['price'].sum():.1f}M / $100M  "
        )
        st.markdown(
            f"**MD{results['matchday']} predicted:** "
            f"{results['md_predicted']:.1f} "
            f"({results['xi_score']:.1f} XI + {results['captain_bonus']:.1f} captain)"
        )

        # Bench
        st.markdown("**Bench (auto-sub order):**")
        bench = results["bench"].reset_index(drop=True)
        for i, b in bench.iterrows():
            label = "First sub" if i == 0 else f"Sub {i + 1}"
            st.markdown(f"- **{label}:** {b['name']} ({b['team']}, {b['position']})")

    # --- 3. Captain analysis ---
    with st.expander("Captain analysis"):
        st.markdown(f"_{results['captain_rec']['captain_reasoning']}_")
        table = pd.DataFrame(results["captain_rec"]["captain_ev_table"])
        table["est_ownership"] = (table["est_ownership"] * 100).round(0).astype(str) + "%"
        table["p_beat_field"] = (table["p_beat_field"] * 100).round(1).astype(str) + "%"
        st.dataframe(
            table.head(5)[["name", "ep_round", "est_ownership",
                           "p_beat_field", "expected_rank_gain"]],
            hide_index=True, use_container_width=True,
        )

    # --- 4. Why these picks? (advice + risks) ---
    advice = results.get("advice")
    if advice:
        with st.expander("Why these picks?", expanded=True):
            st.markdown(f"**Overall strategy:** {advice.overall_strategy}")
            st.markdown(f"**Captain reasoning:** {advice.captain_reasoning}")
            if advice.key_risks:
                st.markdown("**Key risks:**")
                for r in advice.key_risks:
                    st.markdown(f"- {r}")
    elif results.get("advice_error"):
        st.warning(
            f"Advisor call failed — falling back to model-only output. "
            f"({results['advice_error'][:120]})"
        )

    # --- 5. Differentials (Ceiling mode only) ---
    if mode == "Ceiling":
        with st.expander("Differential picks", expanded=True):
            diffs = results["differentials"][
                ["name", "team", "position", "price",
                 "ownership_pct", "ep90", "ev_total"]
            ]
            st.dataframe(diffs, hide_index=True, use_container_width=True)
            if advice and advice.differentials:
                st.markdown("**Advisor-flagged differentials:**")
                for d in advice.differentials:
                    st.markdown(f"- {d}")

    # --- 6. Chip recommendation ---
    if advice and advice.chip_recommendation:
        st.info(f"**Chip recommendation:** {advice.chip_recommendation}")

    # --- 7. Per-MD projection breakdown (debug-y but useful) ---
    with st.expander("Per-matchday breakdown"):
        st.dataframe(results["md_breakdown"], hide_index=True, use_container_width=True)
else:
    st.info(
        "Set your sidebar options, then click **Get this round's picks** above. "
        "First run takes 1-3 minutes while we pull stats and run 10,000 "
        "tournament simulations; subsequent runs use cache and complete in seconds."
    )
