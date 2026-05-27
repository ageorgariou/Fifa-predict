"""Claude advisor client.

Model: claude-opus-4-7 (config.CLAUDE_MODEL).

For one "Get picks" press, we make exactly one API call. The call uses
tool_use with the Advice pydantic schema so Claude is forced to return a
structurally-valid response. We retry once on validation failure.

Inputs threaded through:
    * Current matchday + stage
    * LP-optimal 15-man squad (with EV, price, ownership)
    * Recommended XI + formation (from optimizer.pick_xi)
    * Captain analysis (from captain_optimizer.optimize_captain)
    * Top differentials (high EV, low ownership)
    * Floor / Ceiling mode
    * Free transfers + chips remaining
    * User-pasted injury / news text (free-form)
    * Optional opponent-captain list

The advisor's job is NOT to redo the optimization (we've already done that).
It's to: validate the recommendation against the injury news, explain
captain logic in plain English, surface the top risks, suggest differentials,
and recommend chip timing.

Public API
----------
    get_advice(...) -> Advice
"""
from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd
import pydantic
from anthropic import Anthropic

from advisor.schemas import Advice
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL

log = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert FIFA World Cup 2026 Fantasy advisor.

You're advising a football-literate but non-fantasy-expert user. Use plain
English — avoid fantasy-football jargon, acronyms, or model talk. Be
specific about names and reasons.

The optimization pipeline has ALREADY chosen an LP-optimal 15-man squad,
an XI for the upcoming matchday, and a captain. Your job:

1. VALIDATE — only override the LP's XI / captain if the user-supplied
   injury or news text gives a specific reason. Do NOT second-guess the
   model just to feel useful.
2. EXPLAIN — captain reasoning in 2-3 plain sentences.
3. RISK — name 2-4 concrete matchday risks (injuries, suspensions,
   rotation, weak fixtures for our XI).
4. DIFFERENTIAL — up to 3 low-ownership picks the user could pivot to in
   Ceiling mode, or skip if Floor mode and the recommendation already
   looks solid.
5. CHIP — recommend a chip if appropriate for this matchday + stage:
   - Wildcard: full squad rewrite. Best used before MD1 or after group
     stage if your squad is mostly out. NOT available for the R32 round.
   - 12th Man: an extra autosub. Useful when bench has high-floor players.
   - Maximum Captain: captain × 3 instead of × 2. Use on a near-certain
     starter against a weak opponent in a high-confidence MD.
   - Qualification Booster (R32+): +2 to XI players whose team progresses.
   - Mystery Booster (R32+): unrevealed bonus.
6. STRATEGY — one or two sentences capturing the overall plan.

Always submit your final answer using the `submit_advice` tool."""


ADVICE_TOOL: dict[str, Any] = {
    "name": "submit_advice",
    "description": "Submit your final World Cup fantasy advice for this matchday.",
    "input_schema": Advice.model_json_schema(),
}


def _df_to_lines(df: pd.DataFrame, cols: list[str]) -> str:
    """Compact text rendering — one player per line, padded for readability."""
    if df is None or df.empty:
        return "(none)"
    return df[cols].to_string(index=False)


def _build_user_prompt(
    *,
    matchday: int,
    stage: str,
    mode: str,
    league_size: int,
    free_transfers: int,
    chips_remaining: list[str],
    squad: pd.DataFrame,
    xi: pd.DataFrame,
    formation: tuple[int, int, int, int],
    bench: pd.DataFrame,
    captain_rec: dict,
    differentials: pd.DataFrame,
    user_news: str,
    opponent_captains: list[str] | None,
) -> str:
    fmt_str = f"{formation[1]}-{formation[2]}-{formation[3]}"
    md_col = f"ev_md{matchday}"

    squad_lines = _df_to_lines(
        squad.sort_values("ev_total", ascending=False),
        ["name", "team", "position", "price", "ep90", "ownership_pct",
         md_col, "ev_total"],
    )
    xi_lines = _df_to_lines(
        xi.sort_values(md_col, ascending=False),
        ["name", "team", "position", "price", md_col],
    )
    bench_lines = _df_to_lines(
        bench, ["name", "team", "position", "price", md_col],
    )
    diff_lines = _df_to_lines(
        differentials.head(8),
        ["name", "team", "position", "price", "ownership_pct", "ep90", "ev_total"],
    )
    captain_table = pd.DataFrame(captain_rec["captain_ev_table"])
    captain_lines = _df_to_lines(
        captain_table, ["name", "ep_round", "est_ownership",
                        "p_beat_field", "expected_rank_gain"],
    )

    opp = ", ".join(opponent_captains) if opponent_captains else "(none provided)"
    chips_str = ", ".join(chips_remaining) if chips_remaining else "(none)"

    return f"""\
TOURNAMENT CONTEXT
- Matchday: MD{matchday} ({stage})
- Mode: {mode}  (Floor = safer; Ceiling = swing for league win)
- League size: {league_size}
- Free transfers available: {free_transfers}
- Chips remaining: {chips_str}

LP-OPTIMAL 15-MAN SQUAD (sorted by tournament EV):
{squad_lines}

RECOMMENDED STARTING XI ({fmt_str}):
{xi_lines}

BENCH (auto-sub order):
{bench_lines}

CAPTAIN ANALYSIS
Recommendation:
  Primary: {captain_rec['primary_captain']}
  Vice:    {captain_rec['vice_captain']}
  Model reasoning: {captain_rec['captain_reasoning']}

Full captain candidate table:
{captain_lines}

TOP DIFFERENTIAL CANDIDATES (high EV, low ownership):
{diff_lines}

USER-SUPPLIED NEWS / INJURY UPDATES:
{user_news or "(none provided)"}

OPPONENT CAPTAINS (if known):
{opp}

Please review the recommendation against the news, then submit your final
advice via the `submit_advice` tool. The XI must be 11 of the 15 squad
members; bench_order must be the other 4 in best-sub-first order.
"""


def get_advice(
    *,
    matchday: int,
    stage: str,
    mode: str,
    league_size: int,
    free_transfers: int,
    chips_remaining: list[str],
    squad: pd.DataFrame,
    xi: pd.DataFrame,
    formation: tuple[int, int, int, int],
    bench: pd.DataFrame,
    captain_rec: dict,
    differentials: pd.DataFrame,
    user_news: str = "",
    opponent_captains: list[str] | None = None,
    max_tokens: int = 4096,
) -> Advice:
    """Make one Claude call and parse the structured response. Retries once
    on pydantic validation failure with a clarifying follow-up message."""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set in env / Streamlit secrets — "
            "advisor cannot run."
        )

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    user_prompt = _build_user_prompt(
        matchday=matchday, stage=stage, mode=mode, league_size=league_size,
        free_transfers=free_transfers, chips_remaining=chips_remaining,
        squad=squad, xi=xi, formation=formation, bench=bench,
        captain_rec=captain_rec, differentials=differentials,
        user_news=user_news, opponent_captains=opponent_captains,
    )

    messages: list[dict] = [{"role": "user", "content": user_prompt}]

    for attempt in range(2):
        log.info("Calling %s (attempt %d/2)", CLAUDE_MODEL, attempt + 1)
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            tools=[ADVICE_TOOL],
            tool_choice={"type": "tool", "name": "submit_advice"},
            messages=messages,
        )
        tool_input = None
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "submit_advice":
                tool_input = block.input
                break
        if tool_input is None:
            log.warning("no tool_use block returned; attempt %d", attempt + 1)
            messages.append({"role": "assistant", "content": resp.content})
            messages.append({
                "role": "user",
                "content": "You did not call the submit_advice tool. "
                           "Please call it now with your final answer.",
            })
            continue
        try:
            return Advice(**tool_input)
        except pydantic.ValidationError as e:
            log.warning("Advice validation failed (attempt %d): %s", attempt + 1, e)
            if attempt == 1:
                raise
            messages.append({"role": "assistant", "content": resp.content})
            messages.append({
                "role": "user",
                "content": (
                    "Your submit_advice call did not validate against the schema. "
                    f"Errors: {e}. Please call submit_advice again with corrected fields."
                ),
            })
    raise RuntimeError("advisor: exhausted retries without a valid Advice")


if __name__ == "__main__":
    # End-to-end test: run the full pipeline, build the advisor inputs,
    # call Claude once, print the Advice.
    from model.captain_optimizer import optimize_captain
    from model.ev import compute_ev
    from model.optimizer import optimize_squad, pick_xi
    from model.projections import compute_ep90
    from model.tournament_sim import simulate
    from data.fifa_players import fetch_player_list

    log.info("Building advisor inputs ...")
    proj = compute_ep90()
    sim = simulate()
    ev = compute_ev(proj, sim)
    squad = optimize_squad(ev)

    matchday = 1
    md_col = f"ev_md{matchday}"
    xi, formation, bench = pick_xi(squad, md_col)
    captain_rec = optimize_captain(xi, md_col, league_size=15, mode="floor")

    # Differentials: low-owned, high-EV players NOT in the current squad
    fifa = fetch_player_list()
    own_map = fifa.set_index(["name", "team"])["ownership_pct"].to_dict()
    ev_with_own = ev.copy()
    ev_with_own["ownership_pct"] = ev_with_own.apply(
        lambda r: own_map.get((r["name"], r["team"]), pd.NA), axis=1
    )
    squad_keys = set(zip(squad["name"], squad["team"]))
    differentials = ev_with_own[
        ev_with_own.apply(
            lambda r: (r["name"], r["team"]) not in squad_keys, axis=1
        )
        & (ev_with_own["ownership_pct"].fillna(100) < 5.0)
    ].head(8)

    # Add ownership to squad for the prompt
    squad = squad.copy()
    squad["ownership_pct"] = squad.apply(
        lambda r: own_map.get((r["name"], r["team"]), pd.NA), axis=1
    )

    print("Calling Claude (claude-opus-4-7) ...")
    advice = get_advice(
        matchday=matchday,
        stage="GROUP",
        mode="floor",
        league_size=15,
        free_transfers=999,        # unlimited before MD1
        chips_remaining=["Wildcard", "12th Man", "Maximum Captain",
                         "Qualification Booster", "Mystery Booster"],
        squad=squad,
        xi=xi,
        formation=formation,
        bench=bench,
        captain_rec=captain_rec,
        differentials=differentials,
        user_news="",
    )

    print()
    print("=" * 72)
    print("ADVICE")
    print("=" * 72)
    print(f"Formation:        {advice.formation}")
    print(f"Captain:          {advice.captain}")
    print(f"Vice-captain:     {advice.vice_captain}")
    print(f"Chip recommendation: {advice.chip_recommendation}")
    print()
    print(f"Captain reasoning:")
    print(f"  {advice.captain_reasoning}")
    print()
    print(f"Starting XI ({len(advice.starting_xi)}):")
    for n in advice.starting_xi:
        print(f"  - {n}")
    print()
    print(f"Bench order ({len(advice.bench_order)}):")
    for n in advice.bench_order:
        print(f"  - {n}")
    print()
    print(f"Key risks:")
    for r in advice.key_risks:
        print(f"  - {r}")
    print()
    print(f"Differentials:")
    for d in advice.differentials:
        print(f"  - {d}")
    print()
    print(f"Overall strategy:")
    print(f"  {advice.overall_strategy}")
