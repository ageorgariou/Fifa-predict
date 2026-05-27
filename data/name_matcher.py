"""Cross-source player matching: FIFA Fantasy ↔ FBref ↔ goalscorer odds.

The hard constraint is (name, national team). Once the country is fixed we
only ever search a pool of 20-35 club-league players from that nation, so
fuzzy matching by name with a rapidfuzz cutoff of 95 is highly reliable.

Matching tiers
--------------
score ≥ 95   →  auto-match
85 ≤ score < 95  →  review queue (written to data/match_review.csv)
score < 85   →  unmatched (written to data/match_unmatched.csv)

Manual overrides (data/name_overrides.json)
-------------------------------------------
    {"FIFA Name|FIFA Team": "FBref Name"}
Hits bypass fuzzy matching entirely. Keep this file under version control.

Public API
----------
    match_fifa_to_fbref(fifa_df, fbref_df, ...) ->
        (matched_df, review_df, unmatched_df)
"""
from __future__ import annotations

import json
import logging

import pandas as pd
from rapidfuzz import fuzz, process

from config import DATA_DIR
from model.groups_2026 import _fold, canonical_team, country_to_team

log = logging.getLogger(__name__)

NAME_OVERRIDES_PATH = DATA_DIR / "name_overrides.json"
REVIEW_CSV = DATA_DIR / "match_review.csv"
UNMATCHED_CSV = DATA_DIR / "match_unmatched.csv"


def _load_overrides() -> dict[str, str]:
    if not NAME_OVERRIDES_PATH.exists():
        return {}
    try:
        return json.loads(NAME_OVERRIDES_PATH.read_text())
    except json.JSONDecodeError:
        log.warning("name_overrides.json is corrupt — ignoring")
        return {}


def _attach_country(fbref_df: pd.DataFrame) -> pd.DataFrame:
    """Annotate FBref rows with the canonical WC team name derived from their
    `nation` 3-letter code. Rows whose nation isn't in the 48-team map keep
    NaN here and are skipped during matching (player isn't going to the WC)."""
    out = fbref_df.copy()
    out["national_team"] = out["nation"].map(country_to_team)
    return out


def match_fifa_to_fbref(
    fifa_df: pd.DataFrame,
    fbref_df: pd.DataFrame,
    threshold_auto: int = 95,
    threshold_review: int = 85,
    write_csvs: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """For each FIFA player, locate the highest-scoring FBref row whose
    `national_team` equals the FIFA player's `team`. See module docstring."""
    overrides = _load_overrides()
    annotated = _attach_country(fbref_df)
    pool_by_country: dict[str, pd.DataFrame] = {
        team: g for team, g in annotated.groupby("national_team", dropna=True)
    }

    matched_rows: list[dict] = []
    review_rows: list[dict] = []
    unmatched_rows: list[dict] = []

    fbref_stat_cols = [c for c in fbref_df.columns if c not in ("league", "season", "team", "player")]

    for _, fp in fifa_df.iterrows():
        fifa_name = fp["name"]
        fifa_team_raw = fp["team"]
        if not isinstance(fifa_name, str) or not isinstance(fifa_team_raw, str):
            continue
        # Normalize FIFA team to canonical (handles "USA" → "United States",
        # "Türkiye" → "Turkiye" etc.). After step-13 fix, fifa_team should
        # already be canonical at the source, but we keep this for safety.
        fifa_team = canonical_team(fifa_team_raw) or fifa_team_raw

        # 1) Manual override
        override_key = f"{fifa_name}|{fifa_team}"
        override_target = overrides.get(override_key)

        # 2) Country pool
        pool = pool_by_country.get(fifa_team)
        if pool is None or pool.empty:
            unmatched_rows.append({
                "fifa_name": fifa_name, "fifa_team": fifa_team,
                "fifa_position": fp.get("position"), "reason": "no FBref players from this country",
            })
            continue

        # 3) Match
        if override_target:
            hit = pool[pool["player"] == override_target]
            if hit.empty:
                hit = pool[pool["player"].fillna("").map(_fold) == _fold(override_target)]
            if hit.empty:
                unmatched_rows.append({
                    "fifa_name": fifa_name, "fifa_team": fifa_team,
                    "fifa_position": fp.get("position"),
                    "reason": f"override target '{override_target}' not in FBref pool",
                })
                continue
            best_row = hit.iloc[0]
            best_score = 100
            via = "override"
        else:
            choices = pool["player"].fillna("").tolist()
            best = process.extractOne(
                fifa_name, choices, scorer=fuzz.WRatio, processor=_fold,
            )
            if best is None:
                unmatched_rows.append({
                    "fifa_name": fifa_name, "fifa_team": fifa_team,
                    "fifa_position": fp.get("position"), "reason": "no candidates",
                })
                continue
            best_name, best_score, best_idx = best
            best_row = pool.iloc[best_idx]
            via = "fuzzy"

        # 4) Tier
        if best_score >= threshold_auto or via == "override":
            row = {
                "fifa_id": fp.get("id"),
                "fifa_name": fifa_name,
                "fifa_team": fifa_team,
                "fifa_position": fp.get("position"),
                "fifa_price": fp.get("price"),
                "fifa_ownership_pct": fp.get("ownership_pct"),
                "fbref_player": best_row["player"],
                "fbref_club": best_row["team"],
                "match_score": int(best_score),
                "match_via": via,
            }
            for c in fbref_stat_cols:
                row[c] = best_row.get(c)
            matched_rows.append(row)
        elif best_score >= threshold_review:
            review_rows.append({
                "fifa_name": fifa_name,
                "fifa_team": fifa_team,
                "fifa_position": fp.get("position"),
                "fifa_price": fp.get("price"),
                "fbref_candidate": best_row["player"],
                "fbref_club": best_row["team"],
                "score": int(best_score),
            })
        else:
            unmatched_rows.append({
                "fifa_name": fifa_name, "fifa_team": fifa_team,
                "fifa_position": fp.get("position"),
                "reason": f"best score {int(best_score)} < {threshold_review}",
                "best_candidate": best_row["player"],
                "fbref_club": best_row["team"],
            })

    matched_df = pd.DataFrame(matched_rows)
    review_df = pd.DataFrame(review_rows)
    unmatched_df = pd.DataFrame(unmatched_rows)

    if write_csvs:
        if not review_df.empty:
            review_df.sort_values("score", ascending=False).to_csv(REVIEW_CSV, index=False)
        else:
            REVIEW_CSV.write_text("fifa_name,fifa_team,fifa_position,fifa_price,fbref_candidate,fbref_club,score\n")
        if not unmatched_df.empty:
            unmatched_df.to_csv(UNMATCHED_CSV, index=False)
        else:
            UNMATCHED_CSV.write_text("fifa_name,fifa_team,fifa_position,reason\n")

    return matched_df, review_df, unmatched_df


if __name__ == "__main__":
    from data.fbref_stats import fetch_outfield_stats
    from data.fifa_players import fetch_player_list

    fifa = fetch_player_list()
    fbref = fetch_outfield_stats()
    # Skip goalkeepers — they're in fetch_gk_stats, not the outfield frame.
    fifa_outfield = fifa[fifa["position"].isin(["DEF", "MID", "FWD"])]

    matched, review, unmatched = match_fifa_to_fbref(fifa_outfield, fbref)

    print(f"FIFA outfield players:  {len(fifa_outfield)}")
    print(f"  auto-matched (≥95):   {len(matched)}")
    print(f"  review (85–94):       {len(review)}")
    print(f"  unmatched (<85):      {len(unmatched)}")
    print()

    # Unmatched breakdown — most will be 'no FBref players from this country',
    # i.e. squad members who play in MLS / Saudi PL / domestic leagues.
    if not unmatched.empty:
        print("Unmatched by reason:")
        print(unmatched["reason"].value_counts().to_string())
        print()

    # Show the full review CSV — these are the borderline calls that need
    # a human sanity check before promotion to auto-match.
    print(f"=== data/match_review.csv (n={len(review)}) ===")
    if review.empty:
        print("(empty — no borderline matches)")
    else:
        print(review.sort_values("score", ascending=False).to_string(index=False))
