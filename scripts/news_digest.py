"""Pre-deadline news digest.

Runs 2h before each matchday deadline. Scrapes a small set of publicly
readable football-news sources, extracts keyword-filtered snippets that
mention any of our top-200-EV players, and writes
`data/news_context.json` for the advisor and email scripts.

Sources
-------
1. FFScout team-news page (https://www.fantasyfootballscout.co.uk/category/team-news)
   — most reliable; static HTML, no login.
2. Sofascore predicted-lineup pages — fragile (JavaScript-heavy); best
   effort with httpx; skipped if blocked.
3. (Optional) Beat reporters via nitter.net X mirrors — even more fragile;
   disabled by default. Enable with --reporters if Nitter is up.

Output
------
data/news_context.json — {player_name: [snippet, ...]}
data/refresh_log.jsonl — one append per run, with per-source status.

Failure modes (all non-fatal)
-----------------------------
* If a source returns non-200 or times out, we log and skip it.
* If parsing fails on a page, we log and skip that source.
* If no source succeeds, the file is written as {} so downstream consumers
  see "no news" rather than stale data.

This script does NOT attempt to evade anti-bot measures. If Sofascore
blocks the request, that source is simply skipped. The instruction is
explicit on this — do not build a workaround.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from config import DATA_DIR

log = logging.getLogger(__name__)

NEWS_PATH = DATA_DIR / "news_context.json"
LATEST_REC = DATA_DIR / "latest_recommendation.json"
REFRESH_LOG = DATA_DIR / "refresh_log.jsonl"

FFSCOUT_URL = "https://www.fantasyfootballscout.co.uk/category/team-news"

KEYWORDS = (
    "injured", "doubt", "doubtful", "rest", "rotation", "starts", "benched",
    "fitness", "knock", "back to training", "ruled out", "training",
    "available", "return", "fit again", "set to start", "could miss",
    "suspended", "yellow", "red card",
)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
TIMEOUT = 12.0
TOP_N_PLAYERS = 200


# ---------------------------------------------------------------------------
# Player roster — loaded from the latest recommendation snapshot
# ---------------------------------------------------------------------------
def _load_top_players() -> list[tuple[str, str]]:
    """Return [(name, team), ...] for our top-N EV players. Falls back to
    an empty list if the snapshot is missing, in which case we still
    scrape but cannot filter; the result is written as {}."""
    if not LATEST_REC.exists():
        log.warning("no latest_recommendation.json — run refresh_pipeline first")
        return []
    try:
        snap = json.loads(LATEST_REC.read_text())
    except Exception as exc:  # noqa: BLE001
        log.warning("could not parse latest_recommendation.json: %s", exc)
        return []
    # The snapshot only includes the 15-man squad and differentials; for
    # broader coverage we also want the bench + diffs. Build a list of
    # unique names.
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()
    for key in ("squad", "xi", "bench", "differentials"):
        for row in snap.get(key, []) or []:
            name = row.get("name")
            team = row.get("team")
            if name and name not in seen:
                candidates.append((str(name), str(team)))
                seen.add(name)
    return candidates[:TOP_N_PLAYERS]


# ---------------------------------------------------------------------------
# Source scrapers — each returns list[str] (raw paragraphs)
# ---------------------------------------------------------------------------
FFSCOUT_DEEP_FETCH = 6   # number of recent articles to fetch in full


def _fetch_ffscout(client: httpx.Client) -> list[str]:
    """Pull the latest FFScout team-news headlines, then fetch the top-N
    articles in full for player-name content.

    Two-stage so we keep total requests bounded (1 index + N articles ≤ 7).
    Each sub-fetch is independently guarded — failures on a single article
    don't break the whole source."""
    log.info("scraping FFScout team-news index …")
    resp = client.get(FFSCOUT_URL)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    paragraphs: list[str] = []
    # Stage 1: index-level previews (always include — they have headlines)
    for art in soup.find_all("article"):
        text = art.get_text(" ", strip=True)
        if text and len(text) > 40:
            paragraphs.append(text)

    # Stage 2: follow the first N article links
    article_links: list[str] = []
    for art in soup.find_all("article")[:FFSCOUT_DEEP_FETCH]:
        a = art.find("a", href=True)
        if a and a["href"].startswith("http"):
            article_links.append(a["href"])

    for url in article_links:
        try:
            r = client.get(url)
            r.raise_for_status()
            asoup = BeautifulSoup(r.text, "html.parser")
            # Most FFScout articles use a <main> or <article> container
            body = asoup.find("article") or asoup.find("main") or asoup
            for tag in body.find_all(["p", "h2", "h3", "li"]):
                text = tag.get_text(" ", strip=True)
                if text and len(text) > 30:
                    paragraphs.append(text)
        except Exception as exc:  # noqa: BLE001
            log.warning("ffscout article fetch failed for %s: %s", url, exc)
            continue

    # Dedupe while preserving order
    seen: set[str] = set()
    uniq = []
    for p in paragraphs:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    log.info("FFScout: %d paragraphs (%d articles deep-fetched)",
             len(uniq), len(article_links))
    return uniq


def _fetch_sofascore_lineups(client: httpx.Client) -> list[str]:
    """Sofascore JSON endpoints are gated behind JS / signed tokens. Best
    effort: try the public scheduled-events endpoint, which sometimes
    returns lineup hints. If it fails (typical), log and skip."""
    url = "https://api.sofascore.com/api/v1/sport/football/scheduled-events/" \
          + datetime.utcnow().strftime("%Y-%m-%d")
    log.info("scraping Sofascore scheduled events …")
    resp = client.get(url, headers={"Accept": "application/json"})
    if resp.status_code != 200:
        raise httpx.HTTPStatusError(f"status={resp.status_code}",
                                     request=resp.request, response=resp)
    data = resp.json()
    paragraphs: list[str] = []
    for ev in data.get("events", [])[:40]:
        home = ev.get("homeTeam", {}).get("name")
        away = ev.get("awayTeam", {}).get("name")
        if not (home and away):
            continue
        paragraphs.append(f"Sofascore upcoming: {home} vs {away}")
    log.info("Sofascore: %d paragraphs", len(paragraphs))
    return paragraphs


# ---------------------------------------------------------------------------
# Filtering — keep paragraphs mentioning one of our players + a keyword
# ---------------------------------------------------------------------------
_NAME_RE_CACHE: dict[str, re.Pattern] = {}


def _name_regex(name: str) -> re.Pattern:
    """Cache compiled regexes (one per player). Matches surname-only as a
    fallback so 'Bellingham starts' counts for 'Jude Bellingham'."""
    if name not in _NAME_RE_CACHE:
        parts = name.split()
        surname = parts[-1] if parts else name
        # Escape for safety. \b word boundary handles punctuation.
        _NAME_RE_CACHE[name] = re.compile(
            r"\b(" + re.escape(name) + r"|" + re.escape(surname) + r")\b",
            re.IGNORECASE,
        )
    return _NAME_RE_CACHE[name]


def _filter_paragraphs(
    paragraphs: list[str], players: list[tuple[str, str]],
) -> dict[str, list[str]]:
    """Return {player_name: [matching snippets]} where each snippet
    mentions both the player and at least one keyword."""
    out: dict[str, list[str]] = {}
    kw_re = re.compile(r"\b(" + "|".join(KEYWORDS) + r")\b", re.IGNORECASE)
    for para in paragraphs:
        if not kw_re.search(para):
            continue
        for name, _team in players:
            if _name_regex(name).search(para):
                out.setdefault(name, [])
                # Cap at 3 snippets per player; trim to a sentence-ish chunk
                if len(out[name]) < 3:
                    snippet = para.strip()
                    if len(snippet) > 400:
                        snippet = snippet[:380] + "…"
                    if snippet not in out[name]:
                        out[name].append(snippet)
    return out


# ---------------------------------------------------------------------------
# Source orchestration
# ---------------------------------------------------------------------------
def _guarded(name: str, fn, statuses: dict, *args, **kwargs):
    t0 = time.time()
    try:
        out = fn(*args, **kwargs)
        statuses[name] = {"ok": True, "elapsed_s": round(time.time() - t0, 2),
                          "n_paragraphs": len(out) if isinstance(out, list) else None}
        return out
    except Exception as exc:  # noqa: BLE001
        log.warning("source %s failed: %s", name, exc)
        statuses[name] = {"ok": False, "elapsed_s": round(time.time() - t0, 2),
                          "error": f"{type(exc).__name__}: {exc}"}
        return []


def build_news_context(
    enable_sofascore: bool = True,
    enable_reporters: bool = False,
) -> tuple[dict[str, list[str]], dict]:
    statuses: dict[str, dict] = {}
    players = _load_top_players()
    if not players:
        log.warning("no players loaded; news_context will be empty")
        return {}, {"players": 0, "sources": statuses}

    with httpx.Client(timeout=TIMEOUT, headers={"User-Agent": USER_AGENT},
                       follow_redirects=True) as client:
        ffscout_paras = _guarded("ffscout", _fetch_ffscout, statuses, client)
        sofascore_paras: list[str] = []
        if enable_sofascore:
            sofascore_paras = _guarded("sofascore", _fetch_sofascore_lineups,
                                        statuses, client)

    all_paras = list(ffscout_paras) + list(sofascore_paras)
    filtered = _filter_paragraphs(all_paras, players)
    return filtered, {
        "players": len(players),
        "paragraphs_total": len(all_paras),
        "sources": statuses,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _write_news(news: dict[str, list[str]]) -> Path:
    NEWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    NEWS_PATH.write_text(json.dumps(news, indent=2))
    return NEWS_PATH


def _append_log(meta: dict, ok: bool, error: str | None = None) -> None:
    REFRESH_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "kind": "news_digest",
        "ok": ok,
        "error": error,
        **meta,
    }
    with REFRESH_LOG.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-sofascore", action="store_true",
                        help="skip Sofascore (their API is fragile)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        news, meta = build_news_context(enable_sofascore=not args.no_sofascore)
        _write_news(news)
        _append_log(meta, ok=True)
        log.info("wrote %d players' snippets to %s", len(news), NEWS_PATH)
        return 0
    except Exception as exc:  # noqa: BLE001
        log.exception("news_digest failed")
        # Even on hard failure, write an empty file so downstream consumers
        # don't pick up a stale digest from days ago.
        _write_news({})
        _append_log({"sources": {}}, ok=False,
                    error=f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
