"""Smoke tests for scripts.email_picks rendering.

The SMTP path isn't unit-tested (requires live Gmail credentials);
HTML / text rendering is, since it's where most field-mapping bugs land.
"""
from __future__ import annotations

import json

import pytest

from scripts.email_picks import render_email_html, render_email_text


def _stub_snapshot() -> dict:
    """Minimal snapshot that exercises every rendered section."""
    return {
        "generated_at": "2026-06-11T14:00:00+00:00",
        "matchday": 1,
        "stage": "GROUP",
        "formation": [1, 3, 4, 3],
        "point_estimate": 435.0,
        "squad": [
            {"name": "Joan García", "team": "Spain", "position": "GK",
             "price": 4.0, "ev_md1": 4.5},
            {"name": "Achraf Hakimi", "team": "Morocco", "position": "DEF",
             "price": 6.0, "ev_md1": 6.0},
            {"name": "Lamine Yamal", "team": "Spain", "position": "MID",
             "price": 10.0, "ev_md1": 8.0},
            {"name": "Harry Kane", "team": "England", "position": "FWD",
             "price": 10.5, "ev_md1": 8.4},
        ],
        "xi": [
            {"name": "Joan García", "team": "Spain", "position": "GK",
             "price": 4.0, "ev_md1": 4.5},
            {"name": "Achraf Hakimi", "team": "Morocco", "position": "DEF",
             "price": 6.0, "ev_md1": 6.0},
            {"name": "Lamine Yamal", "team": "Spain", "position": "MID",
             "price": 10.0, "ev_md1": 8.0},
            {"name": "Harry Kane", "team": "England", "position": "FWD",
             "price": 10.5, "ev_md1": 8.4},
        ],
        "bench": [
            {"name": "Bench A", "team": "X", "position": "DEF",
             "price": 4.0, "ev_md1": 3.0},
            {"name": "Bench B", "team": "Y", "position": "MID",
             "price": 4.0, "ev_md1": 2.5},
        ],
        "captain_rec": {
            "primary_captain": "Harry Kane",
            "vice_captain": "Lamine Yamal",
            "captain_reasoning": "Kane is the consensus pick on opening day.",
        },
        "differentials": [],
        "md_breakdown": [],
        "chip_plan": [
            {"matchday": 1, "chip": "12th Man", "expected_lift": 7.1,
             "reasoning": "Best alt: Dembélé"},
        ],
        "advice": {
            "starting_xi": [],
            "formation": "3-4-3",
            "bench_order": [],
            "captain": "Harry Kane",
            "vice_captain": "Lamine Yamal",
            "captain_reasoning": "He's the safest captain on opening day.",
            "key_risks": ["Kane could be subbed off at 60'",
                          "Spain's defense untested vs Italy"],
            "differentials": [],
            "chip_recommendation": None,
            "overall_strategy": "Floor-mode opener; ride consensus.",
        },
    }


def test_html_renders_with_full_snapshot():
    html = render_email_html(_stub_snapshot())
    assert "<!doctype html>" in html
    assert "MD1" in html
    assert "Harry Kane" in html
    assert "Lamine Yamal" in html
    # Captain & VC badges
    assert 'class="badge C"' in html
    assert 'class="badge VC"' in html
    # First-sub label
    assert "First sub if anyone DNPs" in html
    # Chip recommendation
    assert "12th Man" in html
    # Risks
    assert "Kane could be subbed off" in html
    # Strategy
    assert "Floor-mode opener" in html
    # FIFA application footer
    assert "play.fifa.com" in html


def test_text_fallback_renders():
    txt = render_email_text(_stub_snapshot())
    assert "Harry Kane" in txt
    assert "(C)" in txt
    assert "(VC)" in txt
    assert "[First sub]" in txt
    assert "play.fifa.com" in txt


def test_html_renders_with_no_chip_in_md():
    """When no chip is recommended for this MD, the email shows HOLD."""
    snap = _stub_snapshot()
    snap["chip_plan"] = []
    html = render_email_html(snap)
    assert "HOLD all chips" in html


def test_html_renders_without_news():
    snap = _stub_snapshot()
    html = render_email_html(snap, news_context=None)
    # No "Late news" card when no news provided
    assert "Late news" not in html


def test_html_renders_with_news():
    snap = _stub_snapshot()
    news = {"Harry Kane": ["carrying a knock, 50/50 to start"]}
    html = render_email_html(snap, news_context=news)
    assert "Late news" in html
    assert "carrying a knock" in html


def test_html_renders_when_advice_missing():
    """If the Claude advisor failed, advice is None; email must still render."""
    snap = _stub_snapshot()
    snap["advice"] = None
    html = render_email_html(snap)
    # Strategy & risks cards should be absent (no advice text)
    assert "Floor-mode opener" not in html
    # But captain reasoning falls back to the model's text
    assert "consensus pick on opening day" in html


def test_html_escapes_player_names():
    """Names with HTML-active characters should be escaped."""
    snap = _stub_snapshot()
    snap["xi"][0]["name"] = "<script>alert(1)</script>"
    snap["captain_rec"]["primary_captain"] = "<script>"
    html = render_email_html(snap)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
