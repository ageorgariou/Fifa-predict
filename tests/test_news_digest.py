"""Smoke tests for scripts.news_digest filter logic.

The HTTP-fetching paths aren't unit-tested (they hit live websites);
filter / parse logic is, since it's the part that can silently fail.
"""
from __future__ import annotations

from scripts.news_digest import _filter_paragraphs


def test_filter_matches_player_and_keyword():
    paragraphs = [
        "Bellingham injured in training, doubtful for Sunday.",
        "Random text with no signal.",
        "Mbappé starts on the bench against Tunisia.",
    ]
    players = [("Jude Bellingham", "England"), ("Kylian Mbappé", "France")]
    out = _filter_paragraphs(paragraphs, players)
    assert "Jude Bellingham" in out
    assert "Kylian Mbappé" in out
    assert len(out["Jude Bellingham"]) == 1


def test_filter_requires_keyword():
    paragraphs = ["Bellingham scored a hat-trick yesterday."]   # no keyword
    players = [("Jude Bellingham", "England")]
    out = _filter_paragraphs(paragraphs, players)
    assert out == {}


def test_filter_uses_surname_match():
    paragraphs = ["Yamal ruled out with a calf knock."]
    players = [("Lamine Yamal", "Spain")]
    out = _filter_paragraphs(paragraphs, players)
    assert "Lamine Yamal" in out


def test_filter_caps_at_three_snippets_per_player():
    paragraphs = [f"Bellingham doubtful #{i}" for i in range(10)]
    players = [("Jude Bellingham", "England")]
    out = _filter_paragraphs(paragraphs, players)
    assert len(out["Jude Bellingham"]) == 3


def test_filter_ignores_unknown_players():
    """A paragraph about nobody in our list should not be tagged."""
    paragraphs = ["Cristiano Ronaldo injured ahead of MD1."]
    players = [("Lamine Yamal", "Spain"), ("Kylian Mbappé", "France")]
    out = _filter_paragraphs(paragraphs, players)
    assert out == {}
