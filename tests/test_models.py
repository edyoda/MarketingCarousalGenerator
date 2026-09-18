"""Tests for the structured-output models."""
from __future__ import annotations

from app.models import Critique, Slide, Trend


def test_trend_score_weights_relevance_slightly_higher():
    relevant = Trend(topic="a", why_trending="x", relevance_score=10, content_potential_score=0)
    visual = Trend(topic="b", why_trending="x", relevance_score=0, content_potential_score=10)
    assert relevant.combined_score > visual.combined_score


def test_text_budget_catches_blog_post_slides():
    lean = Slide(slide_number=1, title="Short and punchy", bullets=["a", "b"])
    assert not lean.text_budget_exceeded()

    wordy = Slide(
        slide_number=1,
        title="A title " * 12,
        body="A long body. " * 30,
        bullets=["A bullet that keeps going and going" * 3],
    )
    assert wordy.text_budget_exceeded()


def test_critique_separates_score_from_approval():
    """The graph applies its own threshold, so both fields must exist."""
    critique = Critique(overall_score=5, approved=True)
    assert critique.overall_score == 5 and critique.approved is True
