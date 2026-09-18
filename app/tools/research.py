"""Query-building helpers for trend discovery and deep research.

Kept separate from `web_search` so the *what to ask* logic is testable without
touching the network.
"""
from __future__ import annotations

from datetime import date


def _period() -> str:
    """Current month/year string, e.g. 'September 2026'.

    Used to bias queries towards the present. This is why the app finds a topic
    that is trending *now* rather than whatever was popular in training data.
    """
    return date.today().strftime("%B %Y")


def trend_queries(domain: str, audience: str) -> list[str]:
    """Build a spread of queries so we never depend on one search phrasing."""
    period = _period()
    return [
        f"{domain} trends {period}",
        f"{domain} latest developments news",
        f"what's new in {domain} this week",
        f"{domain} breakthrough announcement {period}",
        f"{domain} news for {audience}",
        f"emerging {domain} tools {period}",
    ]


def topic_research_queries(topic: str, audience: str) -> list[str]:
    """Targeted follow-ups covering each section of the research model."""
    return [
        f"{topic} explained",
        f"what is {topic} and why does it matter",
        f"{topic} how it works architecture",
        f"{topic} real world examples use cases",
        f"{topic} latest developments {_period()}",
        f"{topic} for {audience} implications",
    ]
