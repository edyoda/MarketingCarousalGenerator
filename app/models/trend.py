"""Structured models for trend discovery and analysis.

Every LLM node in this project returns one of these Pydantic models via
`llm.with_structured_output(...)` instead of free-form text. That is what makes
the graph's state predictable enough to route on.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Source(BaseModel):
    """A single web source backing a claim."""

    title: str = Field(description="Title of the page")
    url: str = Field(description="Canonical URL")
    snippet: str = Field(default="", description="Short excerpt used as evidence")


class Trend(BaseModel):
    """One candidate trend discovered during research."""

    topic: str = Field(description="Short, specific name of the trend")
    why_trending: str = Field(description="Why this is getting attention right now")
    sources: list[Source] = Field(default_factory=list)
    recency: str = Field(default="unknown", description="e.g. 'last 7 days', 'this month'")
    relevance_score: int = Field(default=0, ge=0, le=10, description="Fit for the target audience")
    content_potential_score: int = Field(default=0, ge=0, le=10, description="Carousel/visual potential")

    @property
    def combined_score(self) -> float:
        """Weighted score used for ranking. Relevance matters slightly more."""
        return 0.55 * self.relevance_score + 0.45 * self.content_potential_score


class TrendList(BaseModel):
    """Structured-output wrapper: the analyzer must return a list of trends."""

    trends: list[Trend] = Field(description="Distinct trends extracted from the search results")


class TopicSelection(BaseModel):
    """The analyzer's final pick plus the editorial angle to take."""

    selected_topic: str = Field(description="The one topic to build the carousel around")
    reason: str = Field(description="Why this topic beats the alternatives")
    content_angle: str = Field(
        description="ONE sentence (max 25 words) naming the specific angle, not a generic definition"
    )
    target_audience: str = Field(description="Who this is written for")
    hook: str = Field(description="A scroll-stopping first line")
    confidence: int = Field(default=7, ge=0, le=10)


class ResearchClaim(BaseModel):
    """A single researched statement, explicitly typed as fact/opinion/claim.

    Forcing the model to label each statement is what keeps the carousel
    honest: 'FACT' must be traceable to a source URL.
    """

    statement: str
    claim_type: Literal["FACT", "OPINION", "CLAIM"] = Field(
        description="FACT = verifiable in a source, OPINION = someone's view, CLAIM = asserted but unverified"
    )
    source_url: str = Field(default="", description="URL backing this statement; empty for OPINION")
    category: str = Field(
        default="general",
        description="definition | problem | how_it_works | example | use_case | development | implication",
    )


class TopicResearch(BaseModel):
    """Deep-dive research on the selected topic."""

    definition: str = Field(description="Plain-language definition")
    problem: str = Field(description="The problem this solves")
    how_it_works: str = Field(description="Mechanism, explained simply")
    key_facts: list[ResearchClaim] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    use_cases: list[str] = Field(default_factory=list)
    recent_developments: list[str] = Field(default_factory=list)
    implications: str = Field(default="", description="What it means for the target audience")
    sources: list[Source] = Field(default_factory=list)
