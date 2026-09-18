"""Structured models for the carousel narrative, slides and design."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# The renderer has a template for each of these. The LLM must pick one of them,
# which is why it is a Literal and not a free string.
VisualType = Literal[
    "HOOK",
    "STATISTIC",
    "COMPARISON",
    "PROCESS",
    "ARCHITECTURE",
    "FLOW",
    "TIMELINE",
    "CARD_GRID",
    "BEFORE_AFTER",
    "DIAGRAM",
    "QUOTE",
    "CTA",
]


class SlideStrategy(BaseModel):
    """What one slide is supposed to accomplish, before it is written."""

    slide: int = Field(description="1-based slide number")
    purpose: str = Field(description="e.g. Hook, Problem, Explanation, Examples, Takeaway")
    key_message: str = Field(description="The single idea this slide must land")


class ContentStrategy(BaseModel):
    """The narrative arc across the whole carousel."""

    story: str = Field(description="One paragraph describing the arc from slide 1 to the last slide")
    slide_strategy: list[SlideStrategy] = Field(description="One entry per slide, in order")


class SlidePlan(BaseModel):
    """The planner's brief for a single slide (structure, not final copy)."""

    slide_number: int
    purpose: str
    key_message: str
    visual_type: VisualType = "DIAGRAM"
    key_visual: str = Field(
        default="",
        description="The visual idea, e.g. 'AI Agent -> MCP -> Tools'",
    )


class CarouselPlan(BaseModel):
    """Structured-output wrapper for the full plan."""

    slides: list[SlidePlan]


class Slide(BaseModel):
    """A finished slide: copy plus the visual spec the renderer needs."""

    slide_number: int
    title: str = Field(description="Headline. Short - it is rendered very large.")
    subtitle: str = Field(default="", description="One supporting line, optional")
    body: str = Field(default="", description="1-2 short sentences maximum")
    bullets: list[str] = Field(
        default_factory=list,
        description="Up to 4 very short fragments; preferred over long body text",
    )
    visual_type: VisualType = "DIAGRAM"
    key_visual: str = Field(default="", description="Visual concept in words")
    visual_elements: list[str] = Field(
        default_factory=list,
        description="Labels used by the renderer: boxes in a flow, cards in a grid, rows in a timeline",
    )
    highlight: str = Field(default="", description="A single word/number to emphasise, e.g. '10x' or 'MCP'")
    cta: str = Field(default="", description="Only meaningful on the last slide")
    source_references: list[str] = Field(default_factory=list, description="URLs backing any factual claim")

    def text_budget_exceeded(self) -> bool:
        """Cheap deterministic guard against slides that drift into blog posts."""
        total = len(self.title) + len(self.subtitle) + len(self.body) + sum(len(b) for b in self.bullets)
        return total > 420
