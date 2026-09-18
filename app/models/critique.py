"""Structured models for the critic node and the image quality gate."""
from __future__ import annotations

from pydantic import BaseModel, Field


class SlideIssue(BaseModel):
    """A specific, actionable problem with one slide."""

    slide_number: int
    severity: str = Field(default="medium", description="low | medium | high")
    issue: str = Field(description="What is wrong")
    fix: str = Field(description="Concrete instruction for the rewrite")


class Critique(BaseModel):
    """The critic's verdict on the whole carousel.

    `approved` is deliberately separate from `overall_score` so the graph can
    apply its own threshold rather than trusting a model that wants to be nice.
    """

    overall_score: int = Field(ge=0, le=10, description="Holistic quality, 0-10")
    hook_score: int = Field(default=0, ge=0, le=10)
    clarity_score: int = Field(default=0, ge=0, le=10)
    visual_score: int = Field(default=0, ge=0, le=10)
    accuracy_score: int = Field(default=0, ge=0, le=10)
    approved: bool = Field(description="True only if this is genuinely publish-ready")
    issues: list[SlideIssue] = Field(default_factory=list)
    slides_to_rewrite: list[int] = Field(
        default_factory=list, description="Slide numbers that must be rewritten"
    )
    summary: str = Field(default="", description="One-paragraph verdict")


class ImageCheck(BaseModel):
    """Deterministic post-render check - not an LLM call."""

    all_rendered: bool
    expected: int
    actual: int
    problems: list[str] = Field(default_factory=list)
