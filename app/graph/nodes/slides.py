"""Stage 3 nodes: write the slides - in parallel - and rewrite weak ones.

LANGGRAPH CONCEPT: PARALLEL EXECUTION (fan-out / fan-in)
`generate_slide` is written as if only one slide exists. The graph dispatches N
copies of it at once using `Send` (see edges.py), each with its own slice of
work. Their results converge back into `state["slides"]` through the
`merge_slides` reducer defined in state.py.

    carousel_plan
         |
    +----+----+----+----+
    v    v    v    v    v
   s1   s2   s3   s4   s5        <- all five run concurrently
    +----+----+----+----+
         v
    merge_slides                  <- fan-in barrier
"""
from __future__ import annotations

import logging

from app.llm import structured
from app.models import Slide
from app.observability import traced_node

logger = logging.getLogger(__name__)

SLIDE_PROMPT = """You are writing ONE slide of a {slide_count}-slide {platform} carousel.

TOPIC: {topic}
ANGLE: {angle}
AUDIENCE: {audience}
OVERALL STORY: {story}

YOU ARE WRITING SLIDE {slide_number} OF {slide_count}.
  Purpose:      {purpose}
  Key message:  {key_message}
  Visual type:  {visual_type}
  Visual idea:  {key_visual}

Research you may draw on:
{research}

THIS IS A CAROUSEL SLIDE, NOT A BLOG POST. It gets ~2 seconds of attention and
is rendered as a 1080x1350 image, so text length is a hard design constraint:

  title        <= 60 characters. This is set in very large type.
  subtitle     <= 80 characters, optional.
  body         <= 160 characters total. One or two short sentences. Often empty
                 when bullets carry the message.
  bullets      up to 4 items, each <= 60 characters. Fragments, not sentences.
  highlight    a single word or number to emphasise (e.g. "10x", "MCP"), optional.

`visual_elements` are the LABELS PRINTED ON THE IMAGE, word for word. They are
not a description of a picture. The renderer draws each string inside a box - it
cannot draw arrows, bars or lanes you describe in prose.

  WRONG: ["Steep arrow between the two bars", "Flat line labelled review capacity"]
  RIGHT: ["2025: 12%", "2026: 42%"]

  WRONG: ["Left lane: MACHINE - style, formatting, obvious defects"]
  RIGHT: ["Machine: formatting", "Human: correctness"]

Keep every item under 30 characters. Shape by visual_type:
  FLOW / PROCESS / ARCHITECTURE -> node labels in order, e.g.
        ["AI Agent", "MCP Server", "Your Tools"]   (2-4 items, <= 22 chars each)
  COMPARISON / BEFORE_AFTER     -> exactly 2 items: ["Before: ...", "After: ..."]
  CARD_GRID                     -> 3-4 short card labels
  TIMELINE                      -> 3-4 entries like "2024 - spec published"
  STATISTIC                     -> ["<the number>", "<what it measures>"] and put
        the number itself in `highlight` too
  HOOK / QUOTE / CTA / DIAGRAM  -> 0-3 supporting labels, may be empty

Accuracy rules:
- Only state a number if it appears in the research above. Never estimate.
- If you use a FACT from the research, put its source URL in `source_references`.
- Do not write "In today's fast-paced world", "Let's dive in", or "game-changer".

{slide_role_hint}

Write slide {slide_number} now."""

REWRITE_PROMPT = """You are REWRITING slide {slide_number} of a {slide_count}-slide
{platform} carousel because a reviewer rejected it.

TOPIC: {topic}
ANGLE: {angle}
OVERALL STORY: {story}

The slide as it stands:
{previous}

What the reviewer said is wrong with it:
{issues}

Research you may draw on:
{research}

Fix exactly those problems. Keep what already worked - do not rewrite the slide
from scratch if only the title is weak. Respect the same hard limits:
  title <= 60 chars, subtitle <= 80, body <= 160, bullets <= 4 items of <= 60 chars.

Slide purpose: {purpose}
Key message:  {key_message}
Visual type:  {visual_type}

Return the improved slide."""


def _role_hint(slide_number: int, slide_count: int) -> str:
    """Position-specific guidance - the first and last slides do different jobs."""
    if slide_number == 1:
        return (
            "This is the FIRST slide. Its only job is to stop the scroll. Lead with "
            "tension, a surprising fact or a pointed question. Do not summarise the "
            "whole carousel here."
        )
    if slide_number == slide_count:
        return (
            "This is the LAST slide. Land the takeaway and give one clear call to "
            "action in `cta` (e.g. 'Follow for more' or a concrete next step). Keep "
            "it short."
        )
    return (
        "This is a MIDDLE slide. End on something that makes the reader want the "
        "next slide - an open loop, a consequence, or a 'but'."
    )


def _digest(worker_state: dict) -> str:
    from app.graph.nodes.strategy import _research_digest

    return _research_digest(worker_state.get("research", {}), limit=10)


@traced_node
def generate_slide(state: dict) -> dict:
    """Write ONE slide. Many copies of this run concurrently.

    Note the return shape: `{"slides": [one_slide]}`. Each parallel worker
    returns a one-element list, and the reducer merges them into the full deck.
    """
    plan = state["plan"]
    topic = state.get("topic", {})
    slide_number = plan["slide_number"]
    slide_count = state["slide_count"]

    model = structured(Slide, tier="fast")
    slide: Slide = model.invoke(
        SLIDE_PROMPT.format(
            slide_count=slide_count,
            platform=state["platform"],
            topic=topic.get("selected_topic", ""),
            angle=topic.get("content_angle", ""),
            audience=state["audience"],
            story=state.get("strategy", {}).get("story", ""),
            slide_number=slide_number,
            purpose=plan.get("purpose", ""),
            key_message=plan.get("key_message", ""),
            visual_type=plan.get("visual_type", "DIAGRAM"),
            key_visual=plan.get("key_visual", ""),
            research=_digest(state),
            slide_role_hint=_role_hint(slide_number, slide_count),
        )
    )

    # The model chooses copy; the plan owns structure. Pin both so a creative
    # answer cannot renumber the deck or invent an untemplated visual type.
    slide.slide_number = slide_number
    slide.visual_type = plan.get("visual_type", slide.visual_type)

    return {"slides": [slide.model_dump()]}


@traced_node
def rewrite_slide(state: dict) -> dict:
    """Rewrite ONE rejected slide. Also runs in parallel, one per bad slide."""
    plan = state["plan"]
    previous = state.get("previous_slide", {})
    issues = state.get("issues", [])
    slide_number = plan["slide_number"]

    issue_block = "\n".join(
        f"  - [{i.get('severity', 'medium')}] {i.get('issue', '')}\n    Fix: {i.get('fix', '')}"
        for i in issues
    ) or "  - The reviewer flagged this slide as the weakest in the deck."

    previous_block = (
        f"  title: {previous.get('title', '')}\n"
        f"  subtitle: {previous.get('subtitle', '')}\n"
        f"  body: {previous.get('body', '')}\n"
        f"  bullets: {previous.get('bullets', [])}\n"
        f"  visual_elements: {previous.get('visual_elements', [])}"
    )

    model = structured(Slide, tier="smart")  # rewrites get the stronger model
    slide: Slide = model.invoke(
        REWRITE_PROMPT.format(
            slide_number=slide_number,
            slide_count=state["slide_count"],
            platform=state["platform"],
            topic=state.get("topic", {}).get("selected_topic", ""),
            angle=state.get("topic", {}).get("content_angle", ""),
            story=state.get("strategy", {}).get("story", ""),
            previous=previous_block,
            issues=issue_block,
            research=_digest(state),
            purpose=plan.get("purpose", ""),
            key_message=plan.get("key_message", ""),
            visual_type=plan.get("visual_type", "DIAGRAM"),
        )
    )

    slide.slide_number = slide_number
    slide.visual_type = plan.get("visual_type", slide.visual_type)

    # Keep sources from the original if the rewrite dropped them.
    if not slide.source_references:
        slide.source_references = previous.get("source_references", [])

    return {"slides": [slide.model_dump()]}


@traced_node
def merge_slides(state: dict) -> dict:
    """Fan-in barrier after the parallel writers.

    The reducer has already merged and sorted the slides; this node exists so
    the graph has a single, named join point that later edges can hang off, and
    so we can validate the deck once rather than inside every worker.
    """
    slides = state.get("slides", [])
    expected = state["slide_count"]

    problems: list[str] = []
    numbers = {s["slide_number"] for s in slides}
    missing = [n for n in range(1, expected + 1) if n not in numbers]
    if missing:
        problems.append(f"Missing slides: {missing}")

    over_budget = [s["slide_number"] for s in slides if Slide(**s).text_budget_exceeded()]
    if over_budget:
        problems.append(f"Slides over the text budget: {over_budget}")
        logger.warning("text budget exceeded on slides %s", over_budget)

    return {
        "status": "slides_generated",
        "errors": problems,
    }
