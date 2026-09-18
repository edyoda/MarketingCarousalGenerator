"""Stage 2 nodes: decide the narrative, then plan the slides.

Two nodes rather than one, because they answer different questions:
  content_strategy  -> what story are we telling, beat by beat?
  carousel_planner  -> what does each slide contain and how is it drawn?

Splitting them keeps each prompt focused, and gives the graph a natural place to
inspect the arc before any slide copy is written.
"""
from __future__ import annotations

import logging

from app.llm import structured
from app.models import CarouselPlan, ContentStrategy
from app.observability import traced_node

logger = logging.getLogger(__name__)

STRATEGY_PROMPT = """You are planning the narrative for a {slide_count}-slide {platform} carousel.

TOPIC: {topic}
ANGLE: {angle}
AUDIENCE: {audience}
HOOK: {hook}

Research available:
{research}

Design the story arc. A common arc is:
  Hook -> Problem -> Explanation -> Real examples -> Takeaway/CTA

but you should CHOOSE the arc that fits this topic. A comparison topic might go
Hook -> Old way -> New way -> Evidence -> Verdict. A risk topic might go
Hook -> What broke -> Why -> What to do -> Checklist. Pick deliberately.

Return:
- `story`: one paragraph describing the arc and why it fits this topic
- `slide_strategy`: exactly {slide_count} entries, each with the slide number,
  its purpose, and the single key message it must land.

Each slide must earn the swipe to the next one."""

PLANNER_PROMPT = """You are turning a narrative into a concrete {slide_count}-slide plan
for a {platform} carousel.

TOPIC: {topic}
ANGLE: {angle}
STORY: {story}

Slide strategy:
{strategy}

Research available:
{research}

For each slide choose a `visual_type` from EXACTLY this list:
  HOOK          - a big statement, slide 1 material
  STATISTIC     - one number dominates the slide
  COMPARISON    - two things side by side
  PROCESS       - numbered steps
  ARCHITECTURE  - components and how they connect
  FLOW          - A -> B -> C pipeline
  TIMELINE      - events in order
  CARD_GRID     - 3-4 parallel items
  BEFORE_AFTER  - the old way vs the new way
  DIAGRAM       - a general labelled concept drawing
  QUOTE         - a single pulled quote
  CTA           - the closing ask, last slide material

Rules:
- Slide 1 is normally HOOK. The last slide is normally CTA.
- Do NOT use the same visual_type for every middle slide - variety is what makes
  a carousel feel designed rather than templated.
- Only use STATISTIC if the research contains a real number.
- `key_visual` names the labels that will appear on the image, e.g.
  "AI Agent -> MCP -> Tools" or "Before: 5 integrations | After: 1 protocol".
  It must be label text, not a description of a picture: "a steep arrow between
  two bars" cannot be rendered, "2025: 12% / 2026: 42%" can.

Keep `key_message` to one short sentence."""


@traced_node
def content_strategy(state: dict) -> dict:
    """Decide the narrative arc before any copy exists."""
    topic = state.get("selected_topic", {})
    research = state.get("research", {})

    model = structured(ContentStrategy, tier="smart")
    strategy: ContentStrategy = model.invoke(
        STRATEGY_PROMPT.format(
            slide_count=state["slide_count"],
            platform=state["platform"],
            topic=topic.get("selected_topic", ""),
            angle=state.get("content_angle", ""),
            audience=state["audience"],
            hook=topic.get("hook", ""),
            research=_research_digest(research),
        )
    )

    return {"strategy": strategy.model_dump(), "status": "strategy_ready"}


@traced_node
def carousel_planner(state: dict) -> dict:
    """Produce the per-slide brief that the parallel writers will each expand."""
    topic = state.get("selected_topic", {})
    strategy = state.get("strategy", {})
    slide_count = state["slide_count"]

    strategy_block = "\n".join(
        f"  Slide {s['slide']}: {s['purpose']} - {s['key_message']}"
        for s in strategy.get("slide_strategy", [])
    )

    model = structured(CarouselPlan, tier="smart")
    plan: CarouselPlan = model.invoke(
        PLANNER_PROMPT.format(
            slide_count=slide_count,
            platform=state["platform"],
            topic=topic.get("selected_topic", ""),
            angle=state.get("content_angle", ""),
            story=strategy.get("story", ""),
            strategy=strategy_block,
            research=_research_digest(state.get("research", {})),
        )
    )

    slides = sorted(plan.slides, key=lambda s: s.slide_number)[:slide_count]

    # Repair pass: the graph downstream assumes slides 1..N exist, so we never
    # let a model's off-by-one break the fan-out.
    for index, slide in enumerate(slides, start=1):
        slide.slide_number = index
    while len(slides) < slide_count:
        from app.models import SlidePlan

        missing = len(slides) + 1
        logger.warning("planner returned too few slides; synthesising slide %d", missing)
        slides.append(
            SlidePlan(
                slide_number=missing,
                purpose="Takeaway",
                key_message=state.get("content_angle", "Key takeaway"),
                visual_type="CTA" if missing == slide_count else "DIAGRAM",
                key_visual="",
            )
        )

    return {
        "carousel_plan": [s.model_dump() for s in slides],
        "status": "carousel_planned",
    }


def _research_digest(research: dict, limit: int = 12) -> str:
    """Compact the research into something promptable.

    The full research object is large; every downstream prompt gets this digest
    instead so we are not re-sending the same 4000 tokens to five parallel
    workers.
    """
    if not research:
        return "(no research available)"

    lines = [
        f"Definition: {research.get('definition', '')}",
        f"Problem: {research.get('problem', '')}",
        f"How it works: {research.get('how_it_works', '')}",
        f"Implications: {research.get('implications', '')}",
    ]

    facts = research.get("key_facts", [])[:limit]
    if facts:
        lines.append("Key facts (with type and source):")
        for fact in facts:
            src = fact.get("source_url", "")
            lines.append(
                f"  [{fact.get('claim_type', 'CLAIM')}] {fact.get('statement', '')}"
                + (f"  (source: {src})" if src else "")
            )

    if research.get("examples"):
        lines.append("Examples: " + "; ".join(research["examples"][:6]))
    if research.get("use_cases"):
        lines.append("Use cases: " + "; ".join(research["use_cases"][:6]))
    if research.get("recent_developments"):
        lines.append("Recent: " + "; ".join(research["recent_developments"][:6]))

    return "\n".join(lines)
