"""Stage 4 node: critique the finished deck.

The critic is a separate node with its own model call and its own prompt. It
never edits slides - it only judges them. That separation is what makes the
loop in the graph meaningful: critic -> route -> rewrite -> critic.
"""
from __future__ import annotations

import logging

from app.config import settings
from app.llm import structured
from app.models import Critique
from app.observability import traced_node

logger = logging.getLogger(__name__)

CRITIC_PROMPT = """You are a demanding social media editor reviewing a {slide_count}-slide
{platform} carousel before it is published. You are the last line of defence.

TOPIC: {topic}
ANGLE: {angle}
AUDIENCE: {audience}
INTENDED STORY: {story}

The deck:
{deck}

Research the deck was supposed to be based on:
{research}

Score it on four axes, 0-10 each:

CONTENT
  - hook_score: does slide 1 actually stop the scroll? Generic opener = 3 or less.
  - accuracy_score: is every number and claim traceable to the research? A
    statistic that is not in the research above is a serious problem - flag it.

SOCIAL MEDIA
  - clarity_score: is each slide readable in 2 seconds? Text-heavy slides,
    sentences that need re-reading, or jargon without explanation all cost points.
    Does each slide create a reason to swipe to the next?

VISUAL
  - visual_score: can each slide's message actually be drawn? Does visual_type
    fit the content? Are visual_elements concrete labels rather than vague nouns?

Then set `overall_score` holistically.

CRITICAL INSTRUCTION ON `approved`:
Do NOT approve out of politeness. Set approved=true only if you would publish
this on your own account today. If anything is generic, inaccurate, too wordy,
or visually unclear, set approved=false and list the specific slides in
`slides_to_rewrite`.

For every problem, add an entry to `issues` with the slide number, a severity,
what is wrong, and a CONCRETE fix instruction the writer can act on. "Make it
better" is not a fix. "Replace the abstract title with the 3x number from the
Anthropic report" is a fix.

Revision attempt: {revision} of {max_revisions}."""


@traced_node
def carousel_critic(state: dict) -> dict:
    """Evaluate the deck and decide whether it is publishable."""
    slides = sorted(state.get("slides", []), key=lambda s: s["slide_number"])
    topic = state.get("selected_topic", {})

    deck = "\n\n".join(
        f"SLIDE {s['slide_number']} [{s.get('visual_type', '')}]\n"
        f"  title: {s.get('title', '')}\n"
        f"  subtitle: {s.get('subtitle', '')}\n"
        f"  body: {s.get('body', '')}\n"
        f"  bullets: {s.get('bullets', [])}\n"
        f"  highlight: {s.get('highlight', '')}\n"
        f"  visual_elements: {s.get('visual_elements', [])}\n"
        f"  cta: {s.get('cta', '')}\n"
        f"  sources: {s.get('source_references', [])}"
        for s in slides
    )

    from app.graph.nodes.strategy import _research_digest

    model = structured(Critique, tier="smart")
    critique: Critique = model.invoke(
        CRITIC_PROMPT.format(
            slide_count=state["slide_count"],
            platform=state["platform"],
            topic=topic.get("selected_topic", ""),
            angle=state.get("content_angle", ""),
            audience=state["audience"],
            story=state.get("strategy", {}).get("story", ""),
            deck=deck,
            research=_research_digest(state.get("research", {}), limit=10),
            revision=state.get("revision_count", 0) + 1,
            max_revisions=settings.max_revisions,
        )
    )

    payload = critique.model_dump()

    # The graph applies its own bar rather than trusting the model's `approved`
    # flag alone - a model that scores the deck 5/10 but ticks "approved" gets
    # overruled here.
    below_threshold = critique.overall_score < settings.approval_threshold
    if critique.approved and below_threshold:
        logger.info(
            "overriding critic approval: score %d < threshold %d",
            critique.overall_score,
            settings.approval_threshold,
        )
        payload["approved"] = False
        if not payload["slides_to_rewrite"]:
            payload["slides_to_rewrite"] = _weakest_slides(payload, slides)

    # An unapproved critique with no target slides would stall the loop.
    if not payload["approved"] and not payload["slides_to_rewrite"]:
        payload["slides_to_rewrite"] = _weakest_slides(payload, slides)

    # Deduplicate and drop out-of-range slide numbers.
    valid = {s["slide_number"] for s in slides}
    payload["slides_to_rewrite"] = sorted({n for n in payload["slides_to_rewrite"] if n in valid})

    return {
        "critique": payload,
        "critique_history": [payload],
        "status": "critique_complete",
    }


def _weakest_slides(critique: dict, slides: list[dict]) -> list[int]:
    """Pick rewrite targets when the critic rejected the deck but named no slides."""
    by_severity: dict[int, int] = {}
    weight = {"high": 3, "medium": 2, "low": 1}
    for issue in critique.get("issues", []):
        number = issue.get("slide_number")
        if number:
            by_severity[number] = by_severity.get(number, 0) + weight.get(issue.get("severity", "medium"), 2)

    if by_severity:
        ranked = sorted(by_severity, key=lambda n: by_severity[n], reverse=True)
        return ranked[:2]

    # Last resort: the hook carries the most weight, so rewrite slide 1.
    return [slides[0]["slide_number"]] if slides else []


@traced_node
def prepare_rewrite(state: dict) -> dict:
    """Bump the revision counter before the rewrite fan-out.

    This node exists purely because routing functions are read-only: a
    conditional edge can decide *to* loop, but only a node can record that the
    loop happened. Without it the loop would never terminate.
    """
    revisions = state.get("revision_count", 0) + 1
    targets = state.get("critique", {}).get("slides_to_rewrite", [])
    logger.info("revision %d - rewriting slides %s", revisions, targets)
    return {
        "revision_count": revisions,
        "status": f"revising (attempt {revisions})",
    }
