"""Routing logic: the conditional edges and fan-outs that give the graph shape.

LANGGRAPH CONCEPT: CONDITIONAL EDGES
A routing function receives the state and returns either
  * the name of the next node (a string),
  * a list of node names, or
  * a list of `Send(...)` objects, which dispatches parallel work.

Routers are pure read-only functions - they never modify state. Anything that
needs to be written (like bumping the revision counter) has to happen in a node,
which is why `prepare_rewrite` exists.
"""
from __future__ import annotations

import logging

from langgraph.types import Send

from app.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Human-in-the-loop routing
# ---------------------------------------------------------------------------
def route_after_topic_approval(state: dict) -> str:
    """approve/choose_other -> research; reject -> pick a different topic.

    Rejecting sends the graph BACK to topic_selection, which is a second loop in
    the graph (the critic loop is the first). The rejected topic is remembered in
    state so the selector does not offer it again.
    """
    decision = state.get("human_decision", "approve")
    if decision == "reject":
        logger.info("human rejected the topic - re-selecting")
        return "topic_selection"
    return "research_topic"


def route_after_image_approval(state: dict) -> str:
    """Last gate before we spend time rendering PNGs."""
    if state.get("human_decision") == "reject_images":
        return "finalize"
    return "render_images"


# ---------------------------------------------------------------------------
# Parallel slide generation (fan-out)
# ---------------------------------------------------------------------------
def fan_out_slides(state: dict) -> list[Send]:
    """Dispatch one `generate_slide` worker per planned slide, all at once.

    Each Send carries only what that worker needs (SlideWorkerState), not the
    whole CarouselState. Their `{"slides": [...]}` returns are merged by the
    `merge_slides` reducer in state.py.
    """
    plan = state.get("carousel_plan", [])
    if not plan:
        logger.error("no carousel plan to fan out from")
        return [Send("merge_slides", state)]

    shared = _shared_context(state)
    return [Send("generate_slide", {"plan": slide_plan, **shared}) for slide_plan in plan]


def fan_out_rewrites(state: dict) -> list[Send]:
    """Dispatch one `rewrite_slide` worker per slide the critic rejected.

    Same fan-out mechanism as above, but targeted: only the weak slides are
    regenerated, and the reducer upserts them over the originals.
    """
    critique = state.get("critique", {})
    targets = critique.get("slides_to_rewrite", [])
    plan_by_number = {p["slide_number"]: p for p in state.get("carousel_plan", [])}
    slides_by_number = {s["slide_number"]: s for s in state.get("slides", [])}

    if not targets:
        return [Send("carousel_critic", state)]

    shared = _shared_context(state)
    sends = []
    for number in targets:
        plan = plan_by_number.get(number)
        if not plan:
            continue
        issues = [i for i in critique.get("issues", []) if i.get("slide_number") == number]
        sends.append(
            Send(
                "rewrite_slide",
                {
                    "plan": plan,
                    "previous_slide": slides_by_number.get(number, {}),
                    "issues": issues,
                    **shared,
                },
            )
        )

    logger.info("rewriting slides %s", targets)
    return sends or [Send("carousel_critic", state)]


def _shared_context(state: dict) -> dict:
    """The slice of state every slide worker needs."""
    return {
        "topic": {**state.get("selected_topic", {}), "content_angle": state.get("content_angle", "")},
        "research": state.get("research", {}),
        "strategy": state.get("strategy", {}),
        "audience": state["audience"],
        "platform": state["platform"],
        "slide_count": state["slide_count"],
    }


# ---------------------------------------------------------------------------
# The quality loop
# ---------------------------------------------------------------------------
def quality_gate(state: dict) -> str:
    """The conditional edge that creates the rewrite loop.

        carousel_critic --> quality_gate --+-- approved -------> image approval
                                           |
                                           +-- rejected ------> prepare_rewrite
                                                                    |
                                                                (rewrite slides)
                                                                    |
                                                                    v
                                                             back to carousel_critic

    The loop is bounded: after `max_revisions` failed attempts the graph gives up
    and continues with the best version it has, flagged rather than silently
    shipped.
    """
    critique = state.get("critique", {})
    revisions = state.get("revision_count", 0)

    if critique.get("approved"):
        logger.info("critique approved at score %s", critique.get("overall_score"))
        return "approved"

    if revisions >= settings.max_revisions:
        logger.warning(
            "max revisions (%d) reached with score %s - continuing with best available version",
            settings.max_revisions,
            critique.get("overall_score"),
        )
        return "give_up"

    return "rewrite"


def route_after_image_check(state: dict) -> str:
    """Retry rendering once if any image failed to write, then finish either way."""
    check = state.get("image_check", {})
    attempts = state.get("render_attempts", 0)

    if not check.get("all_rendered", True) and attempts < 2:
        logger.warning("re-rendering: %s", check.get("problems"))
        return "render_images"
    return "finalize"
