"""Stage 5 nodes: turn approved slide data into real PNG files.

These nodes make NO LLM calls. They are the boundary between "content" and
"pixels" - the graph decided what to say, the renderer decides how it looks.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path

from langgraph.types import interrupt

from app.config import settings
from app.models import ImageCheck
from app.observability import traced_node
from app.rendering.renderer import render_carousel

logger = logging.getLogger(__name__)


def _slug(text: str, limit: int = 48) -> str:
    """Filesystem-safe folder name derived from the topic."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", (text or "carousel").lower()).strip("-")
    return (cleaned[:limit] or "carousel").strip("-")


def run_output_dir(state: dict) -> Path:
    """Where this run's files live: output/<timestamp>-<topic-slug>/"""
    topic = state.get("selected_topic", {}).get("selected_topic", "carousel")
    run_id = state.get("run_id") or datetime.now().strftime("%Y%m%d-%H%M%S")
    return settings.output_dir / f"{run_id}-{_slug(topic)}"


# ---------------------------------------------------------------------------
# Optional second human gate
# ---------------------------------------------------------------------------
@traced_node
def human_image_approval(state: dict) -> dict:
    """Pause before rendering, so a human can veto the final copy.

    LANGGRAPH CONCEPT: A SECOND INTERRUPT
    Same mechanism as the topic gate, at a different point in the graph. It
    demonstrates that interrupts are just nodes - you can put as many as you
    want wherever a decision is worth a human's attention.
    """
    if not settings.require_image_approval:
        return {"human_decision": "auto_approved"}

    slides = sorted(state.get("slides", []), key=lambda s: s["slide_number"])
    decision = interrupt(
        {
            "type": "image_approval",
            "question": "Render these slides to PNG?",
            "topic": state.get("selected_topic", {}).get("selected_topic", ""),
            "score": state.get("critique", {}).get("overall_score"),
            "slides": [
                {
                    "slide_number": s["slide_number"],
                    "title": s.get("title", ""),
                    "visual_type": s.get("visual_type", ""),
                }
                for s in slides
            ],
            "options": ["approve", "reject_images"],
        }
    )

    if isinstance(decision, str):
        decision = {"action": decision}
    action = (decision or {}).get("action", "approve")
    return {
        "human_decision": action,
        "status": "render_approved" if action == "approve" else "render_declined",
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
@traced_node
def render_images(state: dict) -> dict:
    """Render every slide to a 1080x1350 PNG."""
    slides = state.get("slides", [])
    if not slides:
        return {
            "image_paths": [],
            "status": "render_failed",
            "errors": ["No slides available to render."],
        }

    topic = state.get("selected_topic", {}).get("selected_topic", "")
    out_dir = run_output_dir(state)
    attempts = state.get("render_attempts", 0) + 1

    try:
        paths, theme, backend = render_carousel(
            slides=slides,
            topic=topic,
            out_dir=out_dir,
            theme_name=state.get("design_theme") or None,
            handle=state.get("platform", ""),
            style=state.get("slide_style") or None,
        )
    except Exception as exc:
        logger.exception("rendering failed")
        return {
            "image_paths": [],
            "render_attempts": attempts,
            "status": "render_failed",
            "errors": [f"Rendering failed: {exc}"],
        }

    logger.info("rendered %d images with %s into %s", len(paths), backend, out_dir)
    return {
        "image_paths": [str(p) for p in paths],
        "design_theme": theme,
        "render_attempts": attempts,
        "status": "images_rendered",
    }


@traced_node
def image_quality_check(state: dict) -> dict:
    """A deterministic check that real files of the right size exist on disk.

    This is intentionally NOT an LLM call. "Did five PNGs get written at
    1080x1350?" is a question with a correct answer, and code answers it more
    reliably and far more cheaply than a model.
    """
    expected = state["slide_count"]
    paths = [Path(p) for p in state.get("image_paths", [])]
    problems: list[str] = []

    if len(paths) != expected:
        problems.append(f"Expected {expected} images, found {len(paths)}")

    for path in paths:
        if not path.exists():
            problems.append(f"{path.name} was not written")
            continue
        if path.stat().st_size < 5_000:
            problems.append(f"{path.name} is suspiciously small ({path.stat().st_size} bytes)")
        try:
            from PIL import Image

            with Image.open(path) as image:
                if image.size != (settings.canvas_width, settings.canvas_height):
                    problems.append(f"{path.name} is {image.size}, expected "
                                    f"({settings.canvas_width}, {settings.canvas_height})")
        except Exception as exc:
            problems.append(f"{path.name} could not be opened: {exc}")

    check = ImageCheck(
        all_rendered=not problems,
        expected=expected,
        actual=len(paths),
        problems=problems,
    )

    if problems:
        logger.warning("image check found problems: %s", problems)
    else:
        logger.info("image check passed: %d/%d images valid", len(paths), expected)

    return {"image_check": check.model_dump(), "status": "images_checked"}


# ---------------------------------------------------------------------------
# Final output
# ---------------------------------------------------------------------------
@traced_node
def finalize(state: dict) -> dict:
    """Write carousel.json next to the images and mark the run complete."""
    out_dir = run_output_dir(state)
    out_dir.mkdir(parents=True, exist_ok=True)

    topic = state.get("selected_topic", {})
    critique = state.get("critique", {})
    slides = sorted(state.get("slides", []), key=lambda s: s["slide_number"])

    # Sources are collected per claim and surfaced here so anything factual in
    # the deck can be traced back to where it came from.
    sources: list[dict] = []
    seen: set[str] = set()
    for source in state.get("research_sources", []) + topic.get("sources", []):
        url = source.get("url", "")
        if url and url not in seen:
            seen.add(url)
            sources.append({"title": source.get("title", ""), "url": url})

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input": {
            "domain": state.get("domain"),
            "audience": state.get("audience"),
            "platform": state.get("platform"),
            "slide_count": state.get("slide_count"),
        },
        "topic": {
            "selected_topic": topic.get("selected_topic", ""),
            "angle": state.get("content_angle", ""),
            "hook": topic.get("hook", ""),
            "reason": topic.get("reason", ""),
            "why_trending": topic.get("why_trending", ""),
        },
        "strategy": state.get("strategy", {}),
        "slides": slides,
        "quality": {
            "overall_score": critique.get("overall_score"),
            "approved": critique.get("approved"),
            "revisions": state.get("revision_count", 0),
            "flag": state.get("quality_flag", ""),
            "summary": critique.get("summary", ""),
            "outstanding_issues": critique.get("issues", []) if not critique.get("approved") else [],
        },
        "images": state.get("image_paths", []),
        "image_check": state.get("image_check", {}),
        "design_theme": state.get("design_theme", ""),
        "slide_style": state.get("slide_style", "") or settings.slide_style,
        "sources": sources,
        "errors": state.get("errors", []),
    }

    manifest_path = out_dir / "carousel.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote %s", manifest_path)

    return {"status": "complete"}


@traced_node
def flag_quality(state: dict) -> dict:
    """Record that the deck shipped without passing the critic.

    Requirement: after max revisions, continue with the best available version
    and FLAG it rather than pretending it passed.
    """
    critique = state.get("critique", {})
    logger.warning(
        "shipping unapproved carousel after %d revisions (score %s)",
        state.get("revision_count", 0),
        critique.get("overall_score"),
    )
    return {
        "quality_flag": "max_revisions_reached",
        "status": "quality_flagged",
        "errors": [
            f"Published without critic approval after {state.get('revision_count', 0)} revisions "
            f"(score {critique.get('overall_score')}/10)."
        ],
    }
