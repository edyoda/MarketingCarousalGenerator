"""Command-line entry point.

    python -m app.main --domain "AI & Technology" --audience "Software Developers"

Also demonstrates how to drive a LangGraph app that can interrupt: run until it
pauses, ask the human, then resume with `Command(resume=...)` on the SAME
thread_id.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from langgraph.types import Command

from app.config import settings
from app.graph.graph import get_app, save_diagram
from app.graph.state import initial_state
from app.llm import describe
from app.observability import NODE_LABELS, configure_logging, run_config, tracing_status
from app.tools.web_search import active_backend

logger = logging.getLogger(__name__)

RECURSION_LIMIT = 60


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="carousel",
        description="Discover a trending topic and turn it into a publish-ready carousel.",
    )
    parser.add_argument("--domain", default="AI & Technology", help="e.g. 'AI & Technology'")
    parser.add_argument("--audience", default="Software Developers")
    parser.add_argument("--platform", default="LinkedIn")
    parser.add_argument("--slides", type=int, default=5)
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Session id for persistence. Reuse it with --resume to continue a run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the run stored under --thread-id instead of starting fresh.",
    )
    parser.add_argument("--approve-topic", action="store_true", help="Pause for topic approval.")
    parser.add_argument("--approve-images", action="store_true", help="Pause before rendering.")
    parser.add_argument(
        "--style",
        default=None,
        choices=["modern", "sketch"],
        help="Visual language: 'modern' (clean editorial) or 'sketch' (hand-drawn whiteboard)",
    )
    parser.add_argument(
        "--theme",
        default=None,
        help="modern: midnight|carbon|forest|daylight   sketch: paper|blueprint",
    )
    parser.add_argument("--show-graph", action="store_true", help="Print the graph and exit.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _ask_human(payload: dict) -> dict:
    """Render an interrupt payload in the terminal and collect a decision."""
    print("\n" + "=" * 68)
    print(f"  HUMAN INPUT NEEDED - {payload.get('type', 'approval')}")
    print("=" * 68)

    if payload.get("type") == "topic_approval":
        print(f"\n  Topic : {payload.get('selected_topic')}")
        print(f"  Angle : {payload.get('angle')}")
        print(f"  Hook  : {payload.get('hook')}")
        print(f"  Why   : {payload.get('reason')}")
        alternatives = payload.get("alternatives", [])
        if alternatives:
            print("\n  Alternatives:")
            for index, name in enumerate(alternatives, start=1):
                print(f"    {index}. {name}")
        print("\n  [a]pprove   [r]eject (pick a different one)   [c]hoose number")
        answer = input("  > ").strip().lower()

        if answer.startswith("r"):
            return {"action": "reject"}
        if answer.startswith("c"):
            index = input("  which number? > ").strip()
            with_index = alternatives[int(index) - 1] if index.isdigit() and 0 < int(index) <= len(alternatives) else ""
            return {"action": "choose_other", "topic": with_index}
        return {"action": "approve"}

    print(f"\n  Topic: {payload.get('topic')}   (critic score: {payload.get('score')})")
    for slide in payload.get("slides", []):
        print(f"    {slide['slide_number']}. [{slide['visual_type']}] {slide['title']}")
    print("\n  [a]pprove and render   [r]eject")
    answer = input("  > ").strip().lower()
    return {"action": "reject_images" if answer.startswith("r") else "approve"}


def _print_progress(node: str) -> None:
    label = NODE_LABELS.get(node, node)
    print(f"  ✓ {label}")


def run(args: argparse.Namespace) -> dict:
    """Drive the graph to completion, handling any interrupts along the way."""
    # The CLI flags drive the HITL toggles that the nodes read from settings.
    settings.require_topic_approval = args.approve_topic or settings.require_topic_approval
    settings.require_image_approval = args.approve_images or settings.require_image_approval

    thread_id = args.thread_id or f"carousel-{datetime.now():%Y%m%d-%H%M%S}"

    # The config carries the tracer callbacks as well as the thread id. Building
    # it in one place is what stopped Langfuse from silently receiving nothing.
    config = run_config(
        thread_id,
        {
            "domain": args.domain,
            "audience": args.audience,
            "platform": args.platform,
            "slide_count": args.slides,
            "slide_style": args.style or settings.slide_style,
        },
        recursion_limit=RECURSION_LIMIT,
    )

    app = get_app()

    print("\n" + "=" * 68)
    print("  AI TREND-TO-CAROUSEL AGENT")
    print("=" * 68)
    print(f"  Domain    : {args.domain}")
    print(f"  Audience  : {args.audience}")
    print(f"  Platform  : {args.platform}")
    print(f"  Slides    : {args.slides}")
    print(f"  Style     : {args.style or settings.slide_style}")
    print(f"  Model     : {describe()}")
    print(f"  Search    : {active_backend()}")
    print(f"  Tracing   : {tracing_status()}")
    print(f"  Thread id : {thread_id}")
    print("=" * 68 + "\n")

    if args.resume:
        snapshot = app.get_state(config)
        if not snapshot.values:
            print(f"  No saved state for thread '{thread_id}'. Starting fresh instead.\n")
            payload = initial_state(
                args.domain, args.audience, args.platform, args.slides, thread_id,
                slide_style=args.style or settings.slide_style,
            )
        else:
            print(f"  Resuming from: {snapshot.values.get('status', 'unknown')}\n")
            payload = None
    else:
        payload = initial_state(
            args.domain, args.audience, args.platform, args.slides, thread_id,
            slide_style=args.style or settings.slide_style,
        )
        if args.theme:
            payload["design_theme"] = args.theme

    # Drive the graph. `stream_mode="updates"` gives one event per node, which
    # is exactly what a progress display needs.
    while True:
        for chunk in app.stream(payload, config=config, stream_mode="updates"):
            for node_name, update in chunk.items():
                if node_name == "__interrupt__":
                    continue
                _print_progress(node_name)
                if isinstance(update, dict) and update.get("errors"):
                    for message in update["errors"]:
                        print(f"      ! {message}")

        # The checkpointer is authoritative about whether the graph is paused.
        snapshot = app.get_state(config)
        pending = next((task.interrupts[0] for task in snapshot.tasks if task.interrupts), None)
        if pending is None:
            break

        # RESUME: hand the human's answer back to the waiting `interrupt()` call.
        payload = Command(resume=_ask_human(pending.value))

    return app.get_state(config).values


def print_summary(state: dict) -> None:
    topic = state.get("selected_topic", {})
    critique = state.get("critique", {})

    print("\n" + "=" * 68)
    print("  RESULT")
    print("=" * 68)
    print(f"\n  TOPIC : {topic.get('selected_topic', '(none)')}")
    print(f"  ANGLE : {state.get('content_angle', '')}")
    print(f"  HOOK  : {topic.get('hook', '')}")

    print(f"\n  Quality: {critique.get('overall_score', '?')}/10"
          f"   approved={critique.get('approved')}"
          f"   revisions={state.get('revision_count', 0)}")
    if state.get("quality_flag"):
        print(f"  FLAG   : {state['quality_flag']}")

    print("\n  SLIDES")
    for slide in sorted(state.get("slides", []), key=lambda s: s["slide_number"]):
        print(f"    {slide['slide_number']}. [{slide.get('visual_type','')}] {slide.get('title','')}")
        if slide.get("subtitle"):
            print(f"       {slide['subtitle']}")

    images = state.get("image_paths", [])
    print(f"\n  IMAGES ({len(images)})")
    for path in images:
        print(f"    {path}")

    if images:
        print(f"\n  Manifest: {Path(images[0]).parent / 'carousel.json'}")

    errors = state.get("errors", [])
    if errors:
        print("\n  NOTES")
        for message in errors:
            print(f"    - {message}")
    print()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(logging.DEBUG if args.verbose else logging.INFO)

    if args.show_graph:
        from app.graph.graph import draw_ascii

        print(draw_ascii())
        path = save_diagram()
        if path:
            print(f"\nDiagram written to {path}")
        return 0

    try:
        final_state = run(args)
    except KeyboardInterrupt:
        print("\n  Interrupted. Re-run with --resume --thread-id <id> to continue.\n")
        return 130
    except Exception as exc:
        logger.exception("run failed")
        print(f"\n  FAILED: {exc}\n")
        return 1

    print_summary(final_state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
