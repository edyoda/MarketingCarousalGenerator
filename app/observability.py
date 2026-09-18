"""Logging / tracing for every LangGraph node.

Each node is wrapped with @traced_node, which records:
    node started -> input summary -> latency -> output summary -> errors

Two sinks:
  * a rotating log file + console (human reading)
  * an in-memory event bus (the Streamlit UI subscribes to it for the
    live progress checklist)

LangSmith/Langfuse need no code here - LangSmith activates from env vars alone
(see .env.example), and Langfuse plugs in as a callback handler in graph.py.
"""
from __future__ import annotations

import functools
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from langgraph.errors import GraphInterrupt

from app.config import LOG_DIR

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Idempotent logging setup - safe to call from CLI and Streamlit alike."""
    root = logging.getLogger()
    if any(getattr(h, "_carousel_handler", False) for h in root.handlers):
        return

    root.setLevel(level)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_LOG_FORMAT))
    console._carousel_handler = True  # type: ignore[attr-defined]
    root.addHandler(console)

    file_handler = logging.FileHandler(LOG_DIR / "carousel.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    file_handler._carousel_handler = True  # type: ignore[attr-defined]
    root.addHandler(file_handler)

    # These are chatty and drown out the node trace.
    for noisy in ("httpx", "httpcore", "urllib3", "primp", "ddgs", "openai", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@dataclass
class NodeEvent:
    """One observable moment in the graph run."""

    node: str
    phase: str  # started | finished | paused | failed
    message: str = ""
    latency_ms: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    run_id: str = ""


class EventBus:
    """Dead-simple pub/sub. Enough for a teaching project; swap for a queue
    or a real tracer without touching node code."""

    def __init__(self) -> None:
        self._subscribers: list[Callable[[NodeEvent], None]] = []
        self.history: list[NodeEvent] = []

    def subscribe(self, fn: Callable[[NodeEvent], None]) -> Callable[[], None]:
        self._subscribers.append(fn)
        return lambda: self._subscribers.remove(fn)

    def emit(self, event: NodeEvent) -> None:
        self.history.append(event)
        for fn in list(self._subscribers):
            try:
                fn(event)
            except Exception:  # a broken listener must never break the graph
                logging.getLogger(__name__).debug("event subscriber failed", exc_info=True)

    def reset(self) -> None:
        self.history.clear()


bus = EventBus()

# Human-readable labels used by the UI progress list.
NODE_LABELS: dict[str, str] = {
    "trend_research": "Discovering trends",
    "trend_analyzer": "Analyzing trends",
    "topic_selection": "Selecting topic",
    "human_topic_approval": "Waiting for topic approval",
    "research_topic": "Researching topic",
    "content_strategy": "Creating content strategy",
    "carousel_planner": "Planning carousel",
    "generate_slide": "Generating slides",
    "merge_slides": "Merging slides",
    "carousel_critic": "Reviewing content",
    "prepare_rewrite": "Planning revisions",
    "rewrite_slide": "Improving content",
    "flag_quality": "Flagging quality",
    "human_image_approval": "Waiting for render approval",
    "render_images": "Generating images",
    "image_quality_check": "Checking images",
    "finalize": "Complete",
}


def _summarize(value: Any, depth: int = 0) -> Any:
    """Shrink state into something loggable - never dump full research text."""
    if depth > 2:
        return "..."
    if isinstance(value, str):
        return value if len(value) <= 120 else value[:117] + "..."
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return f"list[{len(value)}]"
    if isinstance(value, dict):
        return {k: _summarize(v, depth + 1) for k, v in list(value.items())[:8]}
    return type(value).__name__


def traced_node(fn: Callable) -> Callable:
    """Decorator applied to every graph node.

    Keeping observability in a decorator means node bodies stay readable and
    contain only business logic - which is the point of LangGraph nodes.
    """
    name = fn.__name__
    logger = logging.getLogger(f"node.{name}")

    @functools.wraps(fn)
    def wrapper(state, *args, **kwargs):
        run_id = uuid.uuid4().hex[:8]
        label = NODE_LABELS.get(name, name)
        started = time.perf_counter()

        inputs = _summarize(state) if isinstance(state, dict) else _summarize(vars(state))
        logger.info("START %s | input=%s", label, json.dumps(inputs, default=str)[:600])
        bus.emit(NodeEvent(node=name, phase="started", message=label, run_id=run_id))

        try:
            result = fn(state, *args, **kwargs)
        except GraphInterrupt:
            # An interrupt is normal control flow, not an error: `interrupt()`
            # raises to suspend the graph so the checkpointer can persist it.
            # Logging it as a failure (with a traceback) would be misleading.
            latency = (time.perf_counter() - started) * 1000
            logger.info("PAUSE %s | waiting for human input", label)
            bus.emit(
                NodeEvent(
                    node=name,
                    phase="paused",
                    message=label,
                    latency_ms=latency,
                    run_id=run_id,
                )
            )
            raise
        except Exception as exc:
            latency = (time.perf_counter() - started) * 1000
            logger.exception("FAILED %s after %.0fms", label, latency)
            bus.emit(
                NodeEvent(
                    node=name,
                    phase="failed",
                    message=f"{label}: {exc}",
                    latency_ms=latency,
                    run_id=run_id,
                )
            )
            raise

        latency = (time.perf_counter() - started) * 1000
        # A node may return a Command (routing) rather than a plain dict.
        payload = getattr(result, "update", result)
        outputs = _summarize(payload) if isinstance(payload, dict) else _summarize(result)
        logger.info("DONE  %s | %.0fms | output=%s", label, latency, json.dumps(outputs, default=str)[:600])
        bus.emit(
            NodeEvent(
                node=name,
                phase="finished",
                message=label,
                latency_ms=latency,
                detail=outputs if isinstance(outputs, dict) else {},
                run_id=run_id,
            )
        )
        return result

    return wrapper


def get_callbacks() -> list:
    """Optional tracer callbacks (Langfuse). LangSmith needs no callback."""
    callbacks: list = []
    try:
        import os

        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            from langfuse.callback import CallbackHandler

            callbacks.append(CallbackHandler())
    except Exception:
        logging.getLogger(__name__).debug("Langfuse not configured", exc_info=True)
    return callbacks
