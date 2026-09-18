"""Graph assembly - where every LangGraph concept in this project comes together.

    START
      |
    trend_research          (tool node: live web search)
      |
    trend_analyzer          (LLM + structured output)
      |
    topic_selection         (LLM: picks topic + angle)  <---------+
      |                                                           |
    human_topic_approval    (INTERRUPT)                           | reject
      +------------------- route_after_topic_approval ------------+
      | approve
    research_topic          (tool + LLM: deep research)
      |
    content_strategy        (LLM: narrative arc)
      |
    carousel_planner        (LLM: per-slide brief)
      |
      +== fan_out_slides ==> generate_slide x N  (PARALLEL)
                                   |
                              merge_slides       (fan-in)
                                   |
                            carousel_critic      (LLM judge)  <---+
                                   |                              |
                          +-- quality_gate --+                    |
                 approved |                  | rewrite            |
                          |            prepare_rewrite            |
                          |                  |                    |
                          |     +== fan_out_rewrites ==>          |
                          |          rewrite_slide x K (PARALLEL) |
                          |                  +-------------------- LOOP
                          |
                          |  give_up -> flag_quality --+
                          |                            |
                    human_image_approval  (INTERRUPT) <+
                          |
                    render_images         (HTML -> PNG)
                          |
                    image_quality_check   (deterministic)
                          |
                       finalize
                          |
                         END
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import suppress

from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.graph.edges import (
    fan_out_rewrites,
    fan_out_slides,
    quality_gate,
    route_after_image_approval,
    route_after_image_check,
    route_after_topic_approval,
)
from app.graph.nodes.critic import carousel_critic, prepare_rewrite
from app.graph.nodes.images import (
    finalize,
    flag_quality,
    human_image_approval,
    image_quality_check,
    render_images,
)
from app.graph.nodes.research import research_topic
from app.graph.nodes.slides import generate_slide, merge_slides, rewrite_slide
from app.graph.nodes.strategy import carousel_planner, content_strategy
from app.graph.nodes.trends import (
    human_topic_approval,
    topic_selection,
    trend_analyzer,
    trend_research,
)
from app.graph.state import CarouselState

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """Wire the nodes and edges. No compilation here - see `get_app`."""
    graph = StateGraph(CarouselState)

    # ---- nodes ------------------------------------------------------------
    graph.add_node("trend_research", trend_research)
    graph.add_node("trend_analyzer", trend_analyzer)
    graph.add_node("topic_selection", topic_selection)
    graph.add_node("human_topic_approval", human_topic_approval)
    graph.add_node("research_topic", research_topic)
    graph.add_node("content_strategy", content_strategy)
    graph.add_node("carousel_planner", carousel_planner)
    graph.add_node("generate_slide", generate_slide)     # runs N times in parallel
    graph.add_node("merge_slides", merge_slides)
    graph.add_node("carousel_critic", carousel_critic)
    graph.add_node("prepare_rewrite", prepare_rewrite)
    graph.add_node("rewrite_slide", rewrite_slide)       # runs K times in parallel
    graph.add_node("flag_quality", flag_quality)
    graph.add_node("human_image_approval", human_image_approval)
    graph.add_node("render_images", render_images)
    graph.add_node("image_quality_check", image_quality_check)
    graph.add_node("finalize", finalize)

    # ---- linear spine -----------------------------------------------------
    graph.add_edge(START, "trend_research")
    graph.add_edge("trend_research", "trend_analyzer")
    graph.add_edge("trend_analyzer", "topic_selection")
    graph.add_edge("topic_selection", "human_topic_approval")

    # ---- CONDITIONAL EDGE 1: human approval can send us back to re-select --
    graph.add_conditional_edges(
        "human_topic_approval",
        route_after_topic_approval,
        {"research_topic": "research_topic", "topic_selection": "topic_selection"},
    )

    graph.add_edge("research_topic", "content_strategy")
    graph.add_edge("content_strategy", "carousel_planner")

    # ---- PARALLEL FAN-OUT: one generate_slide per planned slide ------------
    # The router returns a list of Send objects, so LangGraph starts N copies of
    # `generate_slide` concurrently. They converge on `merge_slides` because
    # that is the only edge out of `generate_slide`.
    graph.add_conditional_edges(
        "carousel_planner",
        fan_out_slides,
        ["generate_slide", "merge_slides"],
    )
    graph.add_edge("generate_slide", "merge_slides")
    graph.add_edge("merge_slides", "carousel_critic")

    # ---- CONDITIONAL EDGE 2 + THE LOOP ------------------------------------
    graph.add_conditional_edges(
        "carousel_critic",
        quality_gate,
        {
            "approved": "human_image_approval",
            "rewrite": "prepare_rewrite",
            "give_up": "flag_quality",
        },
    )

    # prepare_rewrite bumps the counter, then fans out to the rewrite workers,
    # which loop back into the critic. That is the cycle.
    graph.add_conditional_edges(
        "prepare_rewrite",
        fan_out_rewrites,
        ["rewrite_slide", "carousel_critic"],
    )
    graph.add_edge("rewrite_slide", "carousel_critic")

    graph.add_edge("flag_quality", "human_image_approval")

    # ---- CONDITIONAL EDGE 3: last human gate before rendering -------------
    graph.add_conditional_edges(
        "human_image_approval",
        route_after_image_approval,
        {"render_images": "render_images", "finalize": "finalize"},
    )

    graph.add_edge("render_images", "image_quality_check")

    # ---- CONDITIONAL EDGE 4: bounded re-render retry ----------------------
    graph.add_conditional_edges(
        "image_quality_check",
        route_after_image_check,
        {"render_images": "render_images", "finalize": "finalize"},
    )

    graph.add_edge("finalize", END)
    return graph


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def make_checkpointer():
    """A SQLite checkpointer so runs survive a process restart.

    LANGGRAPH CONCEPT: PERSISTENCE
    Every super-step writes the state to SQLite, keyed by thread_id. That gives
    us three things at once:
      * resume - re-invoke with the same thread_id and work continues
      * interrupts - the state is durable while a human decides
      * history - `app.get_state_history(config)` replays the whole run

    `check_same_thread=False` is required because Streamlit and LangGraph may
    touch the connection from different threads.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    connection = sqlite3.connect(str(settings.checkpoint_db), check_same_thread=False)
    return SqliteSaver(connection)


_compiled = None


def get_app(checkpointer=None, fresh: bool = False):
    """Compile the graph. Cached, because compiling on every call is wasteful."""
    global _compiled
    if _compiled is not None and not fresh and checkpointer is None:
        return _compiled

    saver = checkpointer if checkpointer is not None else make_checkpointer()
    compiled = build_graph().compile(checkpointer=saver)

    if checkpointer is None and not fresh:
        _compiled = compiled
    return compiled


def draw_ascii() -> str:
    """Render the graph topology as text - useful when teaching from it."""
    with suppress(Exception):
        return get_app().get_graph().draw_ascii()
    return "(install grandalf to draw the graph: pip install grandalf)"


def save_diagram(path: str = "output/graph.png") -> str | None:
    """Write a PNG of the graph topology, if mermaid rendering is available."""
    try:
        png = get_app().get_graph().draw_mermaid_png()
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(png)
        return path
    except Exception as exc:
        logger.info("could not render graph diagram: %s", exc)
        return None


def mermaid() -> str:
    """Mermaid source for the graph - embeddable in the README."""
    return get_app().get_graph().draw_mermaid()
