"""LangGraph state definition.

LANGGRAPH CONCEPT: STATE
------------------------
Every node receives the state and returns a *partial* update. LangGraph merges
that update into the state. How it merges is controlled by the reducer attached
to each key with `Annotated[type, reducer]`:

    slides: Annotated[list[dict], upsert_slides]   <- custom merge (parallel-safe)
    errors: Annotated[list[str], operator.add]    <- append
    status: str                                   <- plain overwrite (last write wins)

The reducer matters enormously once nodes run in parallel: five slide writers
all return `{"slides": [...]}` at the same time, and without a reducer LangGraph
would raise an InvalidUpdateError for concurrent writes to one key.
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict


def upsert_slides(existing: list[dict] | None, update: list[dict] | None) -> list[dict]:
    """Reducer for the `slides` key: upsert by slide_number, keep sorted.

    Named `upsert_slides`, not `merge_slides`, so it is never confused with the
    graph NODE called `merge_slides` - the reducer runs on every write, the node
    runs once as the fan-in barrier.

    Why not `operator.add`? Two reasons, both of which show up in this graph:

    1. FAN-OUT - the five parallel writers each append one slide. Plain `add`
       works here, but ordering would be non-deterministic (whichever finishes
       first lands first).
    2. THE REWRITE LOOP - when the critic rejects slides 2 and 4, the rewrite
       workers emit *replacements*. With `add` we would end up with seven slides
       and two duplicates. Upserting by slide_number keeps the list correct on
       every pass through the loop.
    """
    merged: dict[int, dict] = {}
    for slide in existing or []:
        merged[slide["slide_number"]] = slide
    for slide in update or []:
        merged[slide["slide_number"]] = slide
    return [merged[k] for k in sorted(merged)]


class CarouselState(TypedDict, total=False):
    """The single state object flowing through the whole graph."""

    # ---- user input -------------------------------------------------------
    domain: str
    audience: str
    platform: str
    slide_count: int

    # ---- stage 1: trend discovery ----------------------------------------
    search_results: list[dict]      # raw normalised hits from the web
    trends: list[dict]              # Trend models, ranked
    selected_topic: dict            # TopicSelection model
    content_angle: str
    rejected_topics: Annotated[list[str], operator.add]  # topics a human turned down

    # ---- stage 2: research + strategy ------------------------------------
    research: dict                  # TopicResearch model
    research_sources: list[dict]
    strategy: dict                  # ContentStrategy model
    carousel_plan: list[dict]       # list[SlidePlan]

    # ---- stage 3: slides (written in parallel) ---------------------------
    slides: Annotated[list[dict], upsert_slides]

    # ---- stage 4: critique loop ------------------------------------------
    critique: dict                  # latest Critique
    critique_history: Annotated[list[dict], operator.add]
    revision_count: int
    quality_flag: str               # "" | "max_revisions_reached"

    # ---- stage 5: rendering ----------------------------------------------
    image_paths: list[str]
    image_check: dict
    design_theme: str      # palette: midnight/carbon/... or paper/blueprint
    slide_style: str       # visual language: "modern" | "sketch"
    render_attempts: int

    # ---- human-in-the-loop ------------------------------------------------
    human_decision: str             # approve | reject | choose_other
    human_note: str

    # ---- bookkeeping ------------------------------------------------------
    status: str
    errors: Annotated[list[str], operator.add]
    run_id: str


class SlideWorkerState(TypedDict):
    """Payload handed to ONE parallel slide worker via `Send`.

    LANGGRAPH CONCEPT: SEND / MAP-REDUCE
    A `Send("generate_slide", payload)` invokes the node with *this* dict as its
    state instead of the full CarouselState. The node's return value is still
    merged back into the parent state through the reducers above.
    """

    plan: dict
    topic: dict
    research: dict
    strategy: dict
    audience: str
    platform: str
    slide_count: int
    # only present on the rewrite pass
    previous_slide: dict
    issues: list[dict]


def initial_state(
    domain: str,
    audience: str,
    platform: str = "LinkedIn",
    slide_count: int = 5,
    run_id: str = "",
    slide_style: str = "",
) -> CarouselState:
    """Build a clean starting state from the four user inputs."""
    return CarouselState(
        domain=domain,
        audience=audience,
        platform=platform,
        slide_count=slide_count,
        search_results=[],
        trends=[],
        selected_topic={},
        content_angle="",
        rejected_topics=[],
        research={},
        research_sources=[],
        strategy={},
        carousel_plan=[],
        slides=[],
        critique={},
        critique_history=[],
        revision_count=0,
        quality_flag="",
        image_paths=[],
        image_check={},
        design_theme="",
        slide_style=slide_style,
        render_attempts=0,
        human_decision="",
        human_note="",
        status="starting",
        errors=[],
        run_id=run_id,
    )
