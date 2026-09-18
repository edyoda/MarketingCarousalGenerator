"""Fast tests for the graph's pure logic - no API key, no network.

These cover the parts students are most likely to break when experimenting:
the reducer, the routers and the fan-out payloads.
"""
from __future__ import annotations

import pytest

from app.graph.edges import (
    fan_out_rewrites,
    fan_out_slides,
    quality_gate,
    route_after_image_approval,
    route_after_image_check,
    route_after_topic_approval,
)
from app.graph.state import initial_state, upsert_slides


# ---------------------------------------------------------------------------
# reducer
# ---------------------------------------------------------------------------
def test_upsert_slides_sorts_parallel_results():
    """Workers finish out of order; the deck must still be 1..N."""
    out_of_order = [{"slide_number": 3}, {"slide_number": 1}, {"slide_number": 2}]
    merged = upsert_slides([], out_of_order)
    assert [s["slide_number"] for s in merged] == [1, 2, 3]


def test_upsert_slides_upserts_rewrites_instead_of_appending():
    """The rewrite loop replaces slides - it must not duplicate them."""
    existing = [{"slide_number": 1, "title": "a"}, {"slide_number": 2, "title": "b"}]
    rewritten = [{"slide_number": 2, "title": "b-v2"}]
    merged = upsert_slides(existing, rewritten)

    assert len(merged) == 2
    assert merged[1]["title"] == "b-v2"


def test_upsert_slides_handles_empty_sides():
    assert upsert_slides(None, None) == []
    assert upsert_slides([{"slide_number": 1}], None) == [{"slide_number": 1}]


# ---------------------------------------------------------------------------
# quality gate - the loop's exit conditions
# ---------------------------------------------------------------------------
def test_quality_gate_approves():
    assert quality_gate({"critique": {"approved": True}, "revision_count": 0}) == "approved"


def test_quality_gate_requests_rewrite():
    state = {"critique": {"approved": False}, "revision_count": 0}
    assert quality_gate(state) == "rewrite"


def test_quality_gate_gives_up_at_the_ceiling():
    """The loop must terminate even if the critic never approves."""
    from app.config import settings

    state = {"critique": {"approved": False}, "revision_count": settings.max_revisions}
    assert quality_gate(state) == "give_up"


# ---------------------------------------------------------------------------
# routers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "decision,expected",
    [
        ("approve", "research_topic"),
        ("auto_approved", "research_topic"),
        ("choose_other", "research_topic"),
        ("reject", "topic_selection"),
    ],
)
def test_route_after_topic_approval(decision, expected):
    assert route_after_topic_approval({"human_decision": decision}) == expected


def test_route_after_image_approval():
    assert route_after_image_approval({"human_decision": "approve"}) == "render_images"
    assert route_after_image_approval({"human_decision": "reject_images"}) == "finalize"


def test_route_after_image_check_retries_once_then_stops():
    bad = {"image_check": {"all_rendered": False}, "render_attempts": 1}
    assert route_after_image_check(bad) == "render_images"

    exhausted = {"image_check": {"all_rendered": False}, "render_attempts": 2}
    assert route_after_image_check(exhausted) == "finalize"

    good = {"image_check": {"all_rendered": True}, "render_attempts": 1}
    assert route_after_image_check(good) == "finalize"


# ---------------------------------------------------------------------------
# fan-out
# ---------------------------------------------------------------------------
def _state_with_plan(count: int = 5) -> dict:
    state = dict(initial_state("AI", "Developers", "LinkedIn", count))
    state["carousel_plan"] = [
        {"slide_number": n, "purpose": "p", "key_message": "m", "visual_type": "DIAGRAM"}
        for n in range(1, count + 1)
    ]
    state["selected_topic"] = {"selected_topic": "Topic"}
    return state


def test_fan_out_slides_dispatches_one_worker_per_slide():
    sends = fan_out_slides(_state_with_plan(5))

    assert len(sends) == 5
    assert all(send.node == "generate_slide" for send in sends)
    assert [send.arg["plan"]["slide_number"] for send in sends] == [1, 2, 3, 4, 5]
    # every worker gets the shared context it needs
    assert all("research" in send.arg and "strategy" in send.arg for send in sends)


def test_fan_out_rewrites_targets_only_rejected_slides():
    state = _state_with_plan(5)
    state["slides"] = [{"slide_number": n, "title": f"t{n}"} for n in range(1, 6)]
    state["critique"] = {
        "slides_to_rewrite": [2, 4],
        "issues": [
            {"slide_number": 2, "issue": "weak", "fix": "sharpen", "severity": "high"},
            {"slide_number": 5, "issue": "other", "fix": "ignore me", "severity": "low"},
        ],
    }

    sends = fan_out_rewrites(state)

    assert [send.arg["plan"]["slide_number"] for send in sends] == [2, 4]
    # each worker sees the previous version and only ITS issues
    assert sends[0].arg["previous_slide"]["title"] == "t2"
    assert [i["slide_number"] for i in sends[0].arg["issues"]] == [2]
    assert sends[1].arg["issues"] == []


def test_fan_out_slides_without_a_plan_does_not_hang():
    sends = fan_out_slides(dict(initial_state("AI", "Devs")))
    assert sends[0].node == "merge_slides"


# ---------------------------------------------------------------------------
# rejected-topic filtering
# ---------------------------------------------------------------------------
def test_rejected_topic_matching_is_fuzzy():
    """The model's phrasing of a topic rarely matches the trend name exactly."""
    from app.graph.nodes.trends import _matches_any

    rejected = ["AI Agents Moving Into Production Systems"]

    assert _matches_any("AI Agents Moving Into Production Systems", rejected)
    assert _matches_any("AI Agents Moving Into Production", rejected)   # substring
    assert not _matches_any("Local AI and Open-Source Models", rejected)
    assert not _matches_any("", rejected)
    assert not _matches_any("Anything", [])
