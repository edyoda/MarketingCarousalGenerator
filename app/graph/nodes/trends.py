"""Stage 1 nodes: discover what is trending, analyze it, pick one topic.

LANGGRAPH CONCEPT: NODES
A node is just a function `state -> partial state update`. No node here calls
another node directly; the graph wires them together. That is what makes the
flow inspectable, resumable and testable.
"""
from __future__ import annotations

import logging

from langgraph.types import interrupt

from app.config import settings
from app.llm import structured
from app.models import TopicSelection, TrendList
from app.observability import traced_node
from app.tools.research import trend_queries
from app.tools.web_search import active_backend, format_for_llm, multi_search

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Trend research  (TOOL-CALLING NODE - no LLM, just the web)
# ---------------------------------------------------------------------------
@traced_node
def trend_research(state: dict) -> dict:
    """Search the live web for what is happening in the user's domain.

    This is the node that guarantees requirement #4: the topic is never
    hard-coded. Everything downstream is derived from these results.
    """
    domain = state["domain"]
    audience = state["audience"]

    queries = trend_queries(domain, audience)
    logger.info("searching %d queries via %s", len(queries), active_backend())

    results = multi_search(queries, max_results=settings.search_results_per_query)

    if not results:
        return {
            "search_results": [],
            "status": "trend_research_empty",
            "errors": ["Web search returned no results - check network or TAVILY_API_KEY."],
        }

    return {
        "search_results": [r.to_dict() for r in results],
        "status": "trends_discovered",
    }


# ---------------------------------------------------------------------------
# 2. Trend analyzer  (LLM NODE - structured output)
# ---------------------------------------------------------------------------
TREND_ANALYZER_PROMPT = """You are a social media content strategist analysing live web search results.

Domain: {domain}
Target audience: {audience}
Platform: {platform}

Below are real search results gathered moments ago. Extract the DISTINCT trends
they describe. Do not invent trends that are not supported by these results.

{results}

For every trend you extract:
- `topic`: a specific, concrete name (not "AI is growing")
- `why_trending`: what happened recently to cause the attention
- `sources`: the URLs from the list above that support it (use the real URLs)
- `recency`: how fresh it looks based on the snippets
- `relevance_score` (0-10): how much {audience} would care
- `content_potential_score` (0-10): how well it would work as a visual,
  5-slide social carousel. A topic that needs 2000 words to explain scores low.
  A topic with a clear before/after, architecture or process scores high.

Score honestly and spread the scores out - if everything scores 8 the ranking is
useless. Return between 4 and 8 trends."""


@traced_node
def trend_analyzer(state: dict) -> dict:
    """Turn raw search noise into scored, structured Trend objects.

    LANGGRAPH CONCEPT: STRUCTURED OUTPUT
    `with_structured_output(TrendList)` forces the model to answer as a
    validated Pydantic object, so the next node can rely on the shape.
    """
    results_block = format_for_llm(
        [_as_result(r) for r in state.get("search_results", [])],
        limit=28,
    )

    model = structured(TrendList, tier="smart")
    response: TrendList = model.invoke(
        TREND_ANALYZER_PROMPT.format(
            domain=state["domain"],
            audience=state["audience"],
            platform=state["platform"],
            results=results_block,
        )
    )

    # Deterministic ranking happens in code, not in the prompt: the model
    # supplies the judgement, Python supplies the ordering.
    ranked = sorted(response.trends, key=lambda t: t.combined_score, reverse=True)

    return {
        "trends": [t.model_dump() for t in ranked],
        "status": "trends_analyzed",
    }


def _as_result(payload: dict):
    """Rehydrate a stored dict into the SearchResult shape the formatter wants."""
    from app.tools.web_search import SearchResult

    return SearchResult(
        title=payload.get("title", ""),
        url=payload.get("url", ""),
        snippet=payload.get("snippet", ""),
        published=payload.get("published", ""),
        query=payload.get("query", ""),
    )


# ---------------------------------------------------------------------------
# 3. Topic selection  (LLM NODE - picks the angle)
# ---------------------------------------------------------------------------
TOPIC_SELECTION_PROMPT = """You are choosing ONE topic for a {slide_count}-slide {platform} carousel.

Audience: {audience}
Domain: {domain}

Here are the ranked trend candidates discovered from live web research:

{candidates}

Pick the single best topic for a carousel. The highest-scoring trend is NOT
automatically the right answer - prefer the one that can be explained visually
in {slide_count} slides and gives {audience} something genuinely useful.

{rejection_note}Then define the ANGLE. The angle is the difference between a boring post and a
good one:

  BAD angle:    "MCP is a protocol."
  GOOD angle:   "Why AI Agents Need a Standard Way to Talk to Tools"

  BAD angle:    "An overview of vector databases."
  GOOD angle:   "Your RAG App Is Slow Because You Chose the Wrong Index"

Keep `content_angle` to ONE sentence - it is the editorial line, not the slide
plan. The slides get planned later by a different node.

Also write a `hook`: the first line of slide 1. It must make someone stop
scrolling. No "In today's world". No "Let's dive in"."""


@traced_node
def topic_selection(state: dict) -> dict:
    """Select the topic and the editorial angle to take on it."""
    trends = state.get("trends", [])
    if not trends:
        return {
            "status": "topic_selection_failed",
            "errors": ["No trends available to select from."],
        }

    rejected = state.get("rejected_topics", [])
    # Skip anything a human already turned down on a previous pass through the
    # approval loop. The comparison is fuzzy because what gets rejected is the
    # model's phrasing of the topic, which rarely matches the trend's name
    # character for character.
    available = [t for t in trends if not _matches_any(t["topic"], rejected)] or trends
    shortlist = available[:6]
    candidates = "\n\n".join(
        f"- {t['topic']}\n"
        f"  why trending: {t['why_trending']}\n"
        f"  recency: {t.get('recency', 'unknown')} | "
        f"relevance: {t.get('relevance_score', 0)}/10 | "
        f"carousel potential: {t.get('content_potential_score', 0)}/10\n"
        f"  sources: {', '.join(s['url'] for s in t.get('sources', [])[:3])}"
        for t in shortlist
    )

    rejection_note = ""
    if rejected:
        rejection_note = (
            "A human reviewer already REJECTED these topics - do not pick them again:\n"
            + "\n".join(f"  - {name}" for name in rejected)
            + "\n\n"
        )

    model = structured(TopicSelection, tier="smart")
    choice: TopicSelection = model.invoke(
        TOPIC_SELECTION_PROMPT.format(
            slide_count=state["slide_count"],
            platform=state["platform"],
            audience=state["audience"],
            domain=state["domain"],
            candidates=candidates,
            rejection_note=rejection_note,
        )
    )

    # Carry the matching trend's sources forward so claims stay traceable.
    matched = _match_trend(choice.selected_topic, trends)
    payload = choice.model_dump()
    payload["sources"] = matched.get("sources", []) if matched else []
    payload["why_trending"] = matched.get("why_trending", "") if matched else ""

    return {
        "selected_topic": payload,
        "content_angle": choice.content_angle,
        "status": "topic_selected",
    }


def _matches_any(topic: str, candidates: list[str]) -> bool:
    """Loose containment test used to filter out already-rejected topics."""
    needle = (topic or "").lower().strip()
    if not needle:
        return False
    return any(
        needle in other.lower() or other.lower() in needle
        for other in candidates
        if other and other.strip()
    )


def _match_trend(topic: str, trends: list[dict]) -> dict | None:
    """Fuzzy-match the chosen topic back to the trend it came from."""
    needle = topic.lower()
    for trend in trends:
        name = trend["topic"].lower()
        if name in needle or needle in name:
            return trend
    # fall back to word overlap
    needle_words = set(needle.split())
    best, best_overlap = None, 0
    for trend in trends:
        overlap = len(needle_words & set(trend["topic"].lower().split()))
        if overlap > best_overlap:
            best, best_overlap = trend, overlap
    return best


# ---------------------------------------------------------------------------
# 4. Human approval  (INTERRUPT NODE)
# ---------------------------------------------------------------------------
@traced_node
def human_topic_approval(state: dict) -> dict:
    """Pause the graph and wait for a human verdict on the chosen topic.

    LANGGRAPH CONCEPT: HUMAN-IN-THE-LOOP / INTERRUPT
    `interrupt(payload)` stops execution and persists the state via the
    checkpointer. The caller sees the payload, then resumes with
    `graph.invoke(Command(resume={...}), config)` and execution continues from
    exactly this line - no replay of the expensive research above.
    """
    if not settings.require_topic_approval:
        return {"human_decision": "auto_approved"}

    topic = state.get("selected_topic", {})
    decision = interrupt(
        {
            "type": "topic_approval",
            "question": "Approve this topic for the carousel?",
            "selected_topic": topic.get("selected_topic", ""),
            "angle": topic.get("content_angle", ""),
            "hook": topic.get("hook", ""),
            "reason": topic.get("reason", ""),
            "alternatives": [t["topic"] for t in state.get("trends", [])[:6]],
            "options": ["approve", "reject", "choose_other"],
        }
    )

    # `decision` is whatever the caller passed to Command(resume=...)
    if isinstance(decision, str):
        decision = {"action": decision}
    action = (decision or {}).get("action", "approve")

    if action == "choose_other":
        alternative = decision.get("topic", "")
        chosen = _match_trend(alternative, state.get("trends", [])) or {}
        new_topic = dict(state.get("selected_topic", {}))
        new_topic["selected_topic"] = alternative or new_topic.get("selected_topic", "")
        new_topic["sources"] = chosen.get("sources", [])
        new_topic["reason"] = "Chosen by a human reviewer."
        return {
            "selected_topic": new_topic,
            "human_decision": "choose_other",
            "human_note": decision.get("note", ""),
            "status": "topic_replaced_by_human",
        }

    if action == "reject":
        # Remembering the rejection is what stops the re-selection loop from
        # proposing the same topic forever. Store the model's phrasing AND the
        # underlying trend name, since the selector filters on the latter.
        rejected = [topic.get("selected_topic", "")]
        source_trend = _match_trend(topic.get("selected_topic", ""), state.get("trends", []))
        if source_trend and source_trend["topic"] not in rejected:
            rejected.append(source_trend["topic"])

        return {
            "human_decision": "reject",
            "human_note": (decision or {}).get("note", ""),
            "rejected_topics": [name for name in rejected if name],
            "status": "topic_rejected",
        }

    return {
        "human_decision": action,
        "human_note": (decision or {}).get("note", ""),
        "status": "topic_approved",
    }
