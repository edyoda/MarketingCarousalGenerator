"""Stage 2 node: deep research on the selected topic.

Separate from trend discovery on purpose. Trend research answers "what should we
talk about"; this node answers "what do we actually know about it, and who says
so". Keeping them apart is what lets the human-approval interrupt sit between
them - you approve the topic *before* paying for the deep research.
"""
from __future__ import annotations

import logging

from app.llm import structured
from app.models import TopicResearch
from app.observability import traced_node
from app.tools.research import topic_research_queries
from app.tools.web_search import format_for_llm, multi_search

logger = logging.getLogger(__name__)

RESEARCH_PROMPT = """You are a technical researcher preparing source material for a
{platform} carousel aimed at {audience}.

TOPIC: {topic}
ANGLE: {angle}

Below are live web search results about this topic. Build the research brief
from THEM - this is the evidence base.

{results}

Rules that matter more than completeness:

1. NEVER invent a statistic, a percentage, a date or a benchmark. If the sources
   do not contain a number, do not produce one.
2. Label every entry in `key_facts` honestly:
     FACT    - directly supported by one of the sources above; put its URL in source_url
     CLAIM   - asserted somewhere but not verified; source_url if you have one
     OPINION - somebody's judgement or prediction; source_url may be empty
3. `source_url` for a FACT must be one of the real URLs listed above. Do not
   construct plausible-looking URLs.
4. Write `definition`, `problem` and `how_it_works` in plain language a busy
   developer understands on first read. Short sentences.
5. `examples` and `use_cases` should be concrete and named, not categories.

Produce 6-10 key_facts covering: definition, problem, how_it_works, example,
use_case, development and implication categories."""


@traced_node
def research_topic(state: dict) -> dict:
    """Search deeper on the chosen topic, then structure the findings."""
    topic_payload = state.get("selected_topic", {})
    topic = topic_payload.get("selected_topic", "")
    angle = state.get("content_angle", "")

    if not topic:
        return {"status": "research_failed", "errors": ["No topic selected to research."]}

    # TOOL CALL: a second, topic-specific sweep of the web.
    results = multi_search(topic_research_queries(topic, state["audience"]), max_results=5)
    logger.info("deep research gathered %d sources for %r", len(results), topic)

    # Include the sources the trend itself came from, so nothing is lost.
    seen = {r.url for r in results}
    carried = [s for s in topic_payload.get("sources", []) if s.get("url") not in seen]

    model = structured(TopicResearch, tier="smart")
    research: TopicResearch = model.invoke(
        RESEARCH_PROMPT.format(
            platform=state["platform"],
            audience=state["audience"],
            topic=topic,
            angle=angle,
            results=format_for_llm(results, limit=24),
        )
    )

    payload = research.model_dump()

    # Defensive pass: drop any FACT whose URL was not in the evidence we gathered.
    # The prompt forbids invented URLs; this makes it structurally impossible for
    # one to survive into the slides.
    known_urls = {r.url for r in results} | {s.get("url", "") for s in topic_payload.get("sources", [])}
    for fact in payload.get("key_facts", []):
        url = fact.get("source_url", "")
        if url and url not in known_urls:
            logger.warning("dropping unverifiable source on a %s: %s", fact.get("claim_type"), url)
            fact["source_url"] = ""
            if fact.get("claim_type") == "FACT":
                fact["claim_type"] = "CLAIM"

    all_sources = [r.to_dict() for r in results] + carried

    return {
        "research": payload,
        "research_sources": all_sources,
        "status": "topic_researched",
    }
