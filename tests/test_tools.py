"""Tests for the search tool's filtering and the query builders."""
from __future__ import annotations

from datetime import date

from app.tools.research import topic_research_queries, trend_queries
from app.tools.web_search import SearchResult, _clean_url, _is_ad, format_for_llm


def test_trend_queries_are_date_aware():
    """Recency bias is what keeps discovered topics current."""
    queries = trend_queries("AI", "Developers")
    period = date.today().strftime("%B %Y")

    assert any(period in q for q in queries)
    assert any("Developers" in q for q in queries)
    assert len(queries) >= 5


def test_topic_research_queries_cover_the_research_model():
    queries = " ".join(topic_research_queries("MCP", "Developers")).lower()
    for aspect in ("explained", "how it works", "examples", "latest"):
        assert aspect in queries


def test_sponsored_results_are_dropped():
    assert _is_ad("https://www.bing.com/aclick?ld=abc")
    assert _is_ad("https://duckduckgo.com/y.js?ad_provider=x")
    assert not _is_ad("https://modelcontextprotocol.io/roadmap")


def test_duckduckgo_redirects_are_unwrapped():
    wrapped = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpost&rut=x"
    assert _clean_url(wrapped) == "https://example.com/post"
    assert _clean_url("https://example.com/direct") == "https://example.com/direct"


def test_format_for_llm_is_empty_safe():
    assert "no search results" in format_for_llm([])

    block = format_for_llm([SearchResult(title="T", url="https://e.com", snippet="S")])
    assert "https://e.com" in block and "[1]" in block
