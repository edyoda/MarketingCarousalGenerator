"""Web search tool.

This is the project's only source of "what is happening right now". The graph
never hard-codes a topic - every trend comes out of this module.

Two backends:
  * Tavily  - used automatically when TAVILY_API_KEY is set (better snippets)
  * DuckDuckGo (ddgs) - keyless default so the demo runs with no extra signup

Both return the same normalised shape, so nodes don't care which one ran.
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from urllib.parse import parse_qs, unquote, urlparse

from app.config import settings, TAVILY_API_KEY

logger = logging.getLogger(__name__)

# ddgs logs a warning every time a backend is unavailable. We handle that by
# falling through to the next one, so the warnings are noise.
logging.getLogger("ddgs").setLevel(logging.ERROR)

# DuckDuckGo's HTML endpoint mixes sponsored results into the list. They are
# useless as research sources, so they get dropped before the LLM ever sees them.
_AD_HOST_MARKERS = (
    "bing.com/aclick",
    "duckduckgo.com/y.js",
    "googleadservices",
    "doubleclick.net",
)
_LOW_VALUE_HOSTS = {"pinterest.com", "facebook.com", "instagram.com", "x.com", "twitter.com"}


@dataclass
class SearchResult:
    """One normalised search hit."""

    title: str
    url: str
    snippet: str
    published: str = ""
    query: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _is_ad(url: str) -> bool:
    return any(marker in url for marker in _AD_HOST_MARKERS)


def _clean_url(url: str) -> str:
    """Unwrap DuckDuckGo redirect links (…/l/?uddg=<encoded real url>)."""
    if "duckduckgo.com/l/" in url:
        qs = parse_qs(urlparse(url).query)
        if "uddg" in qs:
            return unquote(qs["uddg"][0])
    return url


def _host(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "").lower()
    except Exception:
        return ""


def _search_tavily(query: str, max_results: int) -> list[SearchResult]:
    from tavily import TavilyClient

    client = TavilyClient(api_key=TAVILY_API_KEY)
    raw = client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",
        topic="news",
        days=30,
    )
    return [
        SearchResult(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=(item.get("content") or "")[:600],
            published=item.get("published_date", "") or "",
            query=query,
        )
        for item in raw.get("results", [])
    ]


# ddgs rotates across several upstream engines, and any one of them can be
# rate-limited or returning TLS errors at a given moment. Rather than pinning a
# single engine (which just moves the problem), try them in order and take the
# first that actually answers.
# Valid names as of ddgs 9.x: brave, duckduckgo, google, grokipedia, mojeek,
# startpage, wikipedia, yahoo. Unknown names are ignored by ddgs (it falls back
# to "auto"), so this list stays safe across versions.
_DDG_BACKENDS = ("auto", "duckduckgo", "google", "startpage", "brave", "mojeek")


def _parse_ddg_items(items, query: str, max_results: int) -> list[SearchResult]:
    results: list[SearchResult] = []
    for item in items:
        url = _clean_url(item.get("href") or item.get("url") or "")
        if not url or _is_ad(url) or _host(url) in _LOW_VALUE_HOSTS:
            continue
        results.append(
            SearchResult(
                title=item.get("title", ""),
                url=url,
                snippet=(item.get("body") or "")[:600],
                query=query,
            )
        )
        if len(results) >= max_results:
            break
    return results


def _search_ddg(query: str, max_results: int) -> list[SearchResult]:
    from ddgs import DDGS

    last_error: Exception | None = None
    for backend in _DDG_BACKENDS:
        try:
            with DDGS() as ddgs:
                # `timelimit="m"` biases towards the last month, which is what
                # makes a discovered trend actually current rather than evergreen.
                kwargs = {"max_results": max_results * 2, "timelimit": "m"}
                if backend != "auto":
                    kwargs["backend"] = backend
                items = ddgs.text(query, **kwargs)

            results = _parse_ddg_items(items, query, max_results)
            if results:
                if backend != "auto":
                    logger.debug("query %r answered by fallback backend %s", query, backend)
                return results
        except Exception as exc:
            last_error = exc
            continue

    # Every backend either errored or returned nothing. Say so once, at WARNING
    # - a silent empty result is how a run ends up researching thin air.
    logger.warning(
        "no search results for %r after trying %d backends%s",
        query,
        len(_DDG_BACKENDS),
        f" (last error: {last_error})" if last_error else "",
    )
    return []


def search(query: str, max_results: int | None = None) -> list[SearchResult]:
    """Run one query against the active backend. Never raises."""
    max_results = max_results or settings.search_results_per_query
    try:
        if TAVILY_API_KEY:
            return _search_tavily(query, max_results)
        return _search_ddg(query, max_results)
    except Exception as exc:  # a dead search backend must not kill the graph
        logger.warning("search failed for %r: %s", query, exc)
        return []


def multi_search(queries: list[str], max_results: int | None = None) -> list[SearchResult]:
    """Run several queries concurrently and return de-duplicated results.

    Concurrency here is plain threads - it is *inside* a single node. The
    graph-level parallelism (LangGraph fan-out) happens later, during slide
    generation.
    """
    collected: list[SearchResult] = []
    with ThreadPoolExecutor(max_workers=min(6, len(queries) or 1)) as pool:
        futures = {pool.submit(search, q, max_results): q for q in queries}
        for future in as_completed(futures):
            try:
                collected.extend(future.result())
            except Exception as exc:
                logger.warning("query %r raised: %s", futures[future], exc)

    seen: set[str] = set()
    unique: list[SearchResult] = []
    for item in collected:
        key = re.sub(r"[#?].*$", "", item.url.rstrip("/"))
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def format_for_llm(results: list[SearchResult], limit: int = 30) -> str:
    """Render results as a compact numbered block for a prompt."""
    lines = []
    for idx, item in enumerate(results[:limit], start=1):
        published = f" ({item.published})" if item.published else ""
        lines.append(
            f"[{idx}] {item.title}{published}\n"
            f"    URL: {item.url}\n"
            f"    {item.snippet.strip()[:400]}"
        )
    return "\n".join(lines) if lines else "(no search results available)"


def active_backend() -> str:
    return "tavily" if TAVILY_API_KEY else "duckduckgo"
