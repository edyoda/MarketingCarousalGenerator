"""Tests for tracing wiring. No API keys, no network, no accounts."""
from __future__ import annotations

import logging

import pytest

from app.observability import (
    get_callbacks,
    langsmith_enabled,
    run_config,
    tracing_status,
)

TRACE_VARS = (
    "LANGSMITH_TRACING",
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_PROJECT",
    "LANGCHAIN_PROJECT",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
)


@pytest.fixture
def no_tracing(monkeypatch):
    """A clean environment: nothing configured."""
    for var in TRACE_VARS:
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# defaults - tracing must be entirely optional
# ---------------------------------------------------------------------------
def test_nothing_configured_is_silent(no_tracing):
    assert langsmith_enabled() is False
    assert get_callbacks() == []
    assert tracing_status() == "local logs only"


def test_run_config_omits_callbacks_when_none(no_tracing):
    """An empty callbacks list must not be attached - it only adds noise."""
    config = run_config("carousel-1", {"domain": "AI"})
    assert "callbacks" not in config


# ---------------------------------------------------------------------------
# LangSmith - env vars only, no callback object
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("var", ["LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"])
def test_either_langsmith_env_var_enables_it(no_tracing, monkeypatch, var):
    """Both the current and the legacy variable are honoured."""
    monkeypatch.setenv(var, "true")
    assert langsmith_enabled() is True
    assert "LangSmith" in tracing_status()


def test_langsmith_value_must_be_true(no_tracing, monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    assert langsmith_enabled() is False


def test_langsmith_status_names_the_project(no_tracing, monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_PROJECT", "carousel-class")
    assert "carousel-class" in tracing_status()


# ---------------------------------------------------------------------------
# Langfuse - needs a callback handler, and needs its packages
# ---------------------------------------------------------------------------
def test_langfuse_keys_without_the_package_warn_and_degrade(
    no_tracing, monkeypatch, caplog
):
    """Half-configured tracing must say so, not fail silently or crash the run."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")

    real_import = __import__

    def no_langfuse(name, *args, **kwargs):
        if name.startswith("langfuse"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", no_langfuse)

    with caplog.at_level(logging.WARNING):
        callbacks = get_callbacks()

    assert callbacks == []
    assert "pip install langfuse" in caplog.text


def test_partial_langfuse_credentials_are_ignored(no_tracing, monkeypatch):
    """One key alone is a misconfiguration, not a reason to try connecting."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    assert get_callbacks() == []


# ---------------------------------------------------------------------------
# run_config - what makes a hosted trace searchable
# ---------------------------------------------------------------------------
def test_run_config_carries_the_thread_id(no_tracing):
    config = run_config("carousel-123", {"domain": "AI"})
    assert config["configurable"]["thread_id"] == "carousel-123"


def test_run_config_tags_and_metadata_describe_the_run(no_tracing):
    """Without tags a hosted trace is a wall of identical-looking runs."""
    state = {
        "domain": "AI",
        "audience": "Software Developers",
        "platform": "LinkedIn",
        "slide_count": 5,
        "slide_style": "sketch",
    }
    config = run_config("carousel-9", state)

    assert config["run_name"] == "carousel:AI"
    assert "domain:AI" in config["tags"]
    assert "platform:LinkedIn" in config["tags"]
    assert "style:sketch" in config["tags"]

    assert config["metadata"]["audience"] == "Software Developers"
    assert config["metadata"]["slide_count"] == 5
    assert config["metadata"]["thread_id"] == "carousel-9"


def test_run_config_tolerates_an_empty_state(no_tracing):
    """Resuming a run may not have the original inputs to hand."""
    config = run_config("carousel-9")
    assert config["configurable"]["thread_id"] == "carousel-9"
    assert "style:modern" in config["tags"]


def test_recursion_limit_is_preserved(no_tracing):
    assert run_config("t", {}, recursion_limit=99)["recursion_limit"] == 99


# ---------------------------------------------------------------------------
# the regression this whole module exists to prevent
# ---------------------------------------------------------------------------
def test_callbacks_actually_reach_the_graph(no_tracing):
    """get_callbacks() used to be dead code: defined, never passed to a run.

    This builds a throwaway graph and asserts a handler attached through
    run_config sees the node events.
    """
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph

    from app.graph.state import CarouselState

    seen: list[str] = []

    class Spy(BaseCallbackHandler):
        def on_chain_start(self, serialized, inputs, **kwargs):
            name = kwargs.get("name") or (serialized or {}).get("name")
            if name:
                seen.append(name)

    graph = StateGraph(CarouselState)
    graph.add_node("only_node", lambda state: {"status": "done"})
    graph.add_edge(START, "only_node")
    graph.add_edge("only_node", END)
    app = graph.compile(checkpointer=MemorySaver())

    config = run_config("trace-test", {"domain": "AI"})
    config["callbacks"] = [Spy()]

    app.invoke({"domain": "AI", "audience": "D", "platform": "LinkedIn", "slide_count": 5}, config)

    assert "only_node" in seen
    assert "carousel:AI" in seen
