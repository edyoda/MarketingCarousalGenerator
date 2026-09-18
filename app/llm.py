"""LLM factory.

One place that knows how to build a chat model, so switching provider is an
env-var change rather than an edit across every node.
"""
from __future__ import annotations

import functools

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import ANTHROPIC_API_KEY, OPENAI_API_KEY, settings


class MissingCredentialsError(RuntimeError):
    """Raised with an actionable message instead of a deep SDK stack trace."""


# Current-generation Claude models removed the sampling parameters: sending
# `temperature` to them returns a 400 ("`temperature` is deprecated for this
# model"). Older Claude models still accept it, so we check rather than assume.
_NO_TEMPERATURE_MARKERS = (
    "opus-5",
    "sonnet-5",
    "fable-5",
    "mythos-5",
    "opus-4-6",
    "opus-4-7",
    "opus-4-8",
    "sonnet-4-6",
)


def _accepts_temperature(model: str) -> bool:
    return not any(marker in model for marker in _NO_TEMPERATURE_MARKERS)


@functools.lru_cache(maxsize=8)
def get_llm(tier: str = "smart", temperature: float | None = None) -> BaseChatModel:
    """Return a chat model.

    tier="smart" -> judgement-heavy nodes (trend analysis, strategy, critique)
    tier="fast"  -> the parallel per-slide writers, where throughput matters
    """
    model = settings.smart_model if tier == "smart" else settings.fast_model
    temp = settings.temperature if temperature is None else temperature

    if settings.provider == "anthropic":
        if not ANTHROPIC_API_KEY:
            raise MissingCredentialsError(
                "No Anthropic key found. Set ANTHROPIC_API_KEY in .env, "
                "or set LLM_PROVIDER=openai to use OpenAI instead."
            )
        from langchain_anthropic import ChatAnthropic

        kwargs: dict = {
            "model": model,
            "api_key": ANTHROPIC_API_KEY,
            "max_tokens": settings.max_tokens,
            "timeout": 180,
            "max_retries": 3,
        }
        if _accepts_temperature(model):
            kwargs["temperature"] = temp
        return ChatAnthropic(**kwargs)

    if not OPENAI_API_KEY:
        raise MissingCredentialsError(
            "No OpenAI key found. Set OPENAI_API_KEY in .env, "
            "or set LLM_PROVIDER=anthropic to use Claude instead."
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=OPENAI_API_KEY,
        temperature=temp,
        max_tokens=settings.max_tokens,
        timeout=180,
        max_retries=3,
    )


def structured(schema, tier: str = "smart", temperature: float | None = None):
    """Bind a Pydantic schema to a model.

    Every LLM node in this project goes through here: LangGraph state stays
    typed because the model is forced to answer in a known shape.
    """
    return get_llm(tier=tier, temperature=temperature).with_structured_output(schema)


def describe() -> str:
    return f"{settings.provider}:{settings.smart_model} (fast tier: {settings.fast_model})"
