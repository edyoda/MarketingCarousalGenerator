"""Central configuration.

Everything that varies between environments (API keys, model names, limits)
lives here so that nodes stay focused on graph logic.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", ROOT_DIR / "output"))
LOG_DIR = Path(os.getenv("LOG_DIR", ROOT_DIR / "logs"))
CHECKPOINT_DB = Path(os.getenv("CHECKPOINT_DB", ROOT_DIR / "checkpoints.sqlite"))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _first_env(*names: str) -> str | None:
    """Return the first environment variable that is actually set.

    Some machines export the Anthropic key under a non-standard name, so we
    accept a couple of spellings rather than failing with 'no key found'.
    """
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


ANTHROPIC_API_KEY = _first_env("ANTHROPIC_API_KEY", "ANTHROPIC_KEY")
OPENAI_API_KEY = _first_env("OPENAI_API_KEY")
TAVILY_API_KEY = _first_env("TAVILY_API_KEY")


def detect_provider() -> str:
    """Pick an LLM provider from whatever credentials exist.

    Explicit LLM_PROVIDER always wins; otherwise Anthropic is preferred and
    OpenAI is the fallback.
    """
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit in {"anthropic", "openai"}:
        return explicit
    if ANTHROPIC_API_KEY:
        return "anthropic"
    if OPENAI_API_KEY:
        return "openai"
    return "anthropic"  # will raise a clear error at call time


@dataclass
class Settings:
    """Runtime knobs for the whole application.

    Deliberately NOT frozen: the CLI flags and the UI checkboxes flip the
    human-in-the-loop gates at runtime (`settings.require_topic_approval = True`).
    """

    provider: str = field(default_factory=detect_provider)

    # Two tiers: a strong model for judgement-heavy nodes (analysis, critique)
    # and a fast model for the many small parallel slide-writing calls.
    anthropic_smart_model: str = os.getenv("ANTHROPIC_SMART_MODEL", "claude-opus-5")
    anthropic_fast_model: str = os.getenv("ANTHROPIC_FAST_MODEL", "claude-sonnet-5")
    openai_smart_model: str = os.getenv("OPENAI_SMART_MODEL", "gpt-4o")
    openai_fast_model: str = os.getenv("OPENAI_FAST_MODEL", "gpt-4o-mini")

    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.4"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "8000"))

    # Graph behaviour
    max_revisions: int = int(os.getenv("MAX_REVISIONS", "3"))
    approval_threshold: int = int(os.getenv("APPROVAL_THRESHOLD", "8"))
    search_results_per_query: int = int(os.getenv("SEARCH_RESULTS_PER_QUERY", "6"))

    # Human-in-the-loop toggles (LangGraph interrupts)
    require_topic_approval: bool = os.getenv("REQUIRE_TOPIC_APPROVAL", "false").lower() == "true"
    require_image_approval: bool = os.getenv("REQUIRE_IMAGE_APPROVAL", "false").lower() == "true"

    # Rendering
    canvas_width: int = int(os.getenv("CANVAS_WIDTH", "1080"))
    canvas_height: int = int(os.getenv("CANVAS_HEIGHT", "1350"))
    renderer: str = os.getenv("RENDERER", "auto")  # auto | playwright | pillow
    slide_style: str = os.getenv("SLIDE_STYLE", "modern")  # modern | sketch

    output_dir: Path = OUTPUT_DIR
    checkpoint_db: Path = CHECKPOINT_DB

    @property
    def smart_model(self) -> str:
        return self.anthropic_smart_model if self.provider == "anthropic" else self.openai_smart_model

    @property
    def fast_model(self) -> str:
        return self.anthropic_fast_model if self.provider == "anthropic" else self.openai_fast_model


settings = Settings()
