"""The visual design system.

One source of truth for canvas size, type scale, spacing and colour. Every slide
template reads from here, which is what makes five separately-generated slides
look like one carousel instead of five unrelated posts.

Changing a value here restyles the whole deck.
"""
from __future__ import annotations

import base64
import functools
from dataclasses import dataclass
from pathlib import Path

FONT_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "fonts"

CANVAS_WIDTH = 1080
CANVAS_HEIGHT = 1350


@dataclass(frozen=True)
class Theme:
    """A colour scheme. All five slides in a run share exactly one."""

    name: str
    bg: str                 # page background
    bg_alt: str             # raised surfaces / cards
    surface_border: str
    ink: str                # primary text
    ink_muted: str          # secondary text
    accent: str             # the one loud colour
    accent_soft: str        # accent at low opacity, for fills
    accent_ink: str         # text drawn on top of accent
    grid: str               # background grid lines
    glow: str               # radial background wash


THEMES: dict[str, Theme] = {
    "midnight": Theme(
        name="midnight",
        bg="#0B1020",
        bg_alt="#141B31",
        surface_border="rgba(255,255,255,0.10)",
        ink="#F5F7FF",
        ink_muted="#9AA6C7",
        accent="#5B8CFF",
        accent_soft="rgba(91,140,255,0.16)",
        accent_ink="#0B1020",
        grid="rgba(255,255,255,0.045)",
        glow="rgba(91,140,255,0.22)",
    ),
    "carbon": Theme(
        name="carbon",
        bg="#0D0D0F",
        bg_alt="#17171C",
        surface_border="rgba(255,255,255,0.10)",
        ink="#FAFAFA",
        ink_muted="#9B9BA5",
        accent="#F97316",
        accent_soft="rgba(249,115,22,0.16)",
        accent_ink="#0D0D0F",
        grid="rgba(255,255,255,0.04)",
        glow="rgba(249,115,22,0.20)",
    ),
    "forest": Theme(
        name="forest",
        bg="#07130F",
        bg_alt="#0F2119",
        surface_border="rgba(255,255,255,0.10)",
        ink="#F2FBF6",
        ink_muted="#8FB3A3",
        accent="#34D399",
        accent_soft="rgba(52,211,153,0.16)",
        accent_ink="#07130F",
        grid="rgba(255,255,255,0.04)",
        glow="rgba(52,211,153,0.20)",
    ),
    "daylight": Theme(
        name="daylight",
        bg="#F7F8FC",
        bg_alt="#FFFFFF",
        surface_border="rgba(15,23,42,0.10)",
        ink="#0F172A",
        ink_muted="#5A6784",
        accent="#2563EB",
        accent_soft="rgba(37,99,235,0.10)",
        accent_ink="#FFFFFF",
        grid="rgba(15,23,42,0.045)",
        glow="rgba(37,99,235,0.14)",
    ),

    # --- themes designed for the hand-drawn "sketch" style ----------------
    "paper": Theme(
        name="paper",
        bg="#FBF7EC",
        bg_alt="#FFFDF7",
        surface_border="#2B2A26",
        ink="#23221E",
        ink_muted="#6B675C",
        accent="#E8590C",
        accent_soft="rgba(232,89,12,0.16)",
        accent_ink="#FFFDF7",
        grid="rgba(43,42,38,0.07)",
        glow="rgba(232,89,12,0.10)",
    ),
    "blueprint": Theme(
        name="blueprint",
        bg="#10273D",
        bg_alt="#16334C",
        surface_border="#BBD8F0",
        ink="#EAF3FB",
        ink_muted="#9FBBD4",
        accent="#7FD1FF",
        accent_soft="rgba(127,209,255,0.16)",
        accent_ink="#10273D",
        grid="rgba(187,216,240,0.12)",
        glow="rgba(127,209,255,0.14)",
    ),
}

DEFAULT_THEME = "midnight"
DEFAULT_SKETCH_THEME = "paper"

# Which themes suit which style. A hand-drawn deck on a near-black grid looks
# wrong; these two lists keep theme choice inside the style's own palette.
MODERN_THEMES = ("midnight", "carbon", "forest", "daylight")
SKETCH_THEMES = ("paper", "blueprint")

STYLES = ("modern", "sketch")


@dataclass(frozen=True)
class DesignSystem:
    """Canvas, type scale and spacing - the non-colour half of the system."""

    canvas_width: int = CANVAS_WIDTH
    canvas_height: int = CANVAS_HEIGHT

    # A real font stack: Inter if the renderer can reach Google Fonts, otherwise
    # the best sans available on a typical Linux box. Never fall through to a
    # serif default, which would break the design.
    font_family: str = (
        "'Inter', 'Noto Sans', 'DejaVu Sans', 'Liberation Sans', "
        "-apple-system, 'Segoe UI', Roboto, sans-serif"
    )
    mono_family: str = "'JetBrains Mono', 'DejaVu Sans Mono', 'Liberation Mono', monospace"

    # Type scale. Deliberately large - this is read on a phone at thumbnail size.
    display_size: int = 96      # slide 1 hook
    heading_size: int = 72      # normal slide titles
    heading_small: int = 56     # long titles
    subtitle_size: int = 36
    body_size: int = 30
    label_size: int = 22
    caption_size: int = 20
    stat_size: int = 200        # the giant number on a STATISTIC slide

    line_height_tight: float = 1.06
    line_height_body: float = 1.45
    letter_spacing_display: str = "-0.03em"

    # Spacing scale (px)
    padding: int = 88
    gap_sm: int = 16
    gap_md: int = 28
    gap_lg: int = 48
    border_radius: int = 28
    border_radius_sm: int = 16

    # Branding
    brand: str = "Built with LangGraph"

    theme_name: str = DEFAULT_THEME

    # "modern" = the clean editorial look; "sketch" = hand-drawn whiteboard.
    # Style changes fonts, borders and shapes; theme only changes colour.
    style: str = "modern"

    # Handwriting stacks, used only when style == "sketch". The fonts are
    # vendored in assets/fonts and embedded directly into the HTML, so the
    # sketch style renders identically with no network access.
    hand_display_family: str = "'Caveat', 'Comic Sans MS', cursive"
    hand_body_family: str = "'Kalam', 'Comic Sans MS', cursive"

    @property
    def theme(self) -> Theme:
        default = DEFAULT_SKETCH_THEME if self.style == "sketch" else DEFAULT_THEME
        return THEMES.get(self.theme_name, THEMES[default])

    @property
    def is_sketch(self) -> bool:
        return self.style == "sketch"

    def configure(self, theme: str | None = None, style: str | None = None) -> "DesignSystem":
        """Return a copy with a different theme and/or style."""
        new_style = style if style in STYLES else self.style
        allowed = SKETCH_THEMES if new_style == "sketch" else MODERN_THEMES
        fallback = DEFAULT_SKETCH_THEME if new_style == "sketch" else DEFAULT_THEME
        new_theme = theme if theme in allowed else (self.theme_name if self.theme_name in allowed else fallback)
        return DesignSystem(theme_name=new_theme, style=new_style)

    def with_theme(self, name: str) -> "DesignSystem":
        """Backwards-compatible helper: change theme, keep the current style."""
        return self.configure(theme=name)

    def as_css_variables(self) -> str:
        """Emit the system as CSS custom properties consumed by the templates."""
        t = self.theme
        return f"""
      --canvas-w: {self.canvas_width}px;
      --canvas-h: {self.canvas_height}px;
      --font: {self.hand_body_family if self.is_sketch else self.font_family};
      --display-font: {self.hand_display_family if self.is_sketch else self.font_family};
      --mono: {self.hand_body_family if self.is_sketch else self.mono_family};

      --bg: {t.bg};
      --bg-alt: {t.bg_alt};
      --border: {t.surface_border};
      --ink: {t.ink};
      --ink-muted: {t.ink_muted};
      --accent: {t.accent};
      --accent-soft: {t.accent_soft};
      --accent-ink: {t.accent_ink};
      --grid: {t.grid};
      --glow: {t.glow};

      --display: {self.display_size}px;
      --heading: {self.heading_size}px;
      --heading-sm: {self.heading_small}px;
      --subtitle: {self.subtitle_size}px;
      --body: {self.body_size}px;
      --label: {self.label_size}px;
      --caption: {self.caption_size}px;
      --stat: {self.stat_size}px;

      --lh-tight: {self.line_height_tight};
      --lh-body: {self.line_height_body};
      --ls-display: {self.letter_spacing_display};

      --pad: {self.padding}px;
      --gap-sm: {self.gap_sm}px;
      --gap-md: {self.gap_md}px;
      --gap-lg: {self.gap_lg}px;
      --radius: {self.border_radius}px;
      --radius-sm: {self.border_radius_sm}px;
"""


@functools.lru_cache(maxsize=8)
def _font_data_uri(filename: str) -> str | None:
    """Base64-encode a vendored font file as a data: URI.

    Embedding rather than linking means a rendered slide is self-contained: the
    sketch style looks the same offline, and the saved .html files stay portable.
    """
    path = FONT_DIR / filename
    if not path.exists():
        return None
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:font/woff2;base64,{encoded}"


# (family, weight, file) for the handwriting faces used by the sketch style.
_SKETCH_FACES = (
    ("Caveat", 700, "Caveat-700.woff2"),
    ("Kalam", 400, "Kalam-400.woff2"),
    ("Kalam", 700, "Kalam-700.woff2"),
)


def sketch_font_face_css() -> str:
    """@font-face rules for the handwriting fonts, with the bytes inlined.

    Returns an empty string if the fonts were never vendored - the stacks fall
    back to a system cursive face rather than breaking the render.
    """
    rules = []
    for family, weight, filename in _SKETCH_FACES:
        uri = _font_data_uri(filename)
        if not uri:
            continue
        rules.append(
            f"@font-face {{\n"
            f"  font-family: '{family}';\n"
            f"  font-style: normal;\n"
            f"  font-weight: {weight};\n"
            f"  font-display: block;\n"
            f"  src: url({uri}) format('woff2');\n"
            f"}}"
        )
    return "\n".join(rules)


def fonts_available() -> bool:
    """True when the handwriting fonts are vendored locally."""
    return all((FONT_DIR / filename).exists() for _, _, filename in _SKETCH_FACES)


# The dict form asked for in the brief - handy for teaching and for tests.
DESIGN_SYSTEM = {
    "canvas": f"{CANVAS_WIDTH}x{CANVAS_HEIGHT}",
    "font_family": DesignSystem().font_family,
    "heading_size": DesignSystem().heading_size,
    "body_size": DesignSystem().body_size,
    "spacing": DesignSystem().padding,
    "border_radius": DesignSystem().border_radius,
    "layout": "full-bleed vertical: meta bar / headline block / visual zone / footer",
    "themes": list(THEMES),
}


def pick_theme(topic: str, override: str | None = None, style: str = "modern") -> str:
    """Choose a theme deterministically from the topic string.

    Deterministic rather than random so that re-rendering the same carousel
    produces identical images - which matters for a resumable graph. The pool is
    restricted to themes that suit the requested style.
    """
    pool = SKETCH_THEMES if style == "sketch" else MODERN_THEMES
    if override and override in pool:
        return override
    if not topic:
        return pool[0]
    return pool[sum(ord(c) for c in topic) % len(pool)]


def title_size_for(title: str, ds: DesignSystem, is_hook: bool = False) -> int:
    """Shrink the headline when the copy is long, so it never overflows.

    The slide prompts cap title length, but a model can still exceed it; this is
    the renderer's own safety net.

    The sketch style needs its own calibration: Caveat is a narrower face with a
    smaller x-height than Inter, so at the same pixel size it *reads* smaller and
    fits more characters per line. It therefore starts larger and shrinks later.
    """
    length = len(title or "")
    base = ds.display_size if is_hook else ds.heading_size

    if ds.is_sketch:
        base = int(base * 1.25)
        long, medium, short = 96, 72, 54
    else:
        long, medium, short = 78, 58, 42

    if length > long:
        return int(base * 0.62)
    if length > medium:
        return int(base * 0.74)
    if length > short:
        return int(base * 0.86)
    return base
