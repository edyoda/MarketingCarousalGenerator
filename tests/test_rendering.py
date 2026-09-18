"""Rendering tests - design system, templates and the PNG pipeline."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.rendering.design_system import (
    DESIGN_SYSTEM,
    THEMES,
    DesignSystem,
    pick_theme,
    title_size_for,
)
from app.rendering.renderer import _highlighted_title, _visual_items, build_html


def test_design_system_canvas_matches_the_brief():
    assert DESIGN_SYSTEM["canvas"] == "1080x1350"


def test_every_theme_defines_every_token():
    for name, theme in THEMES.items():
        for field in ("bg", "ink", "accent", "accent_ink", "ink_muted"):
            assert getattr(theme, field), f"{name} is missing {field}"


def test_theme_choice_is_deterministic():
    """Re-rendering a resumed run must produce identical images."""
    assert pick_theme("MCP and AI agents") == pick_theme("MCP and AI agents")
    assert pick_theme("anything", override="carbon") == "carbon"
    assert pick_theme("anything", override="not-a-theme") in THEMES


def test_long_titles_shrink_so_they_cannot_overflow():
    ds = DesignSystem()
    short = title_size_for("Short title", ds)
    long = title_size_for("A very long title that runs well past the intended limit", ds)
    assert long < short


def test_highlight_is_injected_after_escaping():
    """A title with markup characters must not be able to break the page."""
    out = _highlighted_title("Tools & <agents> need MCP", "MCP")
    assert "&amp;" in out and "&lt;agents&gt;" in out
    assert '<span class="mark">MCP</span>' in out


def test_highlight_absent_from_title_is_ignored():
    assert _highlighted_title("No match here", "MCP") == "No match here"


def test_visual_items_fall_back_to_bullets_for_structural_visuals():
    slide = {"visual_type": "FLOW", "visual_elements": [], "bullets": ["A", "B"]}
    assert _visual_items(slide) == ["A", "B"]

    slide = {"visual_type": "HOOK", "visual_elements": [], "bullets": ["A", "B"]}
    assert _visual_items(slide) == []


@pytest.mark.parametrize(
    "visual_type",
    ["HOOK", "STATISTIC", "COMPARISON", "PROCESS", "ARCHITECTURE", "FLOW",
     "TIMELINE", "CARD_GRID", "BEFORE_AFTER", "DIAGRAM", "QUOTE", "CTA"],
)
def test_every_visual_type_renders_without_error(visual_type):
    """Each declared VisualType must have a working template."""
    slide = {
        "slide_number": 2,
        "title": "A title",
        "subtitle": "A subtitle",
        "body": "Some body copy.",
        "bullets": ["one", "two"],
        "visual_type": visual_type,
        "highlight": "42%",
        "visual_elements": ["First: a", "Second: b", "Third: c"],
        "cta": "Follow for more",
    }
    markup = build_html(slide, "A topic", 5, DesignSystem())
    assert "<html" in markup and "A title" in markup


def test_statistic_caption_never_repeats_the_number():
    """Regression: the deck used to read '42%' above a caption of '2026: 42%'."""
    slide = {
        "slide_number": 4, "title": "t", "subtitle": "", "body": "", "bullets": [],
        "visual_type": "STATISTIC", "highlight": "42%",
        "visual_elements": ["2026: 42%", "2025: 12%"], "cta": "",
    }
    markup = build_html(slide, "topic", 5, DesignSystem())
    assert "2025: 12%" in markup
    assert "2026: 42%" not in markup


def test_renders_real_pngs(tmp_path: Path):
    """End-to-end render. Works with either backend."""
    from PIL import Image

    from app.rendering.renderer import render_carousel

    slides = [
        {
            "slide_number": n, "title": f"Slide {n}", "subtitle": "sub", "body": "body",
            "bullets": ["a", "b"], "visual_type": "DIAGRAM", "highlight": "",
            "visual_elements": ["x", "y"], "cta": "", "source_references": [],
        }
        for n in range(1, 4)
    ]

    paths, theme, backend = render_carousel(slides, "Test topic", tmp_path, theme_name="midnight")

    assert len(paths) == 3
    assert theme == "midnight"
    assert backend in ("playwright", "pillow")
    for path in paths:
        assert path.exists() and path.stat().st_size > 5_000
        with Image.open(path) as image:
            assert image.size == (1080, 1350)


# ---------------------------------------------------------------------------
# sketch style
# ---------------------------------------------------------------------------
def test_style_and_theme_are_independent_axes():
    from app.rendering.design_system import DEFAULT_SKETCH_THEME

    modern = DesignSystem()
    sketch = modern.configure(style="sketch")

    assert modern.style == "modern" and not modern.is_sketch
    assert sketch.is_sketch and sketch.theme_name == DEFAULT_SKETCH_THEME


def test_a_style_only_accepts_its_own_themes():
    """A hand-drawn deck on the near-black modern grid looks wrong, so the
    design system refuses the combination rather than rendering it."""
    ds = DesignSystem()

    assert ds.configure(theme="midnight", style="sketch").theme_name == "paper"
    assert ds.configure(theme="blueprint", style="sketch").theme_name == "blueprint"
    assert ds.configure(theme="paper", style="modern").theme_name == "midnight"
    assert ds.configure(theme="carbon", style="modern").theme_name == "carbon"


def test_theme_choice_stays_inside_the_style_pool():
    from app.rendering.design_system import MODERN_THEMES, SKETCH_THEMES

    for topic in ("MCP agents", "code review", "", "vector search"):
        assert pick_theme(topic, style="sketch") in SKETCH_THEMES
        assert pick_theme(topic, style="modern") in MODERN_THEMES


def test_handwriting_fonts_are_vendored_and_embeddable():
    """The sketch style must not depend on network access at render time."""
    from app.rendering.design_system import fonts_available, sketch_font_face_css

    assert fonts_available(), "run the font vendoring step - assets/fonts is empty"

    css = sketch_font_face_css()
    assert css.count("@font-face") == 3
    assert "data:font/woff2;base64," in css
    assert "https://" not in css  # fully self-contained


def _slide_classes(markup: str) -> str:
    """The class list on the slide element itself.

    The sketch CSS rules are always present in the stylesheet - they are inert
    until the `slide--sketch` class switches them on - so a test has to look at
    the element, not at the whole document.
    """
    match = re.search(r'<div class="(slide[^"]*)"', markup)
    assert match, "no slide element found"
    return match.group(1)


def test_fonts_are_only_embedded_for_sketch():
    """A modern slide should not carry ~130KB of handwriting fonts."""
    slide = {
        "slide_number": 1, "title": "T", "subtitle": "", "body": "", "bullets": [],
        "visual_type": "FLOW", "highlight": "", "visual_elements": ["A", "B"], "cta": "",
    }
    modern = build_html(slide, "topic", 5, DesignSystem())
    sketch = build_html(slide, "topic", 5, DesignSystem().configure(style="sketch"))

    assert "@font-face" not in modern
    assert "slide--sketch" not in _slide_classes(modern)

    assert "@font-face" in sketch
    assert "slide--sketch" in _slide_classes(sketch)
    # the embedded fonts are the bulk of the difference
    assert len(sketch) - len(modern) > 100_000


def test_sketch_titles_are_scaled_up_for_the_narrower_face():
    """Caveat reads smaller than Inter at the same pixel size."""
    modern = DesignSystem()
    sketch = modern.configure(style="sketch")

    for title in ("Short", "A medium length headline here", "x" * 95):
        assert title_size_for(title, sketch) > title_size_for(title, modern)


def test_sketch_renders_real_pngs(tmp_path: Path):
    from PIL import Image

    from app.rendering.renderer import render_carousel

    slides = [
        {
            "slide_number": n, "title": f"Slide {n}", "subtitle": "sub", "body": "",
            "bullets": [], "visual_type": "PROCESS", "highlight": "",
            "visual_elements": ["one", "two", "three"], "cta": "", "source_references": [],
        }
        for n in range(1, 4)
    ]

    paths, theme, backend = render_carousel(
        slides, "Sketch topic", tmp_path, theme_name="paper", style="sketch"
    )

    assert theme == "paper"
    assert len(paths) == 3
    for path in paths:
        with Image.open(path) as image:
            assert image.size == (1080, 1350)


def test_sketch_html_has_no_external_references():
    """The sketch style must render identically with no network at all.

    Regression guard: the base template used to emit a Google Fonts <link> for
    Inter on every slide, including sketch slides that never use it.
    """
    slide = {
        "slide_number": 1, "title": "T", "subtitle": "", "body": "", "bullets": [],
        "visual_type": "FLOW", "highlight": "", "visual_elements": ["A", "B"], "cta": "",
    }
    sketch = build_html(slide, "topic", 5, DesignSystem().configure(style="sketch"))

    assert "http://" not in sketch
    assert "https://" not in sketch

    # the modern style is still allowed to fetch Inter
    modern = build_html(slide, "topic", 5, DesignSystem())
    assert "fonts.googleapis.com" in modern
