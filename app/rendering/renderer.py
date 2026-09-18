"""Slide JSON -> HTML -> PNG.

RENDERING IS DELIBERATELY SEPARATE FROM CONTENT GENERATION.
The graph produces slide *data*; this module turns data into pixels. Nothing
here calls an LLM, and nothing in the graph knows what a pixel is. That split is
what makes the text crisp: headlines are laid out by a browser, not drawn by an
image model.

    Slide JSON -> Jinja2 template -> HTML + CSS design system
                                      -> Playwright screenshot -> 1080x1350 PNG

A Pillow fallback exists so the demo still produces real PNGs on a machine where
no browser can be installed.
"""
from __future__ import annotations

import html
import logging
import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.config import settings
from app.rendering.design_system import (
    DesignSystem,
    pick_theme,
    sketch_font_face_css,
    title_size_for,
)

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"

_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(["html", "xml"]),
    trim_blocks=True,
    lstrip_blocks=True,
)


class RenderError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# HTML building
# ---------------------------------------------------------------------------
def _highlighted_title(title: str, highlight: str) -> str:
    """Escape the title, then wrap the highlight word in an accent span.

    Escaping first and injecting markup second is what keeps a model-written
    title with an ampersand or a '<' from breaking the page.
    """
    safe = html.escape(title or "")
    if not highlight:
        return safe

    safe_highlight = html.escape(highlight.strip())
    if not safe_highlight or safe_highlight.lower() not in safe.lower():
        return safe

    pattern = re.compile(re.escape(safe_highlight), re.IGNORECASE)
    return pattern.sub(lambda m: f'<span class="mark">{m.group(0)}</span>', safe, count=1)


def _visual_items(slide: dict) -> list[str]:
    """What the visual macro draws.

    Prefer the explicit `visual_elements` the writer produced; fall back to the
    bullets so a slide is never visually empty.
    """
    items = [str(i).strip() for i in slide.get("visual_elements", []) if str(i).strip()]
    if items:
        return items[:4]
    if slide.get("visual_type") in ("FLOW", "PROCESS", "ARCHITECTURE", "CARD_GRID", "TIMELINE"):
        return [str(b).strip() for b in slide.get("bullets", [])][:4]
    return []


def build_html(slide: dict, topic: str, slide_count: int, design: DesignSystem, handle: str = "") -> str:
    """Render one slide dict into a complete standalone HTML page."""
    is_hook = slide.get("slide_number") == 1
    template = _env.get_template("slide.html.j2")

    return template.render(
        slide=slide,
        css_variables=design.as_css_variables(),
        sketch=design.is_sketch,
        # The handwriting fonts are inlined only when the sketch style is on, so
        # a modern-style slide carries no extra bytes.
        font_face_css=sketch_font_face_css() if design.is_sketch else "",
        title_size=title_size_for(slide.get("title", ""), design, is_hook=is_hook),
        title_html=_highlighted_title(slide.get("title", ""), slide.get("highlight", "")),
        visual_items=_visual_items(slide),
        topic_tag=(topic or "")[:64],
        slide_count=slide_count,
        is_last=slide.get("slide_number") == slide_count,
        brand=design.brand,
        handle=handle or "Save this post",
    )


# ---------------------------------------------------------------------------
# Playwright backend (preferred - real browser layout, crisp text)
# ---------------------------------------------------------------------------
def _render_with_playwright(pages: list[tuple[str, Path]], design: DesignSystem) -> list[Path]:
    from playwright.sync_api import sync_playwright

    written: list[Path] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--font-render-hinting=none", "--force-color-profile=srgb"])
        try:
            page = browser.new_page(
                viewport={"width": design.canvas_width, "height": design.canvas_height},
                device_scale_factor=1,
            )
            for markup, path in pages:
                page.set_content(markup, wait_until="load")
                # Give webfonts a moment; fall back silently if offline.
                try:
                    page.wait_for_function("document.fonts.ready.then(() => true)", timeout=4000)
                except Exception:
                    logger.debug("webfont wait timed out - using local font stack")
                page.screenshot(path=str(path), type="png")
                written.append(path)
                logger.info("rendered %s", path.name)
        finally:
            browser.close()
    return written


def _playwright_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            browser.close()
        return True
    except Exception as exc:
        logger.warning("Playwright browser unavailable (%s); falling back to Pillow", exc)
        return False


# ---------------------------------------------------------------------------
# Pillow backend (fallback - no browser required)
# ---------------------------------------------------------------------------
def _font(size: int, bold: bool = False):
    from PIL import ImageFont

    candidates = (
        ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
         "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"]
        if bold else
        ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
         "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"]
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def _wrap(draw, text: str, font, max_width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _render_with_pillow(slides: list[dict], topic: str, out_dir: Path, design: DesignSystem) -> list[Path]:
    """A simplified but on-brand renderer using the same design tokens.

    Limitation worth knowing: this backend cannot reproduce the sketch style.
    The handwriting fonts are vendored as woff2, which Pillow cannot load, and
    the hand-drawn borders are CSS. A sketch deck rendered here keeps its
    palette and layout but comes out in the modern letterforms.
    """
    from PIL import Image, ImageDraw

    if design.is_sketch:
        logger.warning(
            "sketch style falls back to standard lettering in the Pillow backend "
            "(handwriting fonts are woff2, which Pillow cannot load). Install a "
            "browser with `playwright install chromium` for the hand-drawn look."
        )

    theme = design.theme
    width, height = design.canvas_width, design.canvas_height
    pad = design.padding
    written: list[Path] = []

    for slide in slides:
        image = Image.new("RGB", (width, height), theme.bg)
        draw = ImageDraw.Draw(image)

        # background grid
        for x in range(0, width, 60):
            draw.line([(x, 0), (x, height)], fill=theme.bg_alt, width=1)
        for y in range(0, height, 60):
            draw.line([(0, y), (width, y)], fill=theme.bg_alt, width=1)

        number = slide.get("slide_number", 1)
        draw.rounded_rectangle([pad, pad, pad + 96, pad + 72], radius=16, fill=theme.accent)
        draw.text((pad + 28, pad + 16), f"{number:02d}", font=_font(34, True), fill=theme.accent_ink)
        draw.text((pad + 130, pad + 26), topic[:40].upper(), font=_font(20, True), fill=theme.ink_muted)

        y = pad + 150
        title_font = _font(title_size_for(slide.get("title", ""), design, number == 1), True)
        for line in _wrap(draw, slide.get("title", ""), title_font, width - 2 * pad):
            draw.text((pad, y), line, font=title_font, fill=theme.ink)
            y += int(title_font.size * 1.12)

        y += 24
        draw.rounded_rectangle([pad, y, pad + 128, y + 8], radius=4, fill=theme.accent)
        y += 52

        if slide.get("subtitle"):
            sub_font = _font(design.subtitle_size)
            for line in _wrap(draw, slide["subtitle"], sub_font, width - 2 * pad):
                draw.text((pad, y), line, font=sub_font, fill=theme.ink_muted)
                y += int(sub_font.size * 1.34)
            y += 24

        if slide.get("body"):
            body_font = _font(design.body_size)
            for line in _wrap(draw, slide["body"], body_font, width - 2 * pad):
                draw.text((pad, y), line, font=body_font, fill=theme.ink_muted)
                y += int(body_font.size * 1.42)
            y += 20

        items = _visual_items(slide) or slide.get("bullets", [])
        item_font = _font(30, True)
        for item in items[:4]:
            box_top = y
            lines = _wrap(draw, str(item), item_font, width - 2 * pad - 80)
            box_height = 34 + len(lines) * int(item_font.size * 1.3)
            draw.rounded_rectangle(
                [pad, box_top, width - pad, box_top + box_height],
                radius=20, fill=theme.bg_alt, outline=theme.accent, width=2,
            )
            ty = box_top + 17
            for line in lines:
                draw.text((pad + 32, ty), line, font=item_font, fill=theme.ink)
                ty += int(item_font.size * 1.3)
            y = box_top + box_height + 18
            if y > height - 220:
                break

        # footer with progress dots
        footer_y = height - pad - 20
        draw.line([(pad, footer_y - 34), (width - pad, footer_y - 34)], fill=theme.bg_alt, width=2)
        draw.text((pad, footer_y - 14), design.brand, font=_font(20, True), fill=theme.ink_muted)
        dot_x = width // 2 - (len(slides) * 18) // 2
        for index in range(1, len(slides) + 1):
            on = index == number
            draw.rounded_rectangle(
                [dot_x, footer_y - 6, dot_x + (26 if on else 10), footer_y + 4],
                radius=5, fill=theme.accent if on else theme.bg_alt,
            )
            dot_x += (36 if on else 20)

        path = out_dir / f"slide_{number:02d}.png"
        image.save(path, "PNG")
        written.append(path)
        logger.info("rendered %s (pillow fallback)", path.name)

    return written


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def render_carousel(
    slides: list[dict],
    topic: str,
    out_dir: Path,
    theme_name: str | None = None,
    handle: str = "",
    save_html: bool = True,
    style: str | None = None,
) -> tuple[list[Path], str, str]:
    """Render every slide to a PNG.

    `style` selects the visual language - "modern" (clean editorial) or
    "sketch" (hand-drawn whiteboard). Theme selects the palette within it.

    Returns (image_paths, theme_used, backend_used).
    All five slides share one DesignSystem instance - that shared object is the
    reason the deck looks like one carousel.
    """
    if not slides:
        raise RenderError("No slides to render.")

    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(slides, key=lambda s: s.get("slide_number", 0))

    style = style or settings.slide_style
    theme = pick_theme(topic, theme_name, style=style)
    design = DesignSystem().configure(theme=theme, style=style)
    # configure() may reject a theme that does not belong to the chosen style.
    theme = design.theme_name

    backend = settings.renderer
    if backend == "auto":
        backend = "playwright" if _playwright_available() else "pillow"

    if backend == "playwright":
        pages: list[tuple[str, Path]] = []
        for slide in ordered:
            markup = build_html(slide, topic, len(ordered), design, handle=handle)
            if save_html:
                (out_dir / f"slide_{slide['slide_number']:02d}.html").write_text(markup, encoding="utf-8")
            pages.append((markup, out_dir / f"slide_{slide['slide_number']:02d}.png"))
        try:
            paths = _render_with_playwright(pages, design)
            return paths, theme, "playwright"
        except Exception as exc:
            logger.warning("Playwright render failed (%s); falling back to Pillow", exc)

    paths = _render_with_pillow(ordered, topic, out_dir, design)
    return paths, theme, "pillow"
