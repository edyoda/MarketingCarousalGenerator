"""Streamlit UI for the trend-to-carousel agent.

    streamlit run streamlit_app.py

Shows the four inputs, a live progress checklist driven by the graph's node
events, the human-approval gates (LangGraph interrupts), and the finished
carousel with a download.
"""
from __future__ import annotations

import io
import logging
import zipfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from langgraph.types import Command

from app.config import settings
from app.graph.graph import get_app
from app.graph.state import initial_state
from app.llm import describe
from app.rendering.design_system import MODERN_THEMES, SKETCH_THEMES
from app.observability import NODE_LABELS, configure_logging, run_config, tracing_status
from app.tools.web_search import active_backend

configure_logging(logging.INFO)

st.set_page_config(page_title="Trend to Carousel", page_icon="🎠", layout="wide")

# The order the progress checklist is displayed in. Parallel and loop nodes are
# folded into single lines so the list reads like a plan, not a trace.
PROGRESS_STEPS = [
    "trend_research",
    "trend_analyzer",
    "topic_selection",
    "research_topic",
    "content_strategy",
    "carousel_planner",
    "generate_slide",
    "carousel_critic",
    "rewrite_slide",
    "render_images",
    "finalize",
]

RECURSION_LIMIT = 60


# ---------------------------------------------------------------------------
# session helpers
# ---------------------------------------------------------------------------
def _init_session() -> None:
    st.session_state.setdefault("thread_id", f"carousel-{datetime.now():%Y%m%d-%H%M%S}")
    st.session_state.setdefault("completed", [])
    st.session_state.setdefault("pending_interrupt", None)
    st.session_state.setdefault("final_state", None)
    st.session_state.setdefault("running", False)
    st.session_state.setdefault("log", [])


def _config() -> dict:
    """Run config, including any tracer callbacks and the run's tags."""
    return run_config(
        st.session_state.thread_id,
        st.session_state.get("run_meta", {}),
        recursion_limit=RECURSION_LIMIT,
    )


def _render_progress(container) -> None:
    """Draw the checklist of completed steps."""
    done = st.session_state.completed
    lines = []
    for node in PROGRESS_STEPS:
        label = NODE_LABELS.get(node, node)
        if node in done:
            lines.append(f"✅ {label}")
        elif node == "rewrite_slide":
            continue  # only shown once it actually happens
        else:
            lines.append(f"⬜ {label}")
    container.markdown("\n\n".join(lines))


def _drive(payload, progress_box, status_box) -> None:
    """Stream the graph forward until it finishes or hits an interrupt."""
    app = get_app()
    config = _config()

    for chunk in app.stream(payload, config=config, stream_mode="updates"):
        for node_name, update in chunk.items():
            if node_name == "__interrupt__":
                continue
            if node_name not in st.session_state.completed:
                st.session_state.completed.append(node_name)
            status_box.info(f"▶ {NODE_LABELS.get(node_name, node_name)}")
            _render_progress(progress_box)
            if isinstance(update, dict) and update.get("errors"):
                for message in update["errors"]:
                    st.session_state.log.append(message)

    snapshot = app.get_state(config)
    pending = next(
        (task.interrupts[0].value for task in snapshot.tasks if task.interrupts),
        None,
    )
    st.session_state.pending_interrupt = pending
    st.session_state.final_state = snapshot.values
    st.session_state.running = bool(pending)


def _zip_bytes(paths: list[str], manifest: Path | None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            file = Path(path)
            if file.exists():
                archive.write(file, file.name)
        if manifest and manifest.exists():
            archive.write(manifest, manifest.name)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
_init_session()

st.title("🎠 AI Trend-to-Carousel Agent")
st.caption(
    "Give it a domain. It finds what people are talking about right now and turns "
    "it into a publish-ready 5-slide carousel."
)

with st.sidebar:
    st.header("Input")
    domain = st.text_input("Domain", value="AI & Technology")
    audience = st.text_input("Target audience", value="Software Developers")
    platform = st.selectbox("Platform", ["LinkedIn", "Instagram", "X / Twitter"], index=0)
    slide_count = st.slider("Number of slides", min_value=3, max_value=8, value=5)

    st.divider()
    st.header("Options")

    style = st.radio(
        "Visual style",
        ["modern", "sketch"],
        horizontal=True,
        help="modern = clean editorial carousel · sketch = hand-drawn whiteboard",
    )
    # Themes are style-specific: a hand-drawn deck on a near-black grid looks
    # wrong, so only the palettes that suit the chosen style are offered.
    theme_options = ["auto"] + list(SKETCH_THEMES if style == "sketch" else MODERN_THEMES)
    theme = st.selectbox("Colour theme", theme_options)
    # Topic approval is ON by default: the topic is the one decision worth a
    # human's attention, and it sits *before* the expensive deep research.
    approve_topic = st.checkbox(
        "Pause for topic approval",
        value=True,
        help="The run stops and waits for you to approve the discovered topic.",
    )
    approve_images = st.checkbox(
        "Pause before rendering",
        value=False,
        help="A second checkpoint to review slide copy before the PNGs are made.",
    )

    st.divider()
    st.caption(f"**Model:** {describe()}")
    st.caption(f"**Search:** {active_backend()}")
    st.caption(f"**Thread:** `{st.session_state.thread_id}`")
    st.caption(f"**Tracing:** {tracing_status()}")

    start = st.button("Generate carousel", type="primary", use_container_width=True)
    if st.button("New session", use_container_width=True):
        for key in ("thread_id", "completed", "pending_interrupt", "final_state", "running", "log"):
            st.session_state.pop(key, None)
        st.rerun()

left, right = st.columns([1, 2])

with left:
    st.subheader("Progress")
    progress_box = st.empty()
    status_box = st.empty()
    _render_progress(progress_box)

with right:
    result_box = st.container()

# --- start a run -----------------------------------------------------------
if start:
    # The HITL toggles are read by the nodes from settings.
    settings.require_topic_approval = approve_topic
    settings.require_image_approval = approve_images

    st.session_state.completed = []
    st.session_state.log = []
    st.session_state.final_state = None
    st.session_state.pending_interrupt = None

    state = initial_state(
        domain, audience, platform, slide_count, st.session_state.thread_id,
        slide_style=style,
    )
    # Remembered so that resumes after an interrupt tag the trace identically.
    st.session_state.run_meta = {
        "domain": domain,
        "audience": audience,
        "platform": platform,
        "slide_count": slide_count,
        "slide_style": style,
    }
    if theme != "auto":
        state["design_theme"] = theme

    with st.spinner("Running the graph… trend research and deep research take the longest."):
        try:
            _drive(state, progress_box, status_box)
        except Exception as exc:
            st.error(f"Run failed: {exc}")
            logging.getLogger(__name__).exception("run failed")

# --- handle an interrupt ---------------------------------------------------
pending = st.session_state.pending_interrupt
if pending:
    with right:
        st.subheader("⏸ Your approval is needed")

        if pending.get("type") == "topic_approval":
            st.markdown(f"**Topic:** {pending.get('selected_topic')}")
            st.markdown(f"**Angle:** {pending.get('angle')}")
            st.markdown(f"**Hook:** _{pending.get('hook')}_")
            with st.expander("Why this topic?"):
                st.write(pending.get("reason", ""))

            alternatives = pending.get("alternatives", [])
            choice = st.selectbox("Or pick a different trend", ["(keep this one)"] + alternatives)

            col_a, col_b = st.columns(2)
            if col_a.button("✅ Approve", use_container_width=True):
                decision = (
                    {"action": "approve"}
                    if choice == "(keep this one)"
                    else {"action": "choose_other", "topic": choice}
                )
                st.session_state.pending_interrupt = None
                with st.spinner("Continuing…"):
                    _drive(Command(resume=decision), progress_box, status_box)
                st.rerun()

            if col_b.button("🔄 Reject, find another", use_container_width=True):
                st.session_state.pending_interrupt = None
                with st.spinner("Selecting a different topic…"):
                    _drive(Command(resume={"action": "reject"}), progress_box, status_box)
                st.rerun()

        else:  # image approval
            st.markdown(f"**Topic:** {pending.get('topic')}")
            st.markdown(f"**Critic score:** {pending.get('score')}/10")
            for slide in pending.get("slides", []):
                st.markdown(f"- **{slide['slide_number']}.** `{slide['visual_type']}` {slide['title']}")

            col_a, col_b = st.columns(2)
            if col_a.button("🎨 Render images", use_container_width=True):
                st.session_state.pending_interrupt = None
                with st.spinner("Rendering…"):
                    _drive(Command(resume={"action": "approve"}), progress_box, status_box)
                st.rerun()
            if col_b.button("✋ Stop here", use_container_width=True):
                st.session_state.pending_interrupt = None
                _drive(Command(resume={"action": "reject_images"}), progress_box, status_box)
                st.rerun()

# --- show results ----------------------------------------------------------
final = st.session_state.final_state
if final and not pending:
    with right:
        topic = final.get("selected_topic", {})
        critique = final.get("critique", {})

        st.subheader("Selected topic")
        st.markdown(f"### {topic.get('selected_topic', '')}")
        st.markdown(f"**Angle:** {final.get('content_angle', '')}")
        st.markdown(f"**Hook:** _{topic.get('hook', '')}_")

        cols = st.columns(3)
        cols[0].metric("Critic score", f"{critique.get('overall_score', '?')}/10")
        cols[1].metric("Revisions", final.get("revision_count", 0))
        cols[2].metric("Approved", "Yes" if critique.get("approved") else "No")

        if final.get("quality_flag"):
            st.warning(
                "Shipped after hitting the revision limit - review before posting. "
                f"({final['quality_flag']})"
            )

        images = [p for p in final.get("image_paths", []) if Path(p).exists()]
        if images:
            st.subheader("Carousel")

            image_cols = st.columns(min(len(images), 5))
            for index, path in enumerate(images):
                column = image_cols[index % len(image_cols)]
                column.image(path, caption=f"Slide {index + 1}", use_container_width=True)
                # Every slide is individually downloadable - people often want
                # to re-order or drop one before posting.
                column.download_button(
                    f"⬇ Slide {index + 1}",
                    data=Path(path).read_bytes(),
                    file_name=Path(path).name,
                    mime="image/png",
                    use_container_width=True,
                    key=f"dl_{index}_{st.session_state.thread_id}",
                )

            manifest = Path(images[0]).parent / "carousel.json"
            st.download_button(
                f"⬇ Download all {len(images)} slides (.zip)",
                data=_zip_bytes(images, manifest),
                file_name=f"carousel-{st.session_state.thread_id}.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True,
                key=f"dl_all_{st.session_state.thread_id}",
            )
            st.caption(
                f"PNG · 1080×1350 · {final.get('slide_style') or 'modern'} style · "
                f"{final.get('design_theme', '')} theme. "
                "The .zip also contains carousel.json (copy, critique and source URLs)."
            )

        with st.expander("Slide content"):
            for slide in sorted(final.get("slides", []), key=lambda s: s["slide_number"]):
                st.markdown(
                    f"**{slide['slide_number']}. {slide.get('title','')}** "
                    f"`{slide.get('visual_type','')}`"
                )
                if slide.get("subtitle"):
                    st.caption(slide["subtitle"])
                if slide.get("body"):
                    st.write(slide["body"])
                if slide.get("bullets"):
                    for bullet in slide["bullets"]:
                        st.write(f"- {bullet}")
                if slide.get("source_references"):
                    st.caption("Sources: " + ", ".join(slide["source_references"]))
                st.divider()

        with st.expander("Critic report"):
            st.write(critique.get("summary", ""))
            for issue in critique.get("issues", []):
                st.markdown(
                    f"- **Slide {issue.get('slide_number')}** "
                    f"({issue.get('severity')}): {issue.get('issue')}  \n"
                    f"  _Fix:_ {issue.get('fix')}"
                )

        with st.expander("Research sources"):
            for source in final.get("research_sources", [])[:20]:
                st.markdown(f"- [{source.get('title','(untitled)')}]({source.get('url','')})")

        with st.expander("Raw state (for teaching)"):
            st.json(
                {
                    key: value
                    for key, value in final.items()
                    if key not in ("search_results", "research_sources")
                },
                expanded=False,
            )

if st.session_state.log:
    with st.sidebar:
        st.divider()
        st.caption("Notes")
        for message in st.session_state.log:
            st.caption(f"• {message}")
