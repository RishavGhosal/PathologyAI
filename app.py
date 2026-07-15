"""PathologyAI research/education review-priority Streamlit app."""

from __future__ import annotations

from hashlib import sha256

import numpy as np
from PIL import Image
import streamlit as st

try:
    import plotly.graph_objects as go
except ImportError:  # The basic app remains usable without the optional viewer.
    go = None

from pathology_ai.pipeline import (
    BatchResult,
    UploadPayload,
    format_file_size,
    process_uploads,
)
from pathology_ai.attention import get_attention_provider
from pathology_ai.triage import (
    LOWER_PRIORITY,
    NEEDS_BETTER_IMAGE,
    PRIORITIES,
    REVIEW_FIRST,
    priority_sort_key,
)
from pathology_ai.uni_provider import get_uni_provider_status


DISCLAIMER = (
    "This research and education prototype provides review-priority suggestions only. "
    "It does not provide a medical diagnosis and does not replace review by a qualified "
    "pathologist."
)
MAX_BROWSER_PREVIEW_SIDE = 1600
NATIVE_PREVIEW_PIXEL_LIMIT = 4_000_000


st.set_page_config(
    page_title="PathologyAI | Research/Education Prototype",
    page_icon="🔬",
    layout="wide",
)


@st.cache_data(show_spinner=False, max_entries=8)
def _process_cached(
    payload_values: tuple[tuple[str, bytes, str], ...],
    provider_cache_key: str,
    use_uni: bool,
) -> BatchResult:
    del provider_cache_key  # Included so model/checkpoint changes invalidate cached results.
    payloads = [
        UploadPayload(name=name, data=data, mime_type=mime_type)
        for name, data, mime_type in payload_values
    ]
    provider = get_attention_provider(prefer_uni=use_uni)
    return process_uploads(payloads, provider=provider)


def _batch_fingerprint(payload_values: tuple[tuple[str, bytes, str], ...]) -> str:
    digest = sha256()
    for name, data, mime_type in payload_values:
        digest.update(name.encode("utf-8", errors="replace"))
        digest.update(b"\0")
        digest.update(mime_type.encode("utf-8", errors="replace"))
        digest.update(b"\0")
        digest.update(data)
    return digest.hexdigest()


def _browser_preview(image: Image.Image) -> tuple[Image.Image, bool]:
    preview = image.convert("RGB").copy()
    downsampled = max(preview.size) > MAX_BROWSER_PREVIEW_SIDE
    if downsampled:
        preview.thumbnail(
            (MAX_BROWSER_PREVIEW_SIDE, MAX_BROWSER_PREVIEW_SIDE),
            Image.Resampling.LANCZOS,
        )
    return preview, downsampled


def _render_fallback_viewer(image: Image.Image, key: str, caption: str) -> None:
    reset_key = f"{key}_reset"
    zoom_key = f"{key}_zoom"
    if st.button("Reset view", key=reset_key):
        st.session_state[zoom_key] = 100
    zoom_percent = st.slider(
        "Preview zoom",
        min_value=25,
        max_value=200,
        value=100,
        step=25,
        key=zoom_key,
    )
    preview, downsampled = _browser_preview(image)
    display_width = max(160, int(preview.width * (zoom_percent / 100.0)))
    st.image(preview, width=display_width, caption=caption)
    message = (
        "Interactive viewer unavailable because Plotly is not installed. "
        "Use the zoom slider; browser scrolling provides basic navigation."
    )
    if downsampled:
        message += " This browser preview is downsampled for stability."
    st.info(message)


def render_image_viewer(image: Image.Image, key: str, caption: str) -> None:
    """Render an offline zoom/pan viewer, or a reliable native fallback."""

    if go is None:
        _render_fallback_viewer(image, key, caption)
        return

    try:
        preview, downsampled = _browser_preview(image)
        array = np.asarray(preview, dtype=np.uint8)
        height = max(360, min(620, int(760 * preview.height / max(preview.width, 1))))
        figure = go.Figure(go.Image(z=array))
        figure.update_layout(
            dragmode="pan",
            height=height,
            margin=dict(l=0, r=0, t=8, b=0),
            uirevision=key,
        )
        figure.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
        figure.update_yaxes(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        )
        st.plotly_chart(
            figure,
            width="stretch",
            height=height,
            key=f"plot_{key}",
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "scrollZoom": True,
                "modeBarButtonsToRemove": ["select2d", "lasso2d"],
            },
        )
        preview_note = (
            " Browser preview is downsampled for stability." if downsampled else ""
        )
        st.caption(
            f"{caption} Use the mouse wheel or toolbar to zoom, drag to pan, "
            f"Autoscale to fit, and Reset axes to reset.{preview_note}"
        )
    except Exception:
        st.warning(
            "The interactive viewer could not load, so the basic image-viewer fallback "
            "is shown instead."
        )
        _render_fallback_viewer(image, key, caption)


def _initialize_review_state(batch: BatchResult) -> dict[str, dict[str, object]]:
    reviews = dict(st.session_state.get("reviews", {}))
    for record in batch.records:
        state = dict(
            reviews.get(
                record.image_id,
                {
                    "notes": "",
                    "priority": record.triage.suggested_priority,
                    "reviewed": False,
                },
            )
        )
        priority_key = f"priority_{record.image_id}"
        notes_key = f"notes_{record.image_id}"
        reviewed_key = f"reviewed_{record.image_id}"

        if priority_key in st.session_state and st.session_state[priority_key] in PRIORITIES:
            state["priority"] = st.session_state[priority_key]
        if notes_key in st.session_state:
            state["notes"] = st.session_state[notes_key]
        if reviewed_key in st.session_state:
            state["reviewed"] = bool(st.session_state[reviewed_key])

        if state.get("priority") not in PRIORITIES:
            state["priority"] = record.triage.suggested_priority
        state.setdefault("notes", "")
        state.setdefault("reviewed", False)

        if priority_key not in st.session_state:
            st.session_state[priority_key] = state["priority"]
        if notes_key not in st.session_state:
            st.session_state[notes_key] = state["notes"]
        if reviewed_key not in st.session_state:
            st.session_state[reviewed_key] = state["reviewed"]
        reviews[record.image_id] = state

    st.session_state["reviews"] = reviews
    return reviews


def _sync_selected_review(record_id: str) -> None:
    reviews = dict(st.session_state.get("reviews", {}))
    if record_id not in reviews:
        return
    state = dict(reviews[record_id])
    state["priority"] = st.session_state.get(
        f"priority_{record_id}", state.get("priority", LOWER_PRIORITY)
    )
    state["notes"] = st.session_state.get(f"notes_{record_id}", state.get("notes", ""))
    state["reviewed"] = bool(
        st.session_state.get(f"reviewed_{record_id}", state.get("reviewed", False))
    )
    reviews[record_id] = state
    st.session_state["reviews"] = reviews


def _render_dashboard(batch: BatchResult, reviews: dict[str, dict[str, object]]) -> None:
    effective_counts = {priority: 0 for priority in PRIORITIES}
    reviewed_count = 0
    for record in batch.records:
        state = reviews[record.image_id]
        effective_counts[str(state["priority"])] += 1
        reviewed_count += int(bool(state["reviewed"]))

    st.subheader("Summary Dashboard")
    first_row = st.columns(4)
    first_row[0].metric("Total files uploaded", batch.uploaded_count)
    first_row[1].metric("Valid images", len(batch.records))
    first_row[2].metric(REVIEW_FIRST, effective_counts[REVIEW_FIRST])
    first_row[3].metric(NEEDS_BETTER_IMAGE, effective_counts[NEEDS_BETTER_IMAGE])
    second_row = st.columns(3)
    second_row[0].metric(LOWER_PRIORITY, effective_counts[LOWER_PRIORITY])
    second_row[1].metric("Skipped or failed files", len(batch.skipped))
    second_row[2].metric("Reviewed images", reviewed_count)
    st.caption(
        "Priority counts include reviewer overrides. ZIP members are counted under valid "
        "or skipped items; Total files uploaded counts top-level uploads."
    )


def _render_skipped_files(batch: BatchResult) -> None:
    if not batch.skipped:
        return
    with st.expander(f"Skipped or failed files ({len(batch.skipped)})", expanded=True):
        rows = [
            {
                "Uploaded source": item.source_name,
                "File": item.file_name,
                "Reason": item.reason,
            }
            for item in batch.skipped
        ]
        st.dataframe(rows, hide_index=True, width="stretch")


def _render_batch_table(
    batch: BatchResult,
    ordered_records: list,
    reviews: dict[str, dict[str, object]],
) -> None:
    st.subheader("Batch Results")
    rows = []
    for record in ordered_records:
        state = reviews[record.image_id]
        rows.append(
            {
                "Filename": record.display_name,
                "File type": record.file_type,
                "Dimensions": f"{record.width} × {record.height} px",
                "File size": format_file_size(record.size_bytes),
                "Image Quality": "Adequate" if record.quality.adequate else "Needs attention",
                "Analysis source": record.attention.provider_name,
                "Suggested priority": record.triage.suggested_priority,
                "Current priority": state["priority"],
                "Reviewed": "Yes" if state["reviewed"] else "No",
            }
        )
    st.dataframe(rows, hide_index=True, width="stretch")


def _render_original_resolution(record) -> None:
    with st.expander("Original-resolution preview"):
        pixel_count = record.width * record.height
        if pixel_count <= NATIVE_PREVIEW_PIXEL_LIMIT and max(record.image.size) <= 2400:
            st.image(
                record.image,
                width="content",
                caption=f"Native image: {record.width} × {record.height} px",
            )
        else:
            preview = record.image.convert("RGB").copy()
            preview.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
            st.info(
                f"The original is {record.width} × {record.height} px. To keep the "
                "browser responsive, this preview is downscaled to "
                f"{preview.width} × {preview.height} px."
            )
            st.image(preview, width="content")


def _render_image_detail(record, reviews: dict[str, dict[str, object]]) -> None:
    st.divider()
    st.subheader("Selected Image")
    metadata_columns = st.columns(4)
    metadata_columns[0].metric("File type", record.file_type)
    metadata_columns[1].metric("Width", f"{record.width} px")
    metadata_columns[2].metric("Height", f"{record.height} px")
    metadata_columns[3].metric("File size", format_file_size(record.size_bytes))
    st.caption(record.display_name)
    for note in record.metadata_notes:
        if "fallback was used" in note:
            st.warning(note)
        else:
            st.info(note)

    st.markdown("#### Original Image and Model Attention")
    if record.attention.uses_trained_encoder:
        st.info(
            "A local pretrained UNI encoder generated this exploratory feature-variation "
            "visualization. It is not a validated clinical attention map. No trained "
            "review-priority classifier is loaded, and UNI did not generate the priority "
            "label."
        )
    elif record.attention.is_demonstration:
        st.info(
            "Deterministic demonstration attention — based on image appearance such as "
            "contrast and edges, not learned pathology features. No trained or validated "
            "medical model is loaded."
        )
    viewer_columns = st.columns(2)
    with viewer_columns[0]:
        st.markdown("**Original Image**")
        render_image_viewer(
            record.image,
            f"original_{record.image_id}",
            "Original image.",
        )
    with viewer_columns[1]:
        st.markdown("**Attention Overlay**")
        render_image_viewer(
            record.attention.overlay,
            f"attention_{record.image_id}",
            record.attention.overlay_caption,
        )
    st.markdown(f"**Plain-language explanation:** {record.attention.explanation}")
    _render_original_resolution(record)

    quality_column, priority_column = st.columns(2)
    with quality_column:
        st.markdown("#### Image Quality")
        if record.quality.adequate:
            st.success("Image passes the MVP quality checks.")
        else:
            st.warning(f"{NEEDS_BETTER_IMAGE}: quality issues were found.")
            for reason in record.quality.reasons:
                st.write(f"• {reason}")
        with st.expander("Quality check details"):
            st.write(f"Brightness score: {record.quality.metrics['brightness']:.1f} / 255")
            st.write(f"Contrast score: {record.quality.metrics['contrast']:.1f}")
            st.write(f"Edge-sharpness score: {record.quality.metrics['blur_score']:.1f}")
            st.caption(
                "These are simple presentation-quality heuristics, not biological or "
                "clinical measurements."
            )

    with priority_column:
        st.markdown("#### Review Priority")
        state = reviews[record.image_id]
        st.write(f"**Suggested:** {record.triage.suggested_priority}")
        st.write(f"**Current reviewer choice:** {state['priority']}")
        st.write(record.triage.explanation)
        st.caption(f"Priority source: {record.triage.priority_source}")
        if record.attention.uses_trained_encoder:
            st.caption("UNI did not generate this priority label.")
        st.caption("Human Review Required • Priority is review order only.")

    st.markdown("#### Manual Review")
    st.text_area(
        "Reviewer notes (kept in this browser session)",
        key=f"notes_{record.image_id}",
        placeholder="Add education/research review notes here...",
        height=120,
    )
    st.selectbox(
        "Confirm or override the suggested priority",
        options=PRIORITIES,
        key=f"priority_{record.image_id}",
        help="Overrides change review order only; they are not medical conclusions.",
    )
    reviewed = st.checkbox(
        "Mark this image as reviewed",
        key=f"reviewed_{record.image_id}",
    )
    _sync_selected_review(record.image_id)
    if reviewed:
        st.success("Marked as reviewed for this Streamlit session.")
    if not record.quality.adequate:
        st.warning(
            "The image-quality warning remains active even if a reviewer chooses a "
            "different review priority."
        )


def main() -> None:
    st.title("🔬 PathologyAI")
    st.caption("Research/Education Prototype • Review Priority • Human Review Required")
    st.warning(DISCLAIMER)

    uni_status = get_uni_provider_status()
    with st.sidebar:
        st.header("Prototype Status")
        if uni_status.ready:
            st.success(uni_status.summary)
            use_uni = st.toggle(
                "Use local UNI feature visualization",
                value=False,
                help=(
                    "Loads the local ViT-L encoder when an image is processed. CPU "
                    "inference can be slow and uses substantial memory."
                ),
            )
            if use_uni:
                st.write("**Model Attention source:** Local UNI encoder")
                st.caption(
                    "Exploratory UNI feature variation is enabled. The deterministic "
                    "rule still supplies review priority."
                )
            else:
                st.write("**Model Attention source:** Deterministic demonstration")
                st.caption("Enable the toggle to load UNI for uploaded images.")
        else:
            use_uni = False
            st.warning(uni_status.summary)
            st.caption(uni_status.detail)
            st.write("**Model Attention source:** Deterministic demonstration")
        st.caption("No trained review-priority classifier is loaded.")
        if go is None:
            st.warning("Plotly is unavailable; the basic image-viewer fallback is active.")
        else:
            st.success("Offline interactive viewer available")
        with st.expander("MVP scope and limits"):
            st.write(
                "Supports PNG, JPG/JPEG, TIFF, and ZIP batches. This MVP does not process "
                "whole-slide formats, make disease predictions, or download model weights. "
                "UNI is used only when its local checkpoint is present and the toggle is on."
            )

    st.subheader("Upload Pathology Images")
    uploaded_files = st.file_uploader(
        "Choose one or more images or ZIP files",
        accept_multiple_files=True,
        type=None,
        help=(
            "Supported: PNG, JPG/JPEG, TIFF, and ZIP. Other, empty, corrupted, or unreadable "
            "files will be listed with a reason."
        ),
    )
    st.caption(
        "ZIPs are inspected safely in memory. Safe nested image files are processed; folder "
        "entries and unsafe paths are not extracted."
    )

    if not uploaded_files:
        st.info(
            "Upload an image or ZIP to begin. The app will check image quality, suggest "
            "review order, and require a human reviewer to confirm the result."
        )
        st.divider()
        st.caption("PathologyAI • Research/Education Prototype • Human Review Required")
        return

    payload_values = tuple(
        (uploaded.name, uploaded.getvalue(), getattr(uploaded, "type", "") or "")
        for uploaded in uploaded_files
    )
    st.session_state["active_batch_fingerprint"] = _batch_fingerprint(payload_values)
    with st.spinner("Checking files and preparing review-priority suggestions..."):
        selected_provider_key = (
            uni_status.cache_key if use_uni else "deterministic-demo:v1"
        )
        batch = _process_cached(payload_values, selected_provider_key, use_uni)

    reviews = _initialize_review_state(batch)
    _render_dashboard(batch, reviews)
    _render_skipped_files(batch)

    if not batch.records:
        st.error(
            "No valid images were available for review. Check the skipped-file reasons and "
            "upload a supported, readable image."
        )
        st.divider()
        st.caption("PathologyAI • Research/Education Prototype • Human Review Required")
        return

    ordered_records = sorted(
        batch.records,
        key=lambda record: (
            priority_sort_key(str(reviews[record.image_id]["priority"])),
            record.display_name.casefold(),
        ),
    )
    _render_batch_table(batch, ordered_records, reviews)

    record_by_id = {record.image_id: record for record in ordered_records}
    selection_options = [record.image_id for record in ordered_records]
    if st.session_state.get("selected_image_id") not in selection_options:
        st.session_state["selected_image_id"] = selection_options[0]

    selected_id = st.selectbox(
        "Select an image for detailed review",
        options=selection_options,
        key="selected_image_id",
        format_func=lambda image_id: (
            f"{record_by_id[image_id].display_name} — "
            f"{reviews[image_id]['priority']}"
            f"{' ✓' if reviews[image_id]['reviewed'] else ''}"
        ),
    )
    _render_image_detail(record_by_id[selected_id], reviews)

    st.divider()
    st.caption("PathologyAI • Research/Education Prototype • Human Review Required")


if __name__ == "__main__":
    main()
