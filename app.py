"""PathologyAI research/education review-priority Streamlit app."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import math

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
from pathology_ai.dashboard_metrics import build_operational_metrics
from pathology_ai.review_export import (
    build_review_export_csv,
    validate_group_id,
    validate_review_fields,
)
from pathology_ai.review_model import get_review_model, get_review_model_status
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
DOMAIN_LABEL_TO_VALUE = {
    "Unknown or other tissue": "unknown_or_other",
    "MHIST-like colorectal-polyp patches": "mhist_like_colorectal_polyp",
}


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
    review_model_cache_key: str,
    use_review_model: bool,
) -> BatchResult:
    # Keys are included so local artifact changes invalidate processed results.
    del provider_cache_key, review_model_cache_key
    payloads = [
        UploadPayload(name=name, data=data, mime_type=mime_type)
        for name, data, mime_type in payload_values
    ]
    provider = get_attention_provider(prefer_uni=use_uni)
    review_model = get_review_model() if use_review_model else None
    return process_uploads(payloads, provider=provider, review_model=review_model)


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
                    "group_id": "",
                    "priority": record.triage.suggested_priority,
                    "reviewed": False,
                    "priority_overridden": False,
                },
            )
        )
        priority_key = f"priority_{record.image_id}"
        notes_key = f"notes_{record.image_id}"
        group_key = f"group_{record.image_id}"
        reviewed_key = f"reviewed_{record.image_id}"

        if priority_key in st.session_state and st.session_state[priority_key] in PRIORITIES:
            state["priority"] = st.session_state[priority_key]
        if notes_key in st.session_state:
            state["notes"] = st.session_state[notes_key]
        if group_key in st.session_state:
            state["group_id"] = st.session_state[group_key]
        if reviewed_key in st.session_state:
            state["reviewed"] = bool(st.session_state[reviewed_key])

        suggestion_key = (
            f"{record.triage.priority_source}|{record.triage.suggested_priority}"
        )
        previous_suggestion_key = state.get("suggestion_key")
        previous_suggested_priority = state.get("last_suggested_priority")
        if "priority_overridden" not in state:
            state["priority_overridden"] = bool(
                previous_suggested_priority
                and state.get("priority") != previous_suggested_priority
            )
        suggestion_changed = (
            previous_suggestion_key is not None
            and previous_suggestion_key != suggestion_key
        )
        if (
            suggestion_changed
            and not bool(state.get("priority_overridden", False))
            and not bool(state.get("reviewed", False))
        ):
            state["priority"] = record.triage.suggested_priority
            st.session_state[priority_key] = record.triage.suggested_priority
        state["suggestion_key"] = suggestion_key
        state["last_suggested_priority"] = record.triage.suggested_priority

        if state.get("priority") not in PRIORITIES:
            state["priority"] = record.triage.suggested_priority
        state.setdefault("notes", "")
        state.setdefault("group_id", "")
        state.setdefault("group_id_format_validated", False)
        state.setdefault("reviewed", False)
        state.setdefault("reviewed_at_utc", "")

        if priority_key not in st.session_state:
            st.session_state[priority_key] = state["priority"]
        if notes_key not in st.session_state:
            st.session_state[notes_key] = state["notes"]
        if group_key not in st.session_state:
            st.session_state[group_key] = state["group_id"]
        if reviewed_key not in st.session_state:
            st.session_state[reviewed_key] = state["reviewed"]
        reviews[record.image_id] = state

    st.session_state["reviews"] = reviews
    return reviews


def _mark_priority_override(record_id: str) -> None:
    reviews = dict(st.session_state.get("reviews", {}))
    if record_id not in reviews:
        return
    state = dict(reviews[record_id])
    selected = st.session_state.get(f"priority_{record_id}")
    state["priority_overridden"] = selected != state.get("last_suggested_priority")
    state["priority"] = selected
    reviews[record_id] = state
    st.session_state["reviews"] = reviews


def _sync_selected_review(record_id: str) -> None:
    reviews = dict(st.session_state.get("reviews", {}))
    if record_id not in reviews:
        return
    state = dict(reviews[record_id])
    state["priority"] = st.session_state.get(
        f"priority_{record_id}", state.get("priority", LOWER_PRIORITY)
    )
    state["notes"] = st.session_state.get(f"notes_{record_id}", state.get("notes", ""))
    state["group_id"] = st.session_state.get(
        f"group_{record_id}", state.get("group_id", "")
    )
    state["reviewed"] = bool(
        st.session_state.get(f"reviewed_{record_id}", state.get("reviewed", False))
    )
    reviews[record_id] = state
    st.session_state["reviews"] = reviews


def _review_state_from_widgets(record, reviews: dict[str, dict[str, object]]) -> dict[str, object]:
    state = dict(reviews[record.image_id])
    state["priority"] = st.session_state.get(
        f"priority_{record.image_id}", state.get("priority", record.triage.suggested_priority)
    )
    state["notes"] = st.session_state.get(
        f"notes_{record.image_id}", state.get("notes", "")
    )
    state["group_id"] = st.session_state.get(
        f"group_{record.image_id}", state.get("group_id", "")
    )
    state["priority_overridden"] = (
        state["priority"] != record.triage.suggested_priority
    )
    return state


def _save_review(record, next_image_id: str | None = None) -> None:
    reviews = dict(st.session_state.get("reviews", {}))
    if record.image_id not in reviews:
        return
    state = _review_state_from_widgets(record, reviews)
    try:
        validate_review_fields(record, state)
    except ValueError as exc:
        st.session_state[f"review_error_{record.image_id}"] = str(exc)
        return
    state["reviewed"] = True
    state["reviewed_at_utc"] = datetime.now(timezone.utc).isoformat()
    state["group_id_format_validated"] = True
    state["review_validation_version"] = 1
    state["suggested_priority_at_review"] = record.triage.suggested_priority
    state["priority_source_at_review"] = record.triage.priority_source
    state["priority_method_at_review"] = record.triage.priority_method
    state["priority_fallback_reason_at_review"] = record.triage.fallback_reason or ""
    state["review_first_proxy_score_at_review"] = record.triage.review_first_score
    state["attention_provider_at_review"] = record.attention.provider_name
    state["embedding_model_at_review"] = record.attention.embedding_model or ""
    state["embedding_at_review"] = (
        None
        if record.attention.embedding is None
        else tuple(float(value) for value in record.attention.embedding)
    )
    reviews[record.image_id] = state
    st.session_state["reviews"] = reviews
    st.session_state[f"reviewed_{record.image_id}"] = True
    st.session_state.pop(f"review_error_{record.image_id}", None)
    st.session_state["queue_message"] = "Review saved."
    if next_image_id is not None:
        st.session_state["selected_image_id"] = next_image_id


def _reopen_review(record_id: str) -> None:
    reviews = dict(st.session_state.get("reviews", {}))
    if record_id not in reviews:
        return
    state = dict(reviews[record_id])
    state["reviewed"] = False
    state["reviewed_at_utc"] = ""
    state["group_id_format_validated"] = False
    state.pop("review_validation_version", None)
    for key in (
        "suggested_priority_at_review",
        "priority_source_at_review",
        "priority_method_at_review",
        "priority_fallback_reason_at_review",
        "review_first_proxy_score_at_review",
        "attention_provider_at_review",
        "embedding_model_at_review",
        "embedding_at_review",
    ):
        state.pop(key, None)
    reviews[record_id] = state
    st.session_state["reviews"] = reviews
    st.session_state[f"reviewed_{record_id}"] = False


def _select_image(image_id: str) -> None:
    st.session_state["selected_image_id"] = image_id


def _apply_group_to_source(record, records: list) -> None:
    reviews = dict(st.session_state.get("reviews", {}))
    try:
        group_id = validate_group_id(st.session_state.get(f"group_{record.image_id}", ""))
    except ValueError as exc:
        st.session_state[f"group_apply_message_{record.image_id}"] = str(exc)
        return
    updated = 0
    for candidate in records:
        if candidate.source_name != record.source_name:
            continue
        state = dict(reviews[candidate.image_id])
        if str(state.get("group_id", "")).strip():
            continue
        state["group_id"] = group_id
        reviews[candidate.image_id] = state
        st.session_state[f"group_{candidate.image_id}"] = group_id
        updated += 1
    st.session_state["reviews"] = reviews
    st.session_state[f"group_apply_message_{record.image_id}"] = (
        f"Applied {group_id} to {updated} ungrouped image(s) from this uploaded source."
    )


def _record_sort_key(record, reviews: dict[str, dict[str, object]]) -> tuple:
    score = record.triage.review_first_score
    sortable_score = float(score) if score is not None and math.isfinite(float(score)) else -1.0
    return (
        priority_sort_key(str(reviews[record.image_id]["priority"])),
        -sortable_score,
        record.display_name.casefold(),
    )


def _filter_records(
    records: list,
    reviews: dict[str, dict[str, object]],
    status_filter: str,
    priority_filter: list[str],
    group_filter: str,
) -> list:
    filtered = []
    for record in records:
        state = reviews[record.image_id]
        reviewed = bool(state.get("reviewed", False))
        if status_filter == "Awaiting" and reviewed:
            continue
        if status_filter == "Reviewed" and not reviewed:
            continue
        if state.get("priority") not in priority_filter:
            continue
        group_id = str(state.get("group_id", "")).strip()
        if group_filter == "Ungrouped" and group_id:
            continue
        if group_filter not in ("All groups", "Ungrouped") and group_id != group_filter:
            continue
        filtered.append(record)
    return filtered


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


def _render_safety_limitations() -> None:
    st.markdown("#### Safety and limitations")
    limitations = [
        "No cancer classifier is loaded, and no diagnosis is produced.",
        "Proxy scores are not calibrated probabilities.",
        "Threshold-dependent metrics apply only to the displayed classification threshold.",
        "MHIST contains colorectal-polyp images; other tissues are outside the demonstrated domain.",
        "Patient/case-level split independence could not be independently verified for MHIST.",
        "Domain status is reviewer-declared, not automatically detected.",
        (
            "The app enforces group-ID formatting but cannot determine whether text "
            "contains identifying information; de-identification is the reviewer's "
            "responsibility."
        ),
        (
            "The time estimate covers only unusable or failed inputs and does not claim "
            "that priority ranking saves time."
        ),
        "Human review is required for every image.",
    ]
    for limitation in limitations:
        st.write(f"• {limitation}")


def _render_model_evaluation(review_model_status) -> None:
    st.subheader("MHIST Annotator-Agreement Proxy Evaluation — Not Cancer Accuracy")
    st.warning(
        "These values measure a held-out MHIST annotator-agreement proxy. They do not "
        "measure cancer detection, diagnosis accuracy, or clinical urgency."
    )
    if not review_model_status.ready:
        st.info("Evaluation metrics unavailable because the local prototype head is unavailable.")
        _render_safety_limitations()
        return
    if not getattr(review_model_status, "evaluation_valid", False):
        evaluation_error = (
            getattr(review_model_status, "evaluation_error", None)
            or "Evaluation metrics unavailable."
        )
        if "threshold mismatch" in evaluation_error.casefold():
            st.warning(
                "Evaluation artifact mismatch. "
                f"{evaluation_error} Threshold-dependent metrics are hidden; "
                "inference remains available."
            )
        else:
            st.warning(f"{evaluation_error} Inference remains available.")
        _render_safety_limitations()
        return
    report = getattr(review_model_status, "evaluation_report", {})
    overall = report.get("overall_test_metrics", {})
    threshold = getattr(review_model_status, "decision_threshold", None)
    if threshold is None:
        st.warning("Evaluation metrics unavailable because the classification threshold is missing.")
        _render_safety_limitations()
        return

    top = st.columns(4)
    top[0].metric("Classification threshold used", f"{threshold:.3f}")
    top[1].metric("Held-out test images", int(overall["sample_count"]))
    top[2].metric("Balanced accuracy", f"{float(overall['balanced_accuracy']):.3f}")
    top[3].metric("ROC-AUC", f"{float(overall['roc_auc']):.3f}")
    second = st.columns(4)
    second[0].metric("Average precision", f"{float(overall['average_precision']):.3f}")
    second[1].metric(
        "Review First precision",
        f"{float(overall['review_first_precision']):.3f}",
    )
    second[2].metric(
        "Review First recall", f"{float(overall['review_first_recall']):.3f}"
    )
    second[3].metric("Review First F1", f"{float(overall['review_first_f1']):.3f}")
    third = st.columns(3)
    third[0].metric(
        "Lower Priority specificity",
        f"{float(overall['lower_priority_specificity']):.3f}",
    )
    predicted_count = int(overall["predicted_review_first_count"])
    predicted_fraction = float(overall["predicted_review_first_fraction"])
    third[1].metric("Predicted Review First queue", predicted_count)
    third[2].metric("Predicted queue fraction", f"{predicted_fraction:.1%}")

    matrix = overall["confusion_matrix"]
    values = matrix["values"]
    row_labels = matrix["row_labels"]
    column_labels = matrix["column_labels"]
    st.markdown("#### Held-out confusion matrix")
    st.caption("Rows are predicted proxy labels; columns are actual proxy labels.")
    if (
        len(values) == 2
        and all(len(row) == 2 for row in values)
        and len(row_labels) == 2
        and len(column_labels) == 2
    ):
        if go is not None:
            figure = go.Figure(
                data=go.Heatmap(
                    z=values,
                    x=[f"Actual {label}" for label in column_labels],
                    y=[f"Predicted {label}" for label in row_labels],
                    colorscale="Blues",
                    showscale=False,
                    text=values,
                    texttemplate="%{text}",
                )
            )
            figure.update_layout(height=330, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(figure, width="stretch", key="proxy_confusion_matrix")
        else:
            st.dataframe(
                [
                    {
                        "Predicted proxy": row_labels[row_index],
                        f"Actual {column_labels[0]}": values[row_index][0],
                        f"Actual {column_labels[1]}": values[row_index][1],
                    }
                    for row_index in range(2)
                ],
                hide_index=True,
                width="stretch",
            )
    else:
        st.info("Confusion-matrix values are unavailable.")

    capture_rows = overall["review_first_capture_by_queue_fraction"]
    st.markdown("#### Review First captured near the top of the queue")
    if capture_rows:
        st.dataframe(
            [
                {
                    "Top queue fraction": f"{float(row['queue_fraction']):.0%}",
                    "Queue size": int(row["queue_size"]),
                    "Review First captured": (
                        f"{int(row['captured_review_first_count'])}/"
                        f"{int(row['total_review_first_count'])}"
                    ),
                    "Capture": f"{float(row['capture_fraction']):.1%}",
                }
                for row in capture_rows
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("Top-of-queue capture metrics are unavailable.")

    _render_safety_limitations()


def _render_operational_dashboard(
    batch: BatchResult,
    reviews: dict[str, dict[str, object]],
    domain_label: str,
    screening_seconds: float,
    use_uni: bool,
    use_review_model: bool,
) -> None:
    metrics = build_operational_metrics(
        batch,
        reviews,
        domain_declaration=domain_label,
        screening_seconds_per_image=screening_seconds,
    )
    st.subheader("Operational Dashboard")
    first = st.columns(4)
    first[0].metric("Total files uploaded", batch.uploaded_count)
    first[1].metric("Valid images", metrics.total_images)
    first[2].metric("Awaiting review", metrics.awaiting_count)
    first[3].metric("Reviewed images", metrics.reviewed_count)
    second = st.columns(4)
    second[0].metric("Review complete", f"{metrics.reviewed_percentage:.0f}%")
    second[1].metric(REVIEW_FIRST, metrics.effective_priority_counts[REVIEW_FIRST])
    second[2].metric(
        NEEDS_BETTER_IMAGE,
        metrics.effective_priority_counts[NEEDS_BETTER_IMAGE],
    )
    second[3].metric(LOWER_PRIORITY, metrics.effective_priority_counts[LOWER_PRIORITY])
    third = st.columns(3)
    third[0].metric("Image-quality passes", metrics.quality_pass_count)
    third[1].metric("Skipped or failed files", metrics.skipped_count)
    third[2].metric(
        "Estimated time avoided reviewing unusable images",
        f"{metrics.estimated_time_avoided_seconds / 60.0:.1f} min",
    )
    st.caption(
        f"The time estimate uses {metrics.screening_seconds_per_image:.0f} seconds per "
        "unusable image and includes only Needs Better Image plus skipped/failed inputs. "
        "It is user-configured, not measured workflow efficiency."
    )

    st.markdown("#### Image Quality")
    issue_labels = {
        "blur": "Blur failures",
        "small_dimensions": "Small-dimension failures",
        "excessive_darkness": "Excessively dark",
        "excessive_brightness": "Excessively bright",
        "blank_or_nearly_uniform": "Blank/nearly uniform",
    }
    quality_columns = st.columns(6)
    for index, (code, label) in enumerate(issue_labels.items()):
        quality_columns[index].metric(label, metrics.quality_issue_counts.get(code, 0))
    quality_columns[5].metric(
        "Crop advisories",
        metrics.quality_advisory_counts.get("possible_edge_truncation", 0),
    )

    st.markdown("#### Prototype Model Summary")
    st.write(
        f"**Experimental head:** {'Enabled' if use_review_model else 'Disabled'}  •  "
        f"**UNI feature visualization:** {'Enabled' if use_uni else 'Disabled'}"
    )
    model_columns = st.columns(6)
    model_columns[0].metric("UNI embeddings generated", metrics.embedding_success_count)
    model_columns[1].metric(
        "Experimental-head predictions", metrics.experimental_model_prediction_count
    )
    model_columns[2].metric(
        "Deterministic predictions", metrics.deterministic_prediction_count
    )
    model_columns[3].metric("Quality-gated images", metrics.quality_gate_count)
    model_columns[4].metric("Runtime fallbacks", metrics.runtime_fallback_count)
    model_columns[5].metric("Domain warnings", metrics.domain_warning_count)
    if metrics.domain_warning_count:
        st.warning(
            f"{metrics.domain_warning_count} model-scored image(s) have an unknown or "
            "declared non-MHIST domain. This is a reviewer declaration, not automatic detection."
        )
    if metrics.proxy_scores:
        st.metric(
            "Mean experimental agreement-proxy score",
            f"{metrics.mean_proxy_score:.3f}",
            help="This is not calibrated probability or clinical confidence.",
        )
        if go is not None:
            histogram = go.Figure(
                data=go.Histogram(x=list(metrics.proxy_scores), nbinsx=10)
            )
            histogram.update_layout(
                height=280,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="Experimental agreement-proxy score",
                yaxis_title="Images",
            )
            st.plotly_chart(histogram, width="stretch", key="proxy_score_histogram")
        else:
            st.dataframe(
                [{"Experimental agreement-proxy score": score} for score in metrics.proxy_scores],
                hide_index=True,
                width="stretch",
            )
        st.caption("Score distribution is descriptive only and is not a diagnosis output.")
    else:
        st.info("No experimental proxy scores are available in this batch.")

    st.markdown("#### Human Review Performance")
    agreement_columns = st.columns(6)
    agreement_columns[0].metric("Suggestions confirmed", metrics.suggestion_confirmed_count)
    agreement_columns[1].metric("Suggestions overridden", metrics.suggestion_overridden_count)
    agreement_columns[2].metric(
        "Suggestion–reviewer agreement",
        "N/A"
        if metrics.suggestion_agreement_percentage is None
        else f"{metrics.suggestion_agreement_percentage:.1f}%",
    )
    agreement_columns[3].metric("Model-reviewed images", metrics.model_reviewed_count)
    agreement_columns[4].metric("Model overrides", metrics.model_overridden_count)
    agreement_columns[5].metric(
        "Model–reviewer agreement",
        "N/A"
        if metrics.model_agreement_percentage is None
        else f"{metrics.model_agreement_percentage:.1f}%",
    )
    st.dataframe(
        [
            {
                "Suggested priority": row.suggested_priority,
                "Reviewed": row.reviewed_count,
                "Confirmed": row.confirmed_count,
                "Overridden": row.overridden_count,
                "Agreement": (
                    "N/A"
                    if row.agreement_percentage is None
                    else f"{row.agreement_percentage:.1f}%"
                ),
            }
            for row in metrics.agreement_by_suggested_priority
        ],
        hide_index=True,
        width="stretch",
    )
    _render_skipped_files(batch)


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
                "Image Quality": (
                    "Adequate — advisory"
                    if record.quality.adequate and record.quality.advisories
                    else "Adequate"
                    if record.quality.adequate
                    else "Needs Better Image"
                ),
                "Attention source": record.attention.provider_name,
                "Priority source": record.triage.priority_source,
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


def _render_image_detail(
    record,
    reviews: dict[str, dict[str, object]],
    all_records: list,
    next_unreviewed_id: str | None,
) -> None:
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
        if record.triage.is_experimental_model:
            st.info(
                "A local pretrained UNI encoder generated this exploratory feature map. "
                "A separate experimental MHIST agreement-proxy head used the UNI embedding "
                "for review ordering. Neither output is a diagnosis or clinical conclusion."
            )
        else:
            st.info(
                "A local pretrained UNI encoder generated this exploratory feature-variation "
                "visualization. It is not a validated clinical attention map. The "
                "deterministic rule supplied review priority for this image."
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
            st.success("Image passes the blocking MVP quality checks.")
        else:
            st.warning(f"{NEEDS_BETTER_IMAGE}: quality issues were found.")
            for reason in record.quality.reasons:
                st.write(f"• {reason}")
        for advisory in record.quality.advisories:
            st.warning(f"Manual quality advisory: {advisory}")
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
        if record.triage.is_experimental_model:
            st.caption(
                "Experimental disagreement-proxy score: "
                f"{record.triage.review_first_score:.3f}. This is not a calibrated "
                "probability or clinical confidence."
            )
            st.caption(
                "UNI supplies features; the separate experimental head generates the "
                "review-order suggestion."
            )
        elif record.attention.uses_trained_encoder:
            st.caption("The experimental priority head was not used for this label.")
        st.caption("Human Review Required • Priority is review order only.")

    st.markdown("#### Manual Review")
    review_locked = bool(reviews[record.image_id].get("reviewed", False))
    st.text_input(
        "De-identified case/slide group ID (required to mark reviewed)",
        key=f"group_{record.image_id}",
        placeholder="Example: slide-group-001",
        disabled=review_locked,
        help=(
            "Allowed: 1-64 letters, numbers, hyphens, or underscores. The app validates "
            "format only and cannot determine whether text contains identifying information."
        ),
    )
    st.button(
        "Apply this ID to ungrouped images from the same uploaded source",
        key=f"apply_group_{record.image_id}",
        on_click=_apply_group_to_source,
        args=(record, all_records),
        disabled=review_locked,
        help=(
            "Existing group IDs are never overwritten. One ZIP can contain multiple "
            "cases, so use this only when the uploaded source belongs to one case/slide."
        ),
    )
    group_message = st.session_state.get(f"group_apply_message_{record.image_id}")
    if group_message:
        st.caption(group_message)
    st.text_area(
        "Reviewer notes (kept in this browser session)",
        key=f"notes_{record.image_id}",
        placeholder="Required when overriding the suggested priority...",
        height=120,
        disabled=review_locked,
    )
    st.selectbox(
        "Confirm or override the suggested priority",
        options=PRIORITIES,
        key=f"priority_{record.image_id}",
        help="Overrides change review order only; they are not medical conclusions.",
        on_change=_mark_priority_override,
        args=(record.image_id,),
        disabled=review_locked,
    )
    _sync_selected_review(record.image_id)
    state = reviews[record.image_id]
    if bool(state.get("reviewed", False)):
        st.success("Reviewed for this Streamlit session.")
        reviewed_suggestion = state.get("suggested_priority_at_review")
        if reviewed_suggestion:
            st.caption(
                "Suggestion saved with this review: "
                f"{reviewed_suggestion}. Model-setting changes do not rewrite the "
                "completed review record."
            )
        st.button(
            "Reopen review",
            key=f"reopen_{record.image_id}",
            on_click=_reopen_review,
            args=(record.image_id,),
        )
    else:
        action_columns = st.columns(2)
        action_columns[0].button(
            "Save review",
            key=f"save_{record.image_id}",
            on_click=_save_review,
            args=(record, record.image_id),
            width="stretch",
        )
        action_columns[1].button(
            "Save & next unreviewed",
            key=f"save_next_{record.image_id}",
            on_click=_save_review,
            args=(record, next_unreviewed_id),
            disabled=next_unreviewed_id is None,
            help=(
                "No other unreviewed image matches the current queue filters."
                if next_unreviewed_id is None
                else "Save this review and open the next unreviewed image matching the current filters."
            ),
            width="stretch",
        )
    review_error = st.session_state.get(f"review_error_{record.image_id}")
    if review_error:
        st.error(review_error)
    if not record.quality.adequate:
        st.warning(
            "The image-quality warning remains active even if a reviewer chooses a "
            "different review priority."
        )


def _render_review_export(
    batch: BatchResult,
    reviews: dict[str, dict[str, object]],
    batch_fingerprint: str,
    domain_context: str,
) -> None:
    reviewed_count = sum(
        int(bool(reviews[record.image_id].get("reviewed", False)))
        for record in batch.records
    )
    with st.expander("Export reviewed labels for research training"):
        st.write(
            f"Reviewed images in the current batch: **{reviewed_count}**. The CSV "
            "contains reviewer labels and locally generated UNI embeddings when "
            "available. It does not contain raw images, filenames, or local paths."
        )
        st.warning(
            "Do not export names, medical record numbers, dates of birth, or other "
            "identifying information. Group-ID format checks cannot verify "
            "de-identification. The downloaded CSV remains local and unencrypted."
        )
        confirmation_key = f"export_deidentified_{batch_fingerprint}"
        confirmed = st.checkbox(
            "I confirm that reviewer notes and group IDs contain no identifying information.",
            key=confirmation_key,
        )
        try:
            export_data = build_review_export_csv(
                batch.records, reviews, domain_context=domain_context
            )
        except ValueError as exc:
            st.error(f"The research export could not be prepared: {exc}")
            return
        st.download_button(
            "Download reviewed labels and UNI embeddings (CSV)",
            data=export_data,
            file_name=f"pathologyai_review_labels_{batch_fingerprint[:8]}.csv",
            mime="text/csv",
            disabled=reviewed_count == 0 or not confirmed,
            help=(
                "The export includes only images marked reviewed in the current batch. "
                "It is for research/education review-priority development only."
            ),
        )
        if reviewed_count == 0:
            st.caption("Mark at least one image as reviewed before exporting.")
        elif not confirmed:
            st.caption("Confirm de-identification before downloading.")


def _render_review_queue(
    batch: BatchResult,
    reviews: dict[str, dict[str, object]],
    batch_fingerprint: str,
    domain_context: str,
) -> None:
    ordered_records = sorted(
        batch.records,
        key=lambda record: _record_sort_key(record, reviews),
    )
    reviewed_count = sum(
        int(bool(reviews[record.image_id].get("reviewed", False)))
        for record in ordered_records
    )
    total = len(ordered_records)
    progress_columns = st.columns(4)
    progress_columns[0].metric("Awaiting review", total - reviewed_count)
    progress_columns[1].metric("Reviewed", reviewed_count)
    progress_columns[2].metric(
        "Review complete", f"{(reviewed_count / total if total else 0):.0%}"
    )
    progress_columns[3].metric("Valid images", total)
    st.progress(reviewed_count / total if total else 0.0)

    filter_columns = st.columns(3)
    status_filter = filter_columns[0].selectbox(
        "Review status",
        options=("Awaiting", "Reviewed", "All"),
        key=f"queue_status_{batch_fingerprint}",
    )
    priority_filter = filter_columns[1].multiselect(
        "Review priorities",
        options=PRIORITIES,
        default=list(PRIORITIES),
        key=f"queue_priorities_{batch_fingerprint}",
    )
    groups = sorted(
        {
            str(reviews[record.image_id].get("group_id", "")).strip()
            for record in ordered_records
            if str(reviews[record.image_id].get("group_id", "")).strip()
        }
    )
    group_options = ["All groups", "Ungrouped", *groups]
    group_key = f"queue_group_{batch_fingerprint}"
    if st.session_state.get(group_key) not in group_options:
        st.session_state[group_key] = "All groups"
    group_filter = filter_columns[2].selectbox(
        "Case/slide group",
        options=group_options,
        key=group_key,
    )

    filtered_records = _filter_records(
        ordered_records,
        reviews,
        status_filter,
        priority_filter,
        group_filter,
    )
    if st.session_state.pop("queue_message", None):
        st.success("Review saved.")
    if not filtered_records:
        if total and reviewed_count == total:
            st.success("Queue complete — every valid image is marked reviewed.")
        else:
            st.info("No images match the current queue filters.")
        with st.expander("Batch results", expanded=False):
            _render_batch_table(batch, ordered_records, reviews)
        _render_review_export(
            batch, reviews, batch_fingerprint, domain_context=domain_context
        )
        return

    selection_options = [record.image_id for record in filtered_records]
    if st.session_state.get("selected_image_id") not in selection_options:
        st.session_state["selected_image_id"] = selection_options[0]
    record_by_id = {record.image_id: record for record in ordered_records}
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
    selected_index = selection_options.index(selected_id)
    navigation = st.columns((1, 1, 2, 1, 1))
    navigation[0].button(
        "Previous",
        disabled=selected_index == 0,
        key=f"previous_{batch_fingerprint}",
        on_click=_select_image,
        args=(selection_options[max(selected_index - 1, 0)],),
        width="stretch",
    )
    navigation[1].button(
        "Next",
        disabled=selected_index >= len(selection_options) - 1,
        key=f"next_{batch_fingerprint}",
        on_click=_select_image,
        args=(selection_options[min(selected_index + 1, len(selection_options) - 1)],),
        width="stretch",
    )
    navigation[2].markdown(
        f"<div style='text-align:center;padding-top:0.45rem'>Item "
        f"{selected_index + 1} of {len(selection_options)}</div>",
        unsafe_allow_html=True,
    )

    visible_after_selected = (
        filtered_records[selected_index + 1 :] + filtered_records[:selected_index]
    )
    next_unreviewed_id = next(
        (
            candidate.image_id
            for candidate in visible_after_selected
            if not bool(reviews[candidate.image_id].get("reviewed", False))
        ),
        None,
    )
    with st.expander("Batch results", expanded=False):
        _render_batch_table(batch, ordered_records, reviews)
    _render_image_detail(
        record_by_id[selected_id],
        reviews,
        ordered_records,
        next_unreviewed_id,
    )
    _render_review_export(
        batch, reviews, batch_fingerprint, domain_context=domain_context
    )


def main() -> None:
    st.title("🔬 PathologyAI")
    st.caption("Research/Education Prototype • Review Priority • Human Review Required")
    st.warning(DISCLAIMER)

    uni_status = get_uni_provider_status()
    review_model_status = get_review_model_status()
    with st.sidebar:
        st.header("Prototype Status")
        if uni_status.ready:
            st.success(uni_status.summary)
            use_uni = st.toggle(
                "Use local UNI feature visualization",
                value=True,
                help=(
                    "Loads the local ViT-L encoder when an image is processed. CPU "
                    "inference can be slow and uses substantial memory."
                ),
            )
            if use_uni:
                st.write("**Model Attention source:** Local UNI encoder")
                st.caption("Exploratory UNI feature variation is enabled.")
            else:
                st.write("**Model Attention source:** Deterministic demonstration")
                st.caption("UNI was manually disabled for this session.")
        else:
            use_uni = False
            st.warning(uni_status.summary)
            st.caption(uni_status.detail)
            st.write("**Model Attention source:** Deterministic demonstration")
        if use_uni and review_model_status.ready:
            st.success(review_model_status.summary)
            use_review_model = st.toggle(
                "Use experimental MHIST agreement-proxy head",
                value=True,
                help=(
                    "Uses the local quick prototype head with UNI features to suggest "
                    "Review First or Lower Priority. Image-quality checks remain separate."
                ),
            )
            if use_review_model:
                st.write("**Review Priority source:** Experimental local proxy head")
                st.caption(
                    "Trained on an MHIST colorectal-polyp agreement proxy; other tissues "
                    "are out of domain. This is not disease prediction or clinical urgency."
                )
            else:
                st.write("**Review Priority source:** Deterministic fallback")
            if review_model_status.metrics:
                with st.expander("Prototype head evaluation limits"):
                    metrics = review_model_status.metrics
                    st.write(
                        "Official MHIST test split: balanced accuracy "
                        f"{metrics.get('balanced_accuracy', 0.0):.3f}, ROC-AUC "
                        f"{metrics.get('roc_auc', 0.0):.3f}, Review First recall "
                        f"{metrics.get('review_first_recall', 0.0):.3f}."
                    )
                    st.caption(
                        "These measure a dataset-specific annotator-agreement proxy, not "
                        "clinical performance. MHIST lacks case/slide group IDs, so "
                        "patient-level split independence could not be verified."
                    )
        else:
            use_review_model = False
            if review_model_status.ready and not use_uni:
                st.caption(
                    "Experimental priority head disabled because it requires local UNI features."
                )
            else:
                st.warning(
                    "Experimental priority head unavailable; using deterministic "
                    "visual-complexity fallback."
                )
                st.caption(review_model_status.detail)
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
    batch_fingerprint = _batch_fingerprint(payload_values)
    st.session_state["active_batch_fingerprint"] = batch_fingerprint
    context_columns = st.columns(2)
    domain_label = context_columns[0].selectbox(
        "Batch tissue context (reviewer-declared)",
        options=tuple(DOMAIN_LABEL_TO_VALUE),
        key=f"domain_context_{batch_fingerprint}",
        help=(
            "Choose MHIST-like only when the entire batch contains comparable colorectal-"
            "polyp patches. Mixed or uncertain batches must remain Unknown or other tissue."
        ),
    )
    screening_seconds = context_columns[1].number_input(
        "Typical manual screening time per image (seconds)",
        min_value=0.0,
        max_value=600.0,
        value=30.0,
        step=5.0,
        key=f"screening_seconds_{batch_fingerprint}",
        help=(
            "Used only to estimate time avoided reviewing unusable images. It does not "
            "measure time saved by priority ranking."
        ),
    )
    domain_context = DOMAIN_LABEL_TO_VALUE[domain_label]
    with st.spinner("Checking files and preparing review-priority suggestions..."):
        selected_provider_key = (
            uni_status.cache_key if use_uni else "deterministic-demo:v1"
        )
        selected_review_model_key = (
            review_model_status.cache_key
            if use_review_model
            else "review-head-disabled:v1"
        )
        batch = _process_cached(
            payload_values,
            selected_provider_key,
            use_uni,
            selected_review_model_key,
            use_review_model,
        )

    reviews = _initialize_review_state(batch)
    review_tab, operations_tab, evaluation_tab = st.tabs(
        ["Review Queue", "Operational Dashboard", "Model Evaluation & Limits"]
    )
    with review_tab:
        if not batch.records:
            st.error(
                "No valid images were available for review. Check the skipped-file reasons "
                "and upload a supported, readable image."
            )
            _render_skipped_files(batch)
        else:
            _render_review_queue(
                batch,
                reviews,
                batch_fingerprint,
                domain_context=domain_context,
            )
    with operations_tab:
        _render_operational_dashboard(
            batch,
            reviews,
            domain_label=domain_label,
            screening_seconds=screening_seconds,
            use_uni=use_uni,
            use_review_model=use_review_model,
        )
    with evaluation_tab:
        _render_model_evaluation(review_model_status)

    st.divider()
    st.caption("PathologyAI • Research/Education Prototype • Human Review Required")


if __name__ == "__main__":
    main()
