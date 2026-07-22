"""De-identified reviewer-label export for future research model training."""

from __future__ import annotations

import csv
from io import StringIO
import math
import re
from typing import Any, Mapping

from .triage import PRIORITIES
from .uni_provider import UNI_EMBEDDING_DIMENSION


EMBEDDING_COLUMNS = tuple(
    f"uni_{index:04d}" for index in range(UNI_EMBEDDING_DIMENSION)
)
EXPORT_COLUMNS = (
    "image_id",
    "group_id",
    "group_id_format_validated",
    "suggested_priority",
    "reviewer_priority",
    "reviewed",
    "reviewed_at_utc",
    "reviewer_notes",
    "priority_overridden",
    "priority_source",
    "priority_method",
    "priority_fallback_reason",
    "review_first_proxy_score",
    "domain_context",
    "quality_adequate",
    "quality_issue_codes",
    "quality_advisory_codes",
    "quality_reasons",
    "quality_advisories",
    "attention_provider",
    "embedding_available",
    "embedding_model",
    "embedding_dimension",
    "intended_use",
    "human_review_required",
) + EMBEDDING_COLUMNS

GROUP_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
DOMAIN_CONTEXTS = {
    "unknown_or_other",
    "mhist_like_colorectal_polyp",
}


def _safe_text(value: object) -> str:
    """Prevent spreadsheet formula execution for user-controlled text cells."""

    text = str(value or "")
    if text.startswith(("=", "+", "-", "@", "\t", "\r")):
        return f"'{text}"
    return text


def validate_group_id(value: object) -> str:
    """Validate format only; this cannot determine whether text contains PHI."""

    group_id = str(value or "").strip()
    if not group_id:
        raise ValueError("A de-identified case/slide group ID is required.")
    if not GROUP_ID_PATTERN.fullmatch(group_id):
        raise ValueError(
            "Group ID must be 1-64 characters, start with a letter or number, and "
            "contain only letters, numbers, hyphens, or underscores."
        )
    return group_id


def validate_optional_group_id(value: object) -> str:
    """Validate a grouping value when supplied, without blocking review export.

    A group ID is needed by the later grouped-training preparation workflow, but
    an ordinary reviewed-label export is still useful without one.  Keeping the
    blank value explicit lets that downstream workflow refuse ungrouped rows
    rather than making the user re-review an entire browser session.
    """

    group_id = str(value or "").strip()
    return validate_group_id(group_id) if group_id else ""


def validate_review_fields(record: Any, review: Mapping[str, object]) -> None:
    """Enforce fields required for reviewed training feedback."""

    validate_optional_group_id(review.get("group_id", ""))
    reviewer_priority = str(review.get("priority", ""))
    if reviewer_priority not in PRIORITIES:
        raise ValueError("Reviewer priority is not one of the three allowed labels.")
    suggested_priority = str(
        review.get("suggested_priority_at_review")
        or record.triage.suggested_priority
    )
    if (
        reviewer_priority != suggested_priority
        and not str(review.get("notes", "")).strip()
    ):
        raise ValueError("Reviewer notes are required when overriding the suggestion.")


def _validated_embedding(
    record: Any, review: Mapping[str, object]
) -> tuple[float, ...] | None:
    embedding = (
        review.get("embedding_at_review")
        if "embedding_at_review" in review
        else record.attention.embedding
    )
    if embedding is None:
        return None
    embedding_model = str(
        review.get("embedding_model_at_review")
        if "embedding_model_at_review" in review
        else getattr(record.attention, "embedding_model", "") or ""
    )
    if embedding_model == "MahmoodLab/UNI" and len(embedding) != UNI_EMBEDDING_DIMENSION:
        raise ValueError(
            f"UNI embedding for {record.image_id[:12]} has an unexpected dimension."
        )
    values = tuple(float(value) for value in embedding)
    if len(values) < 2:
        raise ValueError(
            f"Model embedding for {record.image_id[:12]} has an unexpected dimension."
        )
    if not all(math.isfinite(value) for value in values):
        raise ValueError(
            f"Model embedding for {record.image_id[:12]} contains non-finite values."
        )
    return values


def build_review_export_csv(
    records: list[Any],
    reviews: Mapping[str, Mapping[str, object]],
    domain_context: str = "unknown_or_other",
) -> bytes:
    """Export reviewed current-batch rows only; never export images or filenames."""

    if domain_context not in DOMAIN_CONTEXTS:
        raise ValueError("Domain context is not one of the supported declarations.")
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=EXPORT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for record in records:
        review = reviews.get(record.image_id)
        if not review or not bool(review.get("reviewed", False)):
            continue
        validate_review_fields(record, review)
        reviewer_priority = str(review.get("priority", ""))
        group_id = validate_optional_group_id(review.get("group_id", ""))
        suggested_priority = str(
            review.get("suggested_priority_at_review")
            or record.triage.suggested_priority
        )
        priority_overridden = reviewer_priority != suggested_priority
        embedding = _validated_embedding(record, review)
        proxy_score = review.get(
            "review_first_proxy_score_at_review",
            record.triage.review_first_score,
        )
        if proxy_score is not None:
            proxy_score = float(proxy_score)
            if not math.isfinite(proxy_score):
                raise ValueError(
                    f"Proxy score for {record.image_id[:12]} is not finite."
                )
        row: dict[str, object] = {
            "image_id": record.image_id,
            "group_id": _safe_text(group_id),
            "group_id_format_validated": str(bool(group_id)),
            "suggested_priority": suggested_priority,
            "reviewer_priority": reviewer_priority,
            "reviewed": "True",
            "reviewed_at_utc": _safe_text(review.get("reviewed_at_utc", "")),
            "reviewer_notes": _safe_text(review.get("notes", "")),
            "priority_overridden": str(priority_overridden),
            "priority_source": review.get(
                "priority_source_at_review", record.triage.priority_source
            ),
            "priority_method": review.get(
                "priority_method_at_review",
                getattr(record.triage, "priority_method", "deterministic"),
            ),
            "priority_fallback_reason": review.get(
                "priority_fallback_reason_at_review",
                getattr(record.triage, "fallback_reason", None) or "",
            ),
            "review_first_proxy_score": (
                ""
                if proxy_score is None
                else format(proxy_score, ".9g")
            ),
            "domain_context": domain_context,
            "quality_adequate": str(bool(record.quality.adequate)),
            "quality_issue_codes": " | ".join(
                getattr(record.quality, "issue_codes", ()) or ()
            ),
            "quality_advisory_codes": " | ".join(
                getattr(record.quality, "advisory_codes", ()) or ()
            ),
            "quality_reasons": _safe_text(" | ".join(record.quality.reasons)),
            "quality_advisories": _safe_text(" | ".join(record.quality.advisories)),
            "attention_provider": review.get(
                "attention_provider_at_review", record.attention.provider_name
            ),
            "embedding_available": str(embedding is not None),
            "embedding_model": review.get(
                "embedding_model_at_review", record.attention.embedding_model or ""
            ),
            "embedding_dimension": len(embedding) if embedding is not None else "",
            "intended_use": "research_education_review_priority_only",
            "human_review_required": "True",
        }
        for index, column in enumerate(EMBEDDING_COLUMNS):
            row[column] = (
                ""
                if embedding is None or len(embedding) != UNI_EMBEDDING_DIMENSION
                else format(embedding[index], ".9g")
            )
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")
