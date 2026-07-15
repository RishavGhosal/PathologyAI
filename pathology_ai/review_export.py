"""De-identified reviewer-label export for future research model training."""

from __future__ import annotations

import csv
from io import StringIO
import math
from typing import Any, Mapping

from .triage import PRIORITIES
from .uni_provider import UNI_EMBEDDING_DIMENSION


EMBEDDING_COLUMNS = tuple(
    f"uni_{index:04d}" for index in range(UNI_EMBEDDING_DIMENSION)
)
EXPORT_COLUMNS = (
    "image_id",
    "group_id",
    "suggested_priority",
    "reviewer_priority",
    "reviewed",
    "reviewer_notes",
    "quality_adequate",
    "quality_reasons",
    "quality_advisories",
    "attention_provider",
    "embedding_available",
    "embedding_model",
    "embedding_dimension",
    "intended_use",
    "human_review_required",
) + EMBEDDING_COLUMNS


def _safe_text(value: object) -> str:
    """Prevent spreadsheet formula execution for user-controlled text cells."""

    text = str(value or "")
    if text.startswith(("=", "+", "-", "@", "\t", "\r")):
        return f"'{text}"
    return text


def _validated_embedding(record: Any) -> tuple[float, ...] | None:
    embedding = record.attention.embedding
    if embedding is None:
        return None
    if len(embedding) != UNI_EMBEDDING_DIMENSION:
        raise ValueError(
            f"UNI embedding for {record.image_id[:12]} has an unexpected dimension."
        )
    values = tuple(float(value) for value in embedding)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(
            f"UNI embedding for {record.image_id[:12]} contains non-finite values."
        )
    return values


def build_review_export_csv(
    records: list[Any],
    reviews: Mapping[str, Mapping[str, object]],
) -> bytes:
    """Export reviewed current-batch rows only; never export images or filenames."""

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=EXPORT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for record in records:
        review = reviews.get(record.image_id)
        if not review or not bool(review.get("reviewed", False)):
            continue
        reviewer_priority = str(review.get("priority", ""))
        if reviewer_priority not in PRIORITIES:
            raise ValueError("Reviewer priority is not one of the three allowed labels.")
        embedding = _validated_embedding(record)
        row: dict[str, object] = {
            "image_id": record.image_id,
            "group_id": _safe_text(review.get("group_id", "")),
            "suggested_priority": record.triage.suggested_priority,
            "reviewer_priority": reviewer_priority,
            "reviewed": "True",
            "reviewer_notes": _safe_text(review.get("notes", "")),
            "quality_adequate": str(bool(record.quality.adequate)),
            "quality_reasons": _safe_text(" | ".join(record.quality.reasons)),
            "quality_advisories": _safe_text(" | ".join(record.quality.advisories)),
            "attention_provider": record.attention.provider_name,
            "embedding_available": str(embedding is not None),
            "embedding_model": record.attention.embedding_model or "",
            "embedding_dimension": len(embedding) if embedding is not None else "",
            "intended_use": "research_education_review_priority_only",
            "human_review_required": "True",
        }
        for index, column in enumerate(EMBEDDING_COLUMNS):
            row[column] = "" if embedding is None else format(embedding[index], ".9g")
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")
