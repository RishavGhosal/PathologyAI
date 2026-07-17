"""Pure operational metrics for the PathologyAI review workspace.

This module intentionally contains no Streamlit code.  It turns a processed
``BatchResult`` and the session's reviewer state into one immutable summary so
the dashboard can render counts without reimplementing business rules.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Mapping

from .triage import LOWER_PRIORITY, NEEDS_BETTER_IMAGE, PRIORITIES, REVIEW_FIRST

if TYPE_CHECKING:
    from .pipeline import BatchResult


MHIST_LIKE_DOMAIN = "MHIST-like colorectal-polyp patches"
DEFAULT_SCREENING_SECONDS_PER_IMAGE = 30.0
UNI_EMBEDDING_DIMENSION = 1024


@dataclass(frozen=True)
class PriorityAgreementRow:
    """Reviewer agreement for one originally suggested priority."""

    suggested_priority: str
    reviewed_count: int
    confirmed_count: int
    overridden_count: int
    agreement_percentage: float | None


@dataclass(frozen=True)
class OperationalMetrics:
    """Dashboard-ready operational values for the current batch.

    Percentage fields are expressed from 0 to 100.  Agreement percentages are
    ``None`` when there are no eligible reviewed records; this distinguishes
    "not measured" from zero agreement.
    """

    total_images: int
    skipped_count: int
    awaiting_count: int
    reviewed_count: int
    reviewed_percentage: float
    effective_priority_counts: dict[str, int]

    quality_pass_count: int
    quality_issue_counts: dict[str, int]
    quality_advisory_counts: dict[str, int]

    embedding_success_count: int
    experimental_model_prediction_count: int
    deterministic_prediction_count: int
    quality_gate_count: int
    runtime_fallback_count: int

    proxy_scores: tuple[float, ...]
    proxy_score_count: int
    mean_proxy_score: float | None

    reviewed_with_priority_count: int
    suggestion_confirmed_count: int
    suggestion_overridden_count: int
    suggestion_agreement_percentage: float | None

    model_reviewed_count: int
    model_confirmed_count: int
    model_overridden_count: int
    model_agreement_percentage: float | None
    agreement_by_suggested_priority: tuple[PriorityAgreementRow, ...]

    domain_declaration: str
    domain_warning_count: int
    screening_seconds_per_image: float
    estimated_time_avoided_seconds: float


def _percentage(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return (float(numerator) / float(denominator)) * 100.0


def _codes_from(quality: Any, names: tuple[str, ...]) -> tuple[str, ...]:
    """Return the first available structured code collection.

    Older ``QualityAssessment`` instances do not have code fields.  They yield
    an empty tuple rather than forcing the dashboard to parse human-readable
    prose.  Multiple aliases are accepted during the backwards-compatible
    transition to stable quality codes.
    """

    for name in names:
        if not hasattr(quality, name):
            continue
        raw_codes = getattr(quality, name)
        if raw_codes is None:
            return ()
        if isinstance(raw_codes, str):
            raw_codes = (raw_codes,)
        try:
            cleaned = {
                str(code).strip()
                for code in raw_codes
                if code is not None and str(code).strip()
            }
        except TypeError:
            return ()
        return tuple(sorted(cleaned))
    return ()


def _valid_uni_embedding(record: Any) -> bool:
    attention = getattr(record, "attention", None)
    embedding = getattr(attention, "embedding", None)
    if embedding is None:
        embedding = getattr(record, "embedding", None)
    if embedding is None:
        return False
    try:
        values = tuple(float(value) for value in embedding)
    except (TypeError, ValueError, OverflowError):
        return False
    return len(values) == UNI_EMBEDDING_DIMENSION and all(
        math.isfinite(value) for value in values
    )


def _normalized_method_kind(
    raw_method: object | None,
    suggested: object,
    *,
    experimental_hint: bool = False,
) -> str:
    if suggested == NEEDS_BETTER_IMAGE:
        return "quality"
    normalized = str(raw_method or "").strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized:
        if "quality" in normalized or normalized == "quality_gate":
            return "quality"
        if any(token in normalized for token in ("experimental", "review_head", "model_head")):
            return "experimental"
        if "deterministic" in normalized:
            return "deterministic"

    if experimental_hint:
        return "experimental"
    return "deterministic"


def _method_kind(record: Any) -> str:
    """Classify one priority outcome as quality, experimental, or deterministic."""

    triage = getattr(record, "triage", None)
    raw_method: object | None = None
    for name in ("priority_method", "method"):
        if hasattr(triage, name):
            raw_method = getattr(triage, name)
            break
    return _normalized_method_kind(
        raw_method,
        getattr(triage, "suggested_priority", None),
        experimental_hint=bool(getattr(triage, "is_experimental_model", False)),
    )


def _has_runtime_fallback(record: Any) -> bool:
    """Prefer structured fallback state, with a narrow legacy-note fallback."""

    structured_found = False
    for owner in (record, getattr(record, "triage", None), getattr(record, "attention", None)):
        if owner is None or not hasattr(owner, "fallback_reason"):
            continue
        structured_found = True
        reason = getattr(owner, "fallback_reason")
        if reason is not None and str(reason).strip():
            return True
    if structured_found:
        return False

    notes = getattr(record, "metadata_notes", ()) or ()
    if isinstance(notes, str):
        notes = (notes,)
    for note in notes:
        normalized = str(note).casefold()
        if "fallback" in normalized and any(
            marker in normalized
            for marker in ("could not", "unavailable", "failed", "fallback was used")
        ):
            return True
    return False


def _finite_proxy_score(record: Any) -> float | None:
    triage = getattr(record, "triage", None)
    value = getattr(triage, "review_first_score", None)
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return score if math.isfinite(score) else None


def build_operational_metrics(
    batch: "BatchResult",
    reviews: Mapping[str, Mapping[str, object]],
    domain_declaration: str = "Unknown or other tissue",
    screening_seconds_per_image: float = DEFAULT_SCREENING_SECONDS_PER_IMAGE,
) -> OperationalMetrics:
    """Calculate review-workflow metrics for a processed image batch.

    ``reviews`` is keyed by stable image ID.  The time estimate deliberately
    covers only records whose *effective* priority is ``Needs Better Image``
    plus skipped/failed inputs; it makes no claim that priority ranking saves
    review time.
    """

    try:
        seconds = float(screening_seconds_per_image)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Screening seconds per image must be a finite non-negative number.") from exc
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError("Screening seconds per image must be a finite non-negative number.")

    records = list(getattr(batch, "records", ()) or ())
    skipped_count = len(getattr(batch, "skipped", ()) or ())
    effective_counts = {priority: 0 for priority in PRIORITIES}
    issue_counts: Counter[str] = Counter()
    advisory_counts: Counter[str] = Counter()
    quality_pass_count = 0
    embedding_success_count = 0
    method_counts = Counter({"experimental": 0, "deterministic": 0, "quality": 0})
    runtime_fallback_count = 0
    proxy_scores: list[float] = []

    reviewed_count = 0
    reviewed_with_priority_count = 0
    suggestion_confirmed_count = 0
    suggestion_overridden_count = 0
    model_reviewed_count = 0
    model_confirmed_count = 0
    model_overridden_count = 0
    per_priority = {
        priority: {"reviewed": 0, "confirmed": 0, "overridden": 0}
        for priority in PRIORITIES
    }

    for record in records:
        image_id = str(getattr(record, "image_id", ""))
        review = reviews.get(image_id, {}) or {}
        triage = getattr(record, "triage", None)
        suggested = str(getattr(triage, "suggested_priority", ""))
        selected = str(review.get("priority", ""))
        effective = selected if selected in PRIORITIES else suggested
        if effective in effective_counts:
            effective_counts[effective] += 1

        quality = getattr(record, "quality", None)
        if bool(getattr(quality, "adequate", False)):
            quality_pass_count += 1
        issue_counts.update(
            _codes_from(quality, ("blocking_codes", "issue_codes", "reason_codes"))
        )
        advisory_counts.update(_codes_from(quality, ("advisory_codes",)))

        if _valid_uni_embedding(record):
            embedding_success_count += 1
        kind = _method_kind(record)
        method_counts[kind] += 1
        if _has_runtime_fallback(record):
            runtime_fallback_count += 1
        score = _finite_proxy_score(record)
        if score is not None:
            proxy_scores.append(score)

        is_reviewed = bool(review.get("reviewed", False))
        if not is_reviewed:
            continue
        reviewed_count += 1
        reviewed_suggestion = str(
            review.get("suggested_priority_at_review") or suggested
        )
        if selected not in PRIORITIES or reviewed_suggestion not in PRIORITIES:
            continue

        reviewed_with_priority_count += 1
        confirmed = selected == reviewed_suggestion
        if confirmed:
            suggestion_confirmed_count += 1
        else:
            suggestion_overridden_count += 1
        bucket = per_priority[reviewed_suggestion]
        bucket["reviewed"] += 1
        bucket["confirmed" if confirmed else "overridden"] += 1

        reviewed_kind = (
            _normalized_method_kind(
                review.get("priority_method_at_review"),
                reviewed_suggestion,
            )
            if "priority_method_at_review" in review
            else kind
        )
        if reviewed_kind == "experimental":
            model_reviewed_count += 1
            if confirmed:
                model_confirmed_count += 1
            else:
                model_overridden_count += 1

    total_images = len(records)
    awaiting_count = total_images - reviewed_count
    reviewed_percentage = _percentage(reviewed_count, total_images) or 0.0
    mean_proxy_score = (
        sum(proxy_scores) / len(proxy_scores) if proxy_scores else None
    )
    agreement_rows = tuple(
        PriorityAgreementRow(
            suggested_priority=priority,
            reviewed_count=per_priority[priority]["reviewed"],
            confirmed_count=per_priority[priority]["confirmed"],
            overridden_count=per_priority[priority]["overridden"],
            agreement_percentage=_percentage(
                per_priority[priority]["confirmed"],
                per_priority[priority]["reviewed"],
            ),
        )
        for priority in PRIORITIES
    )

    declaration = str(domain_declaration or "Unknown or other tissue").strip()
    is_mhist_like = declaration.casefold() == MHIST_LIKE_DOMAIN.casefold()
    domain_warning_count = (
        0 if is_mhist_like else int(method_counts["experimental"])
    )
    needs_better_count = effective_counts[NEEDS_BETTER_IMAGE]

    return OperationalMetrics(
        total_images=total_images,
        skipped_count=skipped_count,
        awaiting_count=awaiting_count,
        reviewed_count=reviewed_count,
        reviewed_percentage=reviewed_percentage,
        effective_priority_counts=effective_counts,
        quality_pass_count=quality_pass_count,
        quality_issue_counts=dict(sorted(issue_counts.items())),
        quality_advisory_counts=dict(sorted(advisory_counts.items())),
        embedding_success_count=embedding_success_count,
        experimental_model_prediction_count=int(method_counts["experimental"]),
        deterministic_prediction_count=int(method_counts["deterministic"]),
        quality_gate_count=int(method_counts["quality"]),
        runtime_fallback_count=runtime_fallback_count,
        proxy_scores=tuple(proxy_scores),
        proxy_score_count=len(proxy_scores),
        mean_proxy_score=mean_proxy_score,
        reviewed_with_priority_count=reviewed_with_priority_count,
        suggestion_confirmed_count=suggestion_confirmed_count,
        suggestion_overridden_count=suggestion_overridden_count,
        suggestion_agreement_percentage=_percentage(
            suggestion_confirmed_count, reviewed_with_priority_count
        ),
        model_reviewed_count=model_reviewed_count,
        model_confirmed_count=model_confirmed_count,
        model_overridden_count=model_overridden_count,
        model_agreement_percentage=_percentage(
            model_confirmed_count, model_reviewed_count
        ),
        agreement_by_suggested_priority=agreement_rows,
        domain_declaration=declaration,
        domain_warning_count=domain_warning_count,
        screening_seconds_per_image=seconds,
        estimated_time_avoided_seconds=(needs_better_count + skipped_count) * seconds,
    )


__all__ = [
    "DEFAULT_SCREENING_SECONDS_PER_IMAGE",
    "MHIST_LIKE_DOMAIN",
    "OperationalMetrics",
    "PriorityAgreementRow",
    "build_operational_metrics",
]
