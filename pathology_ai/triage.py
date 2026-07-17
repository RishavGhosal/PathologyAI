"""Review-order labels for the research/education prototype."""

from __future__ import annotations

from dataclasses import dataclass

from .quality import QualityAssessment


REVIEW_FIRST = "Review First"
NEEDS_BETTER_IMAGE = "Needs Better Image"
LOWER_PRIORITY = "Lower Priority"
PRIORITIES = (REVIEW_FIRST, NEEDS_BETTER_IMAGE, LOWER_PRIORITY)
PRIORITY_ORDER = {label: index for index, label in enumerate(PRIORITIES)}
REVIEW_FIRST_COMPLEXITY_THRESHOLD = 0.46
PRIORITY_METHOD_QUALITY_GATE = "quality_gate"
PRIORITY_METHOD_EXPERIMENTAL_HEAD = "experimental_head"
PRIORITY_METHOD_DETERMINISTIC = "deterministic"
PRIORITY_METHODS = (
    PRIORITY_METHOD_QUALITY_GATE,
    PRIORITY_METHOD_EXPERIMENTAL_HEAD,
    PRIORITY_METHOD_DETERMINISTIC,
)


@dataclass(frozen=True)
class TriageResult:
    suggested_priority: str
    explanation: str
    priority_source: str = "Deterministic visual-complexity heuristic"
    review_first_score: float | None = None
    is_experimental_model: bool = False
    priority_method: str = PRIORITY_METHOD_DETERMINISTIC
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if self.priority_method not in PRIORITY_METHODS:
            raise ValueError(f"Unsupported priority method: {self.priority_method!r}.")
        if self.fallback_reason is not None and not self.fallback_reason.strip():
            raise ValueError("Fallback reason must be non-empty when provided.")


def assign_review_priority(
    quality: QualityAssessment,
    visual_complexity_score: float,
    priority_source: str = "Deterministic visual-complexity heuristic",
    experimental_priority: str | None = None,
    review_first_score: float | None = None,
    fallback_reason: str | None = None,
) -> TriageResult:
    """Assign review order only; never a disease or diagnostic category."""

    if not quality.adequate:
        joined_reasons = " ".join(quality.reasons)
        return TriageResult(
            suggested_priority=NEEDS_BETTER_IMAGE,
            explanation=(
                f"A clearer or more complete image is needed before useful review. "
                f"{joined_reasons}"
            ),
            priority_source="Image-quality checks",
            priority_method=PRIORITY_METHOD_QUALITY_GATE,
            fallback_reason=fallback_reason,
        )

    if experimental_priority is not None:
        if experimental_priority not in (REVIEW_FIRST, LOWER_PRIORITY):
            raise ValueError("Experimental head may only suggest the two review-order labels.")
        if review_first_score is None or not 0.0 <= review_first_score <= 1.0:
            raise ValueError("Experimental head score must be between zero and one.")
        direction = "earlier" if experimental_priority == REVIEW_FIRST else "later"
        return TriageResult(
            suggested_priority=experimental_priority,
            explanation=(
                "This experimental head compares the UNI embedding with MHIST colorectal-"
                "polyp image examples that had lower versus higher agreement among seven "
                f"annotators, placing the image {direction} in the review queue. It predicts "
                "a dataset-specific agreement proxy and may not transfer to other tissues; "
                "it does not predict disease, diagnosis, or clinical urgency."
            ),
            priority_source=priority_source,
            review_first_score=review_first_score,
            is_experimental_model=True,
            priority_method=PRIORITY_METHOD_EXPERIMENTAL_HEAD,
        )

    if visual_complexity_score >= REVIEW_FIRST_COMPLEXITY_THRESHOLD:
        return TriageResult(
            suggested_priority=REVIEW_FIRST,
            explanation=(
                "The deterministic visual-complexity rule found comparatively strong "
                "texture or contrast, so it places this image earlier in the human-review "
                "queue. This is a review-order suggestion, not a disease estimate."
            ),
            priority_source=priority_source,
            priority_method=PRIORITY_METHOD_DETERMINISTIC,
            fallback_reason=fallback_reason,
        )

    return TriageResult(
        suggested_priority=LOWER_PRIORITY,
        explanation=(
            "The image passed the blocking image-quality checks and has less visual "
            "variation under the deterministic rule. It remains in the queue and still "
            "requires human review."
        ),
        priority_source=priority_source,
        priority_method=PRIORITY_METHOD_DETERMINISTIC,
        fallback_reason=fallback_reason,
    )


def priority_sort_key(priority: str) -> int:
    return PRIORITY_ORDER.get(priority, len(PRIORITY_ORDER))
