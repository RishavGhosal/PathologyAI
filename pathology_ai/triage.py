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


@dataclass(frozen=True)
class TriageResult:
    suggested_priority: str
    explanation: str
    priority_source: str = "Deterministic visual-complexity heuristic"


def assign_review_priority(
    quality: QualityAssessment,
    visual_complexity_score: float,
    priority_source: str = "Deterministic visual-complexity heuristic",
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
        )

    return TriageResult(
        suggested_priority=LOWER_PRIORITY,
        explanation=(
            "The image passed the quality checks and has less visual variation under the "
            "deterministic rule. It remains in the queue and still requires human review."
        ),
        priority_source=priority_source,
    )


def priority_sort_key(priority: str) -> int:
    return PRIORITY_ORDER.get(priority, len(PRIORITY_ORDER))
