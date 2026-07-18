"""Tests for pure operational dashboard calculations."""

from __future__ import annotations

from types import SimpleNamespace
import math
import unittest

from pathology_ai.dashboard_metrics import (
    MHIST_LIKE_DOMAIN,
    UNKNOWN_OR_OTHER_DOMAIN,
    build_operational_metrics,
)
from pathology_ai.triage import LOWER_PRIORITY, NEEDS_BETTER_IMAGE, REVIEW_FIRST


def _record(
    image_id: str,
    suggested: str,
    *,
    adequate: bool = True,
    issue_codes: tuple[str, ...] | None = (),
    advisory_codes: tuple[str, ...] | None = (),
    experimental: bool = False,
    score: float | None = None,
    embedding: tuple[float, ...] | None = None,
    notes: tuple[str, ...] = (),
    priority_method: str | None = None,
    fallback_reason: object = ...,
) -> SimpleNamespace:
    quality_values: dict[str, object] = {"adequate": adequate}
    if issue_codes is not None:
        quality_values["issue_codes"] = issue_codes
    if advisory_codes is not None:
        quality_values["advisory_codes"] = advisory_codes
    triage_values: dict[str, object] = {
        "suggested_priority": suggested,
        "is_experimental_model": experimental,
        "review_first_score": score,
    }
    if priority_method is not None:
        triage_values["priority_method"] = priority_method
    if fallback_reason is not ...:
        triage_values["fallback_reason"] = fallback_reason
    return SimpleNamespace(
        image_id=image_id,
        quality=SimpleNamespace(**quality_values),
        attention=SimpleNamespace(embedding=embedding),
        triage=SimpleNamespace(**triage_values),
        metadata_notes=notes,
    )


def _batch(records: list[SimpleNamespace], skipped: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        records=records,
        skipped=[SimpleNamespace(reason="skipped") for _ in range(skipped)],
    )


def _review(priority: str, reviewed: bool = True) -> dict[str, object]:
    return {"priority": priority, "reviewed": reviewed}


class OperationalMetricsTests(unittest.TestCase):
    def test_complete_mixed_batch_metrics(self) -> None:
        embedding = tuple([0.25] * 1024)
        records = [
            _record(
                "model-confirmed",
                REVIEW_FIRST,
                experimental=True,
                score=0.8,
                embedding=embedding,
                advisory_codes=("possible_edge_truncation",),
            ),
            _record(
                "model-overridden",
                LOWER_PRIORITY,
                experimental=True,
                score=0.2,
                embedding=embedding,
            ),
            _record(
                "deterministic-awaiting",
                LOWER_PRIORITY,
                notes=(
                    "The experimental priority head was unavailable; the deterministic "
                    "review-priority fallback was used.",
                ),
            ),
            _record(
                "quality-confirmed",
                NEEDS_BETTER_IMAGE,
                adequate=False,
                issue_codes=("blur", "small_dimensions"),
            ),
        ]
        reviews = {
            "model-confirmed": _review(REVIEW_FIRST),
            "model-overridden": _review(REVIEW_FIRST),
            "deterministic-awaiting": _review(LOWER_PRIORITY, reviewed=False),
            "quality-confirmed": _review(NEEDS_BETTER_IMAGE),
        }

        metrics = build_operational_metrics(
            _batch(records, skipped=2),
            reviews,
            domain_declaration="Unknown or other tissue",
            screening_seconds_per_image=30,
        )

        self.assertEqual(metrics.total_images, 4)
        self.assertEqual(metrics.skipped_count, 2)
        self.assertEqual(metrics.reviewed_count, 3)
        self.assertEqual(metrics.awaiting_count, 1)
        self.assertEqual(metrics.reviewed_percentage, 75.0)
        self.assertEqual(
            metrics.effective_priority_counts,
            {REVIEW_FIRST: 2, NEEDS_BETTER_IMAGE: 1, LOWER_PRIORITY: 1},
        )
        self.assertEqual(metrics.quality_pass_count, 3)
        self.assertEqual(
            metrics.quality_issue_counts,
            {"blur": 1, "small_dimensions": 1},
        )
        self.assertEqual(
            metrics.quality_advisory_counts,
            {"possible_edge_truncation": 1},
        )
        self.assertEqual(metrics.embedding_success_count, 2)
        self.assertEqual(metrics.embedding_failure_count, 2)
        self.assertEqual(metrics.embedding_not_attempted_count, 0)
        self.assertEqual(metrics.experimental_model_prediction_count, 2)
        self.assertEqual(metrics.deterministic_prediction_count, 1)
        self.assertEqual(metrics.deterministic_fallback_prediction_count, 1)
        self.assertEqual(metrics.quality_gate_count, 1)
        self.assertEqual(metrics.runtime_fallback_count, 1)
        self.assertEqual(metrics.proxy_scores, (0.8, 0.2))
        self.assertEqual(metrics.proxy_score_count, 2)
        self.assertAlmostEqual(metrics.mean_proxy_score or 0.0, 0.5)

        self.assertEqual(metrics.reviewed_with_priority_count, 3)
        self.assertEqual(metrics.suggestion_confirmed_count, 2)
        self.assertEqual(metrics.suggestion_overridden_count, 1)
        self.assertAlmostEqual(metrics.suggestion_agreement_percentage or 0.0, 200 / 3)
        self.assertEqual(metrics.model_reviewed_count, 2)
        self.assertEqual(metrics.model_confirmed_count, 1)
        self.assertEqual(metrics.model_overridden_count, 1)
        self.assertEqual(metrics.model_agreement_percentage, 50.0)
        rows = {row.suggested_priority: row for row in metrics.agreement_by_suggested_priority}
        self.assertEqual(rows[REVIEW_FIRST].agreement_percentage, 100.0)
        self.assertEqual(rows[LOWER_PRIORITY].agreement_percentage, 0.0)
        self.assertEqual(rows[NEEDS_BETTER_IMAGE].agreement_percentage, 100.0)
        model_rows = {
            row.suggested_priority: row
            for row in metrics.model_agreement_by_suggested_priority
        }
        self.assertEqual(model_rows[REVIEW_FIRST].agreement_percentage, 100.0)
        self.assertEqual(model_rows[LOWER_PRIORITY].agreement_percentage, 0.0)
        self.assertIsNone(model_rows[NEEDS_BETTER_IMAGE].agreement_percentage)

        self.assertEqual(metrics.domain_warning_count, 2)
        self.assertEqual(
            metrics.domain_declaration_counts,
            {UNKNOWN_OR_OTHER_DOMAIN: 4, MHIST_LIKE_DOMAIN: 0},
        )
        self.assertEqual(metrics.estimated_time_avoided_seconds, 90.0)

    def test_empty_batch_has_zero_progress_and_unmeasured_agreement(self) -> None:
        metrics = build_operational_metrics(_batch([]), {})

        self.assertEqual(metrics.reviewed_percentage, 0.0)
        self.assertEqual(metrics.awaiting_count, 0)
        self.assertIsNone(metrics.mean_proxy_score)
        self.assertIsNone(metrics.suggestion_agreement_percentage)
        self.assertIsNone(metrics.model_agreement_percentage)
        self.assertTrue(
            all(row.agreement_percentage is None for row in metrics.agreement_by_suggested_priority)
        )
        self.assertEqual(
            metrics.domain_declaration_counts,
            {UNKNOWN_OR_OTHER_DOMAIN: 0, MHIST_LIKE_DOMAIN: 0},
        )

    def test_mhist_like_declaration_suppresses_domain_warnings(self) -> None:
        record = _record("model", REVIEW_FIRST, experimental=True, score=0.7)
        metrics = build_operational_metrics(
            _batch([record]),
            {},
            domain_declaration=f"  {MHIST_LIKE_DOMAIN.upper()}  ",
        )

        self.assertEqual(metrics.experimental_model_prediction_count, 1)
        self.assertEqual(metrics.domain_warning_count, 0)
        self.assertEqual(
            metrics.domain_declaration_counts,
            {UNKNOWN_OR_OTHER_DOMAIN: 0, MHIST_LIKE_DOMAIN: 1},
        )

    def test_uni_disabled_is_not_reported_as_embedding_failure(self) -> None:
        record = _record("not-attempted", LOWER_PRIORITY)

        metrics = build_operational_metrics(
            _batch([record]),
            {},
            embedding_expected=False,
        )

        self.assertEqual(metrics.embedding_success_count, 0)
        self.assertEqual(metrics.embedding_failure_count, 0)
        self.assertEqual(metrics.embedding_not_attempted_count, 1)

    def test_structured_fallback_field_takes_precedence_over_legacy_notes(self) -> None:
        legacy_note = (
            "The local provider could not process the image; deterministic fallback was used."
        )
        records = [
            _record(
                "no-fallback",
                LOWER_PRIORITY,
                fallback_reason=None,
                notes=(legacy_note,),
            ),
            _record(
                "structured-fallback",
                LOWER_PRIORITY,
                fallback_reason="review_head_inference_failed",
            ),
        ]

        metrics = build_operational_metrics(_batch(records), {})

        self.assertEqual(metrics.runtime_fallback_count, 1)
        self.assertEqual(metrics.deterministic_fallback_prediction_count, 1)

    def test_priority_method_is_used_when_modern_structured_field_exists(self) -> None:
        records = [
            _record(
                "head",
                LOWER_PRIORITY,
                priority_method="experimental_head",
            ),
            _record(
                "rule",
                REVIEW_FIRST,
                priority_method="deterministic_rule",
            ),
            _record(
                "quality",
                NEEDS_BETTER_IMAGE,
                priority_method="quality_gate",
            ),
        ]

        metrics = build_operational_metrics(_batch(records), {})

        self.assertEqual(metrics.experimental_model_prediction_count, 1)
        self.assertEqual(metrics.deterministic_prediction_count, 1)
        self.assertEqual(metrics.quality_gate_count, 1)

    def test_missing_quality_codes_and_nonfinite_values_are_ignored(self) -> None:
        record = _record(
            "legacy",
            LOWER_PRIORITY,
            issue_codes=None,
            advisory_codes=None,
            score=math.nan,
            embedding=tuple([0.1] * 1023 + [math.inf]),
        )

        metrics = build_operational_metrics(_batch([record]), {})

        self.assertEqual(metrics.quality_issue_counts, {})
        self.assertEqual(metrics.quality_advisory_counts, {})
        self.assertEqual(metrics.proxy_score_count, 0)
        self.assertEqual(metrics.embedding_success_count, 0)

    def test_invalid_reviewer_priority_is_not_counted_as_agreement(self) -> None:
        record = _record("image", REVIEW_FIRST)
        metrics = build_operational_metrics(
            _batch([record]),
            {"image": _review("invalid")},
        )

        self.assertEqual(metrics.reviewed_count, 1)
        self.assertEqual(metrics.reviewed_with_priority_count, 0)
        self.assertIsNone(metrics.suggestion_agreement_percentage)
        self.assertEqual(metrics.effective_priority_counts[REVIEW_FIRST], 1)

    def test_completed_review_uses_saved_model_snapshot_after_settings_change(self) -> None:
        record = _record(
            "image",
            LOWER_PRIORITY,
            priority_method="deterministic",
        )
        review = _review(REVIEW_FIRST)
        review.update(
            {
                "suggested_priority_at_review": REVIEW_FIRST,
                "priority_method_at_review": "experimental_head",
            }
        )

        metrics = build_operational_metrics(_batch([record]), {"image": review})

        self.assertEqual(metrics.suggestion_confirmed_count, 1)
        self.assertEqual(metrics.suggestion_overridden_count, 0)
        self.assertEqual(metrics.model_reviewed_count, 1)
        self.assertEqual(metrics.model_confirmed_count, 1)
        rows = {
            row.suggested_priority: row
            for row in metrics.agreement_by_suggested_priority
        }
        self.assertEqual(rows[REVIEW_FIRST].reviewed_count, 1)
        self.assertEqual(rows[LOWER_PRIORITY].reviewed_count, 0)
        model_rows = {
            row.suggested_priority: row
            for row in metrics.model_agreement_by_suggested_priority
        }
        self.assertEqual(model_rows[REVIEW_FIRST].reviewed_count, 1)
        self.assertEqual(model_rows[LOWER_PRIORITY].reviewed_count, 0)

    def test_screening_time_must_be_finite_and_non_negative(self) -> None:
        for value in (-1, math.inf, math.nan, "not-a-number"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    build_operational_metrics(
                        _batch([]),
                        {},
                        screening_seconds_per_image=value,  # type: ignore[arg-type]
                    )


if __name__ == "__main__":
    unittest.main()
