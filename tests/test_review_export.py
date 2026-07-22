"""Tests for the de-identified reviewed-label CSV export."""

from __future__ import annotations

import csv
from io import StringIO
from types import SimpleNamespace
import unittest

from PIL import Image

from pathology_ai.attention import AttentionResult
from pathology_ai.quality import QualityAssessment
from pathology_ai.review_export import (
    EMBEDDING_COLUMNS,
    build_review_export_csv,
)
from pathology_ai.triage import LOWER_PRIORITY, REVIEW_FIRST, TriageResult
from pathology_ai.uni_provider import UNI_EMBEDDING_DIMENSION, UNI_MODEL_ID


def _record(image_id: str, embedding: tuple[float, ...] | None) -> SimpleNamespace:
    preview = Image.new("RGB", (16, 16), "white")
    return SimpleNamespace(
        image_id=image_id,
        display_name="private_filename.jpg",
        file_name="private_filename.jpg",
        source_name="private_upload.zip",
        quality=QualityAssessment(
            True,
            (),
            {},
            advisories=("Possible edge contact; verify manually.",),
            advisory_codes=("possible_edge_truncation",),
        ),
        attention=AttentionResult(
            overlay=preview,
            heatmap=preview,
            explanation="Research feature visualization.",
            visual_complexity_score=0.5,
            provider_name="Local UNI feature-variation demonstration (CPU)",
            is_demonstration=True,
            uses_trained_encoder=embedding is not None,
            embedding=embedding,
            embedding_model=UNI_MODEL_ID if embedding is not None else None,
        ),
        triage=TriageResult(REVIEW_FIRST, "Review-order fixture."),
    )


class ReviewExportTests(unittest.TestCase):
    def test_export_contains_only_reviewed_rows_and_1024_uni_features(self) -> None:
        embedding = tuple(index / 1000.0 for index in range(UNI_EMBEDDING_DIMENSION))
        reviewed = _record("reviewed-id", embedding)
        unreviewed = _record("unreviewed-id", None)
        reviews = {
            "reviewed-id": {
                "reviewed": True,
                "priority": LOWER_PRIORITY,
                "notes": "reviewed, educational example",
                "group_id": "slide-group-001",
                "reviewed_at_utc": "2026-07-17T20:00:00+00:00",
            },
            "unreviewed-id": {
                "reviewed": False,
                "priority": REVIEW_FIRST,
                "notes": "not exported",
                "group_id": "slide-group-002",
            },
        }

        payload = build_review_export_csv([reviewed, unreviewed], reviews).decode("utf-8")
        rows = list(csv.DictReader(StringIO(payload)))

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["image_id"], "reviewed-id")
        self.assertEqual(row["reviewer_priority"], LOWER_PRIORITY)
        self.assertEqual(row["group_id_format_validated"], "True")
        self.assertEqual(row["embedding_available"], "True")
        self.assertEqual(row["embedding_dimension"], "1024")
        self.assertEqual(row["priority_overridden"], "True")
        self.assertEqual(row["domain_context"], "unknown_or_other")
        self.assertEqual(row["reviewed_at_utc"], "2026-07-17T20:00:00+00:00")
        self.assertEqual(row["quality_issue_codes"], "")
        self.assertEqual(row["quality_advisory_codes"], "possible_edge_truncation")
        self.assertIn("priority_source", row)
        self.assertEqual(len(EMBEDDING_COLUMNS), 1024)
        self.assertNotEqual(row["uni_1023"], "")
        self.assertNotIn("private_filename.jpg", payload)
        self.assertNotIn("private_upload.zip", payload)
        self.assertNotIn("unreviewed-id", payload)

    def test_export_neutralizes_spreadsheet_formula_text(self) -> None:
        record = _record("formula-id", None)
        reviews = {
            "formula-id": {
                "reviewed": True,
                "priority": REVIEW_FIRST,
                "notes": "=HYPERLINK(\"https://example.invalid\")",
                "group_id": "private-value",
            }
        }

        row = next(
            csv.DictReader(
                StringIO(build_review_export_csv([record], reviews).decode("utf-8"))
            )
        )

        self.assertTrue(row["reviewer_notes"].startswith("'="))
        self.assertEqual(row["group_id"], "private-value")
        self.assertEqual(row["embedding_available"], "False")
        self.assertEqual(row["uni_0000"], "")

    def test_export_rejects_wrong_embedding_dimension(self) -> None:
        record = _record("bad-embedding", (0.1, 0.2))
        reviews = {
            "bad-embedding": {
                "reviewed": True,
                "priority": REVIEW_FIRST,
                "notes": "",
                "group_id": "slide-group-003",
            }
        }

        with self.assertRaisesRegex(ValueError, "unexpected dimension"):
            build_review_export_csv([record], reviews)

    def test_export_uses_completed_review_snapshot_after_model_change(self) -> None:
        record = _record("snapshot-id", None)
        record.triage = TriageResult(LOWER_PRIORITY, "Current deterministic result.")
        saved_embedding = tuple([0.125] * UNI_EMBEDDING_DIMENSION)
        reviews = {
            "snapshot-id": {
                "reviewed": True,
                "priority": REVIEW_FIRST,
                "notes": "",
                "group_id": "slide-group-005",
                "suggested_priority_at_review": REVIEW_FIRST,
                "priority_source_at_review": "Saved experimental head",
                "priority_method_at_review": "experimental_head",
                "priority_fallback_reason_at_review": "",
                "review_first_proxy_score_at_review": 0.7,
                "attention_provider_at_review": "Saved UNI provider",
                "embedding_model_at_review": UNI_MODEL_ID,
                "embedding_at_review": saved_embedding,
            }
        }

        row = next(
            csv.DictReader(
                StringIO(build_review_export_csv([record], reviews).decode("utf-8"))
            )
        )

        self.assertEqual(row["suggested_priority"], REVIEW_FIRST)
        self.assertEqual(row["priority_overridden"], "False")
        self.assertEqual(row["priority_method"], "experimental_head")
        self.assertEqual(row["review_first_proxy_score"], "0.7")
        self.assertEqual(row["attention_provider"], "Saved UNI provider")
        self.assertEqual(row["embedding_available"], "True")
        self.assertEqual(row["uni_0000"], "0.125")

    def test_export_allows_missing_group_but_rejects_missing_override_reason(self) -> None:
        record = _record("required-fields", None)
        missing_group = {
            "required-fields": {
                "reviewed": True,
                "priority": REVIEW_FIRST,
                "notes": "",
                "group_id": "",
            }
        }
        row = next(
            csv.DictReader(
                StringIO(build_review_export_csv([record], missing_group).decode("utf-8"))
            )
        )
        self.assertEqual(row["group_id"], "")
        self.assertEqual(row["group_id_format_validated"], "False")

        missing_reason = {
            "required-fields": {
                "reviewed": True,
                "priority": LOWER_PRIORITY,
                "notes": "",
                "group_id": "slide-group-004",
            }
        }
        with self.assertRaisesRegex(ValueError, "notes are required"):
            build_review_export_csv([record], missing_reason)


if __name__ == "__main__":
    unittest.main()
