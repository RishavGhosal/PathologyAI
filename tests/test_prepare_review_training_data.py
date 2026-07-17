"""Tests for group-safe reviewer-export preparation."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from pathology_ai.review_export import validate_group_id


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "prepare_review_training_data.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_review_training_data", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None


FIELDS = [
    "image_id",
    "group_id",
    "suggested_priority",
    "reviewer_priority",
    "reviewed",
    "reviewer_notes",
    "priority_overridden",
    "embedding_available",
    "embedding_dimension",
] + list(MODULE.EMBEDDING_COLUMNS)


def _row(
    image_id: str,
    group_id: str,
    label: str,
    value: float,
    *,
    suggested: str | None = None,
    blank_embedding: bool = False,
) -> dict[str, str]:
    row = {
        "image_id": image_id,
        "group_id": group_id,
        "suggested_priority": suggested or label,
        "reviewer_priority": label,
        "reviewed": "True",
        "reviewer_notes": "",
        "priority_overridden": str(bool(suggested and suggested != label)),
        "embedding_available": "False" if blank_embedding else "True",
        "embedding_dimension": "" if blank_embedding else "1024",
    }
    row.update(
        {
            column: "" if blank_embedding else str(value + index / 10000.0)
            for index, column in enumerate(MODULE.EMBEDDING_COLUMNS)
        }
    )
    return row


def _write_csv(path: Path, rows: list[dict[str, str]], fields=FIELDS) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _valid_binary_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index in range(5):
        rows.append(
            _row(
                f"lower-{index}",
                f"case-lower-{index}",
                MODULE.LOWER_PRIORITY,
                float(index),
            )
        )
        rows.append(
            _row(
                f"first-{index}",
                f"case-first-{index}",
                MODULE.REVIEW_FIRST,
                float(index + 10),
            )
        )
    return rows


class PrepareReviewTrainingDataTests(unittest.TestCase):
    @unittest.skipUnless(
        SKLEARN_AVAILABLE, "scikit-learn is an optional training dependency"
    )
    def test_combines_deduplicates_excludes_quality_rows_and_builds_folds(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_csv = root / "first.csv"
            second_csv = root / "second.csv"
            output_dir = root / "prepared"
            rows = _valid_binary_rows()
            quality_only = _row(
                "quality-only",
                "case-quality",
                MODULE.NEEDS_BETTER_IMAGE,
                0.0,
                blank_embedding=True,
            )
            _write_csv(first_csv, rows[:6])
            _write_csv(second_csv, rows[6:] + [rows[0], quality_only])

            report = MODULE.prepare_review_training_data(
                [second_csv, first_csv], output_dir
            )

            self.assertEqual(report["status"], "success")
            self.assertEqual(report["counts"]["input_rows"], 12)
            self.assertEqual(report["counts"]["exact_duplicate_rows_removed"], 1)
            self.assertEqual(
                report["counts"]["needs_better_image_rows_excluded"], 1
            )
            self.assertEqual(report["counts"]["training_rows"], 10)
            self.assertEqual(
                report["label_balance"],
                {MODULE.LOWER_PRIORITY: 5, MODULE.REVIEW_FIRST: 5},
            )
            self.assertFalse(report["folds"]["group_leakage_detected"])

            with (output_dir / MODULE.MANIFEST_FILENAME).open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                manifest = list(csv.DictReader(handle))
            embeddings = np.load(output_dir / MODULE.EMBEDDINGS_FILENAME)
            self.assertEqual(len(manifest), 10)
            self.assertEqual(embeddings.shape, (10, 1024))
            self.assertEqual(embeddings.dtype, np.float32)
            self.assertFalse(
                any(
                    column.startswith("uni_")
                    for column in manifest[0]
                )
            )
            self.assertEqual([row["image_id"] for row in manifest], sorted(
                row["image_id"] for row in manifest
            ))
            for index, row in enumerate(manifest):
                source = next(item for item in rows if item["image_id"] == row["image_id"])
                self.assertAlmostEqual(embeddings[index, 0], float(source["uni_0000"]))

            groups_to_folds: dict[str, set[str]] = {}
            labels_by_fold: dict[str, set[str]] = {}
            for row in manifest:
                groups_to_folds.setdefault(row["group_id"], set()).add(row["fold"])
                labels_by_fold.setdefault(row["fold"], set()).add(
                    row["reviewer_priority"]
                )
            self.assertTrue(all(len(folds) == 1 for folds in groups_to_folds.values()))
            self.assertEqual(set(labels_by_fold), {"0", "1", "2", "3", "4"})
            self.assertTrue(
                all(labels == set(MODULE.TRAINING_LABELS) for labels in labels_by_fold.values())
            )

    @unittest.skipUnless(
        SKLEARN_AVAILABLE, "scikit-learn is an optional training dependency"
    )
    def test_fold_assignment_is_deterministic_regardless_of_input_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_csv = root / "a.csv"
            second_csv = root / "b.csv"
            rows = _valid_binary_rows()
            _write_csv(first_csv, rows[::2])
            _write_csv(second_csv, rows[1::2])

            first_output = root / "out-1"
            second_output = root / "out-2"
            MODULE.prepare_review_training_data([first_csv, second_csv], first_output)
            MODULE.prepare_review_training_data([second_csv, first_csv], second_output)

            self.assertEqual(
                (first_output / MODULE.MANIFEST_FILENAME).read_bytes(),
                (second_output / MODULE.MANIFEST_FILENAME).read_bytes(),
            )
            np.testing.assert_array_equal(
                np.load(first_output / MODULE.EMBEDDINGS_FILENAME),
                np.load(second_output / MODULE.EMBEDDINGS_FILENAME),
            )

    def test_conflicting_duplicate_fails_and_removes_stale_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_csv = root / "first.csv"
            second_csv = root / "second.csv"
            output_dir = root / "prepared"
            output_dir.mkdir()
            (output_dir / MODULE.MANIFEST_FILENAME).write_text("stale", encoding="utf-8")
            np.save(output_dir / MODULE.EMBEDDINGS_FILENAME, np.ones((1, 1)))
            original = _row("same-id", "case-1", MODULE.REVIEW_FIRST, 1.0)
            conflicting = dict(original)
            conflicting["reviewer_notes"] = "different"
            _write_csv(first_csv, [original])
            _write_csv(second_csv, [conflicting])

            report = MODULE.prepare_review_training_data(
                [first_csv, second_csv], output_dir
            )

            self.assertEqual(report["status"], "failed")
            self.assertIn("Conflicting duplicate image_id", report["errors"][0])
            self.assertFalse((output_dir / MODULE.MANIFEST_FILENAME).exists())
            self.assertFalse((output_dir / MODULE.EMBEDDINGS_FILENAME).exists())
            self.assertTrue((output_dir / MODULE.REPORT_FILENAME).is_file())

    def test_invalid_group_id_is_rejected_with_deidentification_caveat(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "invalid-group.csv"
            output_dir = root / "prepared"
            _write_csv(
                source,
                [_row("image-1", "patient name", MODULE.REVIEW_FIRST, 1.0)],
            )

            report = MODULE.prepare_review_training_data([source], output_dir)

            self.assertEqual(report["status"], "failed")
            self.assertIn("invalid group_id", report["errors"][0])
            self.assertIn("does not prove de-identification", report["errors"][0])

    def test_group_id_format_matches_application_export_validation(self):
        accepted = ["case-1", "CASE_002", "a", "A" * 64]
        rejected = ["", "case.1", "case:1", "-case", "A" * 65, "patient name"]

        for group_id in accepted:
            with self.subTest(group_id=group_id):
                self.assertTrue(MODULE.GROUP_ID_PATTERN.fullmatch(group_id))
                self.assertEqual(validate_group_id(group_id), group_id)
        for group_id in rejected:
            with self.subTest(group_id=group_id):
                self.assertFalse(MODULE.GROUP_ID_PATTERN.fullmatch(group_id))
                with self.assertRaises(ValueError):
                    validate_group_id(group_id)

    def test_nonfinite_training_embedding_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "nonfinite.csv"
            output_dir = root / "prepared"
            rows = _valid_binary_rows()
            rows[0]["uni_0123"] = "nan"
            _write_csv(source, rows)

            report = MODULE.prepare_review_training_data([source], output_dir)

            self.assertEqual(report["status"], "failed")
            self.assertIn("non-finite", report["errors"][0])
            self.assertFalse((output_dir / MODULE.MANIFEST_FILENAME).exists())
            self.assertFalse((output_dir / MODULE.EMBEDDINGS_FILENAME).exists())

    def test_finite_float64_value_that_overflows_float32_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "overflow.csv"
            output_dir = root / "prepared"
            rows = _valid_binary_rows()
            rows[0]["uni_0000"] = "1e300"
            _write_csv(source, rows)

            report = MODULE.prepare_review_training_data([source], output_dir)

            self.assertEqual(report["status"], "failed")
            self.assertIn("non-float32-representable", report["errors"][0])
            self.assertFalse((output_dir / MODULE.MANIFEST_FILENAME).exists())
            self.assertFalse((output_dir / MODULE.EMBEDDINGS_FILENAME).exists())

    def test_missing_embedding_column_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "short-embedding.csv"
            output_dir = root / "prepared"
            fields = [field for field in FIELDS if field != "uni_1023"]
            _write_csv(source, [_row("image-1", "case-1", MODULE.REVIEW_FIRST, 1.0)], fields)

            report = MODULE.prepare_review_training_data([source], output_dir)

            self.assertEqual(report["status"], "failed")
            self.assertIn("exactly 1,024", report["errors"][0])

    def test_insufficient_label_groups_refuses_instead_of_reducing_folds(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "too-few-groups.csv"
            output_dir = root / "prepared"
            rows = _valid_binary_rows()
            rows = [row for row in rows if row["group_id"] != "case-first-4"]
            _write_csv(source, rows)

            exit_code = MODULE.main(
                [str(source), "--output-dir", str(output_dir)]
            )
            report = json.loads(
                (output_dir / MODULE.REPORT_FILENAME).read_text(encoding="utf-8")
            )

            self.assertEqual(exit_code, 1)
            self.assertEqual(report["status"], "failed")
            self.assertIn("at least 5 distinct groups", report["errors"][0])
            self.assertIn("never reduced", report["errors"][0])
            self.assertFalse((output_dir / MODULE.MANIFEST_FILENAME).exists())
            self.assertFalse((output_dir / MODULE.EMBEDDINGS_FILENAME).exists())

    def test_missing_optional_sklearn_reports_failure_and_leaves_no_split_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "valid.csv"
            output_dir = root / "prepared"
            output_dir.mkdir()
            (output_dir / MODULE.MANIFEST_FILENAME).write_text(
                "stale", encoding="utf-8"
            )
            np.save(output_dir / MODULE.EMBEDDINGS_FILENAME, np.ones((1, 1)))
            _write_csv(source, _valid_binary_rows())
            real_import = __import__

            def import_without_sklearn(name, *args, **kwargs):
                if name == "sklearn.model_selection":
                    raise ImportError("simulated optional dependency absence")
                return real_import(name, *args, **kwargs)

            with mock.patch("builtins.__import__", side_effect=import_without_sklearn):
                exit_code = MODULE.main(
                    [str(source), "--output-dir", str(output_dir)]
                )
            report = json.loads(
                (output_dir / MODULE.REPORT_FILENAME).read_text(encoding="utf-8")
            )

            self.assertEqual(exit_code, 1)
            self.assertEqual(report["status"], "failed")
            self.assertIn("scikit-learn is required", report["errors"][0])
            self.assertEqual(report["folds"]["n_splits"], 5)
            self.assertFalse((output_dir / MODULE.MANIFEST_FILENAME).exists())
            self.assertFalse((output_dir / MODULE.EMBEDDINGS_FILENAME).exists())

    def test_override_without_notes_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "override.csv"
            output_dir = root / "prepared"
            row = _row(
                "image-1",
                "case-1",
                MODULE.REVIEW_FIRST,
                1.0,
                suggested=MODULE.LOWER_PRIORITY,
            )
            _write_csv(source, [row])

            report = MODULE.prepare_review_training_data([source], output_dir)

            self.assertEqual(report["status"], "failed")
            self.assertIn("no reviewer notes", report["errors"][0])

    def test_inconsistent_exported_override_flag_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "inconsistent-override.csv"
            output_dir = root / "prepared"
            row = _row(
                "image-1",
                "case-1",
                MODULE.REVIEW_FIRST,
                1.0,
                suggested=MODULE.LOWER_PRIORITY,
            )
            row["priority_overridden"] = "False"
            row["reviewer_notes"] = "Reviewer override reason"
            _write_csv(source, [row])

            report = MODULE.prepare_review_training_data([source], output_dir)

            self.assertEqual(report["status"], "failed")
            self.assertIn("inconsistent", report["errors"][0])


if __name__ == "__main__":
    unittest.main()
