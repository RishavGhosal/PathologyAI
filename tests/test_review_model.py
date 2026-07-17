from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pathology_ai.review_model import LocalPrototypeReviewHead, get_review_model_status


MODEL_DEPENDENCIES_AVAILABLE = all(
    importlib.util.find_spec(name) is not None for name in ("joblib", "sklearn")
)


class _PredictionFixture:
    classes_ = ("Lower Priority", "Review First")

    def predict_proba(self, values):
        return [[0.2, 0.8] for _value in values]


def _metadata(threshold: float = 0.5) -> dict[str, object]:
    return {
        "embedding_dimension": 1024,
        "embedding_model": "MahmoodLab/UNI",
        "model_classes": ["Lower Priority", "Review First"],
        "decision_threshold": threshold,
    }


def _report(threshold: float = 0.5) -> dict[str, object]:
    return {
        "classification_threshold": threshold,
        "overall_test_metrics": {
            "sample_count": 4,
            "review_first_count": 2,
            "lower_priority_count": 2,
            "balanced_accuracy": 0.5,
            "roc_auc": 0.5,
            "average_precision": 0.5,
            "review_first_precision": 0.5,
            "review_first_recall": 0.5,
            "review_first_f1": 0.5,
            "lower_priority_specificity": 0.5,
            "predicted_review_first_count": 2,
            "predicted_review_first_fraction": 0.5,
            "confusion_matrix": {
                "row_axis": "predicted",
                "column_axis": "actual",
                "row_labels": ["Lower Priority", "Review First"],
                "column_labels": ["Lower Priority", "Review First"],
                "label_order": ["Lower Priority", "Review First"],
                "values": [[1, 1], [1, 1]],
            },
            "review_first_capture_by_queue_fraction": [
                {
                    "queue_fraction": 0.1,
                    "queue_size": 1,
                    "captured_review_first_count": 1,
                    "total_review_first_count": 2,
                    "capture_fraction": 0.5,
                },
                {
                    "queue_fraction": 0.25,
                    "queue_size": 1,
                    "captured_review_first_count": 1,
                    "total_review_first_count": 2,
                    "capture_fraction": 0.5,
                },
                {
                    "queue_fraction": 0.5,
                    "queue_size": 2,
                    "captured_review_first_count": 1,
                    "total_review_first_count": 2,
                    "capture_fraction": 0.5,
                },
            ],
        },
    }


def _write_base_artifacts(directory: Path, threshold: float = 0.5) -> None:
    import joblib

    joblib.dump(_PredictionFixture(), directory / "review_priority_head.joblib")
    (directory / "metadata.json").write_text(
        json.dumps(_metadata(threshold)), encoding="utf-8"
    )


@unittest.skipUnless(
    MODEL_DEPENDENCIES_AVAILABLE, "review-head dependencies are optional"
)
class ReviewModelEvaluationStatusTests(unittest.TestCase):
    def test_valid_report_is_exposed_with_active_threshold(self) -> None:
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            _write_base_artifacts(directory)
            report = _report()
            (directory / "metrics.json").write_text(
                json.dumps(report), encoding="utf-8"
            )

            status = get_review_model_status(directory)

        self.assertTrue(status.ready)
        self.assertTrue(status.evaluation_valid, status.evaluation_error)
        self.assertEqual(status.decision_threshold, 0.5)
        self.assertEqual(status.evaluation_report, report)
        self.assertEqual(status.metrics["sample_count"], 4.0)

    def test_threshold_mismatch_does_not_disable_inference(self) -> None:
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            _write_base_artifacts(directory, threshold=0.5)
            (directory / "metrics.json").write_text(
                json.dumps(_report(threshold=0.4)), encoding="utf-8"
            )

            status = get_review_model_status(directory)
            prediction = LocalPrototypeReviewHead(status).predict(
                tuple([0.0] * 1024)
            )

        self.assertTrue(status.ready)
        self.assertFalse(status.evaluation_valid)
        self.assertEqual(status.decision_threshold, 0.5)
        self.assertEqual(status.metrics, {})
        self.assertEqual(status.evaluation_report, {})
        self.assertIn("threshold mismatch", status.evaluation_error or "")
        self.assertEqual(prediction.priority, "Review First")
        self.assertEqual(prediction.review_first_score, 0.8)

    def test_inconsistent_threshold_metrics_are_hidden(self) -> None:
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            _write_base_artifacts(directory)
            report = _report()
            report["overall_test_metrics"]["review_first_recall"] = 0.75
            (directory / "metrics.json").write_text(
                json.dumps(report), encoding="utf-8"
            )

            status = get_review_model_status(directory)

        self.assertTrue(status.ready)
        self.assertFalse(status.evaluation_valid)
        self.assertIn("conflicts with confusion_matrix", status.evaluation_error or "")

    def test_inconsistent_top_queue_size_is_hidden(self) -> None:
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            _write_base_artifacts(directory)
            report = _report()
            report["overall_test_metrics"]["review_first_capture_by_queue_fraction"][0][
                "queue_size"
            ] = 2
            (directory / "metrics.json").write_text(
                json.dumps(report), encoding="utf-8"
            )

            status = get_review_model_status(directory)

        self.assertTrue(status.ready)
        self.assertFalse(status.evaluation_valid)
        self.assertIn("queue size is inconsistent", status.evaluation_error or "")

    def test_missing_metrics_keeps_inference_ready(self) -> None:
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            _write_base_artifacts(directory)

            status = get_review_model_status(directory)

        self.assertTrue(status.ready)
        self.assertFalse(status.evaluation_valid)
        self.assertEqual(status.decision_threshold, 0.5)
        self.assertIn("metrics.json is missing", status.evaluation_error or "")

    def test_malformed_metrics_keeps_inference_ready(self) -> None:
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            _write_base_artifacts(directory)
            (directory / "metrics.json").write_text("{not-json", encoding="utf-8")

            status = get_review_model_status(directory)
            prediction = LocalPrototypeReviewHead(status).predict(
                tuple([0.0] * 1024)
            )

        self.assertTrue(status.ready)
        self.assertFalse(status.evaluation_valid)
        self.assertEqual(status.evaluation_report, {})
        self.assertIn("Evaluation metrics unavailable", status.evaluation_error or "")
        self.assertEqual(prediction.priority, "Review First")


if __name__ == "__main__":
    unittest.main()
