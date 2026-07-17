from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_review_head.py"
SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None
if SKLEARN_AVAILABLE:
    SPEC = importlib.util.spec_from_file_location("train_review_head", SCRIPT_PATH)
    assert SPEC and SPEC.loader
    MODULE = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(MODULE)
else:
    MODULE = None


@unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is an optional training dependency")
class TrainReviewHeadTests(unittest.TestCase):
    def test_metrics_for_uses_review_first_as_positive_class(self):
        truth = np.asarray(["Lower Priority", "Review First", "Review First"])
        predicted = np.asarray(["Lower Priority", "Lower Priority", "Review First"])
        probabilities = np.asarray([0.1, 0.4, 0.8])
        metrics = MODULE.metrics_for(truth, predicted, probabilities)

        self.assertEqual(metrics["sample_count"], 3)
        self.assertEqual(metrics["review_first_count"], 2)
        self.assertEqual(metrics["review_first_recall"], 0.5)
        self.assertEqual(
            metrics["confusion_matrix"]["label_order"],
            ["Lower Priority", "Review First"],
        )
        self.assertEqual(metrics["confusion_matrix"]["row_axis"], "predicted")
        self.assertEqual(metrics["confusion_matrix"]["column_axis"], "actual")
        self.assertEqual(metrics["confusion_matrix"]["values"], [[1, 1], [0, 1]])
        captures = metrics["review_first_capture_by_queue_fraction"]
        self.assertEqual([item["queue_fraction"] for item in captures], [0.1, 0.25, 0.5])
        self.assertEqual([item["queue_size"] for item in captures], [1, 1, 2])
        self.assertEqual([item["captured_review_first_count"] for item in captures], [1, 1, 2])

    def test_queue_capture_is_stable_for_tied_scores(self):
        truth = np.asarray(["Review First", "Lower Priority", "Review First", "Lower Priority"])
        probabilities = np.asarray([0.5, 0.5, 0.5, 0.5])

        captures = MODULE.review_first_capture_by_queue_fraction(truth, probabilities)

        self.assertEqual(captures[0]["queue_size"], 1)
        self.assertEqual(captures[0]["captured_review_first_count"], 1)
        self.assertEqual(captures[-1]["queue_size"], 2)
        self.assertEqual(captures[-1]["captured_review_first_count"], 1)
