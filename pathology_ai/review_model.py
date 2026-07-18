"""Optional local experimental review-priority head.

The head predicts an MHIST annotator-agreement proxy from a UNI embedding. It
does not predict disease, diagnosis, cancer, or clinical urgency. Joblib files
are executable artifacts, so this module loads only the fixed local model path,
never an uploaded file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

from .triage import LOWER_PRIORITY, REVIEW_FIRST


REVIEW_MODEL_NAME = "Experimental MHIST annotator-agreement proxy"
REVIEW_MODEL_SOURCE = f"{REVIEW_MODEL_NAME} (UNI + logistic regression)"
UNI_EMBEDDING_DIMENSION = 1024
_DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[1] / "models" / "review_priority_head"
)


@dataclass(frozen=True)
class ReviewModelStatus:
    ready: bool
    model_path: Path
    metadata_path: Path
    metrics_path: Path
    summary: str
    detail: str
    cache_key: str
    metrics: dict[str, float]
    decision_threshold: float | None = None
    evaluation_report: dict[str, Any] = field(default_factory=dict)
    evaluation_valid: bool = False
    evaluation_error: str | None = None


@dataclass(frozen=True)
class ReviewModelPrediction:
    priority: str
    review_first_score: float
    source: str = REVIEW_MODEL_SOURCE


def _artifact_key(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def _required_count(values: dict[str, Any], key: str) -> int:
    raw = float(values[key])
    if not np.isfinite(raw) or raw < 0 or not raw.is_integer():
        raise ValueError(f"evaluation metric {key!r} must be a non-negative integer")
    return int(raw)


def _required_rate(values: dict[str, Any], key: str) -> float:
    value = float(values[key])
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"evaluation metric {key!r} must be between zero and one")
    return value


def _curve_series(curve: dict[str, Any], key: str) -> np.ndarray:
    values = np.asarray(curve.get(key), dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError(f"evaluation curve {key!r} must contain finite 1D values")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"evaluation curve {key!r} must stay between zero and one")
    return values


def _validate_curve_diagnostics(
    overall: dict[str, Any],
    *,
    roc_auc: float,
    positive_prevalence: float,
) -> None:
    """Validate optional raw coordinates used by dashboard ROC/PR charts."""

    has_roc = "roc_curve" in overall
    has_pr = "precision_recall_curve" in overall
    if not has_roc and not has_pr:
        return
    if not has_roc or not has_pr:
        raise ValueError("evaluation ROC and precision-recall curves must be provided together")

    roc = overall["roc_curve"]
    if not isinstance(roc, dict) or roc.get("positive_class") != REVIEW_FIRST:
        raise ValueError("ROC curve must identify Review First as the positive class")
    false_positive_rate = _curve_series(roc, "false_positive_rate")
    true_positive_rate = _curve_series(roc, "true_positive_rate")
    if false_positive_rate.shape != true_positive_rate.shape:
        raise ValueError("ROC curve coordinate arrays must have equal lengths")
    if np.any(np.diff(false_positive_rate) < 0.0) or np.any(
        np.diff(true_positive_rate) < 0.0
    ):
        raise ValueError("ROC curve coordinates must be non-decreasing")
    if not (
        np.isclose(false_positive_rate[0], 0.0)
        and np.isclose(true_positive_rate[0], 0.0)
        and np.isclose(false_positive_rate[-1], 1.0)
        and np.isclose(true_positive_rate[-1], 1.0)
    ):
        raise ValueError("ROC curve must run from (0, 0) to (1, 1)")
    calculated_auc = float(np.trapezoid(true_positive_rate, false_positive_rate))
    if not np.isclose(calculated_auc, roc_auc, rtol=0.0, atol=1e-12):
        raise ValueError("ROC curve coordinates conflict with roc_auc")

    precision_recall = overall["precision_recall_curve"]
    if (
        not isinstance(precision_recall, dict)
        or precision_recall.get("positive_class") != REVIEW_FIRST
    ):
        raise ValueError(
            "precision-recall curve must identify Review First as the positive class"
        )
    recall = _curve_series(precision_recall, "recall")
    precision = _curve_series(precision_recall, "precision")
    if recall.shape != precision.shape:
        raise ValueError("precision-recall coordinate arrays must have equal lengths")
    if np.any(np.diff(recall) > 0.0):
        raise ValueError("precision-recall recall coordinates must be non-increasing")
    baseline = _required_rate(precision_recall, "baseline_precision")
    if not np.isclose(baseline, positive_prevalence, rtol=0.0, atol=1e-12):
        raise ValueError("precision-recall baseline conflicts with class prevalence")


def _validate_evaluation_report(
    report: Any,
    decision_threshold: float,
) -> dict[str, Any]:
    """Validate the evaluation fields consumed by the dashboard.

    Evaluation problems must not disable inference, but an incomplete or
    threshold-mismatched report must never be presented beside the active head.
    """

    if not isinstance(report, dict):
        raise ValueError("evaluation report must be a JSON object")
    evaluation_threshold = float(report["classification_threshold"])
    if not np.isfinite(evaluation_threshold) or not 0.0 < evaluation_threshold < 1.0:
        raise ValueError("evaluation classification threshold is invalid")
    if not np.isclose(evaluation_threshold, decision_threshold, rtol=0.0, atol=1e-12):
        raise ValueError(
            "evaluation threshold mismatch: metrics use "
            f"{evaluation_threshold:g}, active model uses {decision_threshold:g}"
        )

    overall = report.get("overall_test_metrics")
    if not isinstance(overall, dict):
        raise ValueError("overall_test_metrics is missing or invalid")
    sample_count = _required_count(overall, "sample_count")
    review_first_count = _required_count(overall, "review_first_count")
    lower_priority_count = _required_count(overall, "lower_priority_count")
    predicted_review_first_count = _required_count(
        overall, "predicted_review_first_count"
    )
    if sample_count <= 0:
        raise ValueError("evaluation sample_count must be positive")
    if review_first_count <= 0 or lower_priority_count <= 0:
        raise ValueError("evaluation must contain both proxy classes")
    if review_first_count + lower_priority_count != sample_count:
        raise ValueError("evaluation class counts do not sum to sample_count")
    if predicted_review_first_count > sample_count:
        raise ValueError("predicted Review First count exceeds sample_count")

    rates = {
        key: _required_rate(overall, key)
        for key in (
            "balanced_accuracy",
            "roc_auc",
            "average_precision",
            "review_first_precision",
            "review_first_recall",
            "review_first_f1",
            "lower_priority_specificity",
            "predicted_review_first_fraction",
        )
    }

    confusion = overall.get("confusion_matrix")
    if not isinstance(confusion, dict):
        raise ValueError("confusion_matrix is missing or invalid")
    expected_labels = [LOWER_PRIORITY, REVIEW_FIRST]
    if confusion.get("row_axis") != "predicted" or confusion.get("column_axis") != "actual":
        raise ValueError("confusion_matrix axes must be predicted rows and actual columns")
    if confusion.get("row_labels") != expected_labels:
        raise ValueError("confusion_matrix row labels are incompatible")
    if confusion.get("column_labels") != expected_labels:
        raise ValueError("confusion_matrix column labels are incompatible")
    matrix = np.asarray(confusion.get("values"), dtype=np.float64)
    if (
        matrix.shape != (2, 2)
        or not np.isfinite(matrix).all()
        or np.any(matrix < 0)
        or not np.equal(matrix, np.floor(matrix)).all()
    ):
        raise ValueError("confusion_matrix must contain four non-negative integer counts")
    matrix = matrix.astype(np.int64)
    if int(matrix.sum()) != sample_count:
        raise ValueError("confusion_matrix counts do not sum to sample_count")
    if int(matrix[:, 1].sum()) != review_first_count:
        raise ValueError("confusion_matrix actual Review First count is inconsistent")
    if int(matrix[:, 0].sum()) != lower_priority_count:
        raise ValueError("confusion_matrix actual Lower Priority count is inconsistent")
    if int(matrix[1, :].sum()) != predicted_review_first_count:
        raise ValueError("confusion_matrix predicted Review First count is inconsistent")

    true_lower = int(matrix[0, 0])
    false_lower = int(matrix[0, 1])
    false_review_first = int(matrix[1, 0])
    true_review_first = int(matrix[1, 1])
    expected_precision = true_review_first / max(
        true_review_first + false_review_first, 1
    )
    expected_recall = true_review_first / max(true_review_first + false_lower, 1)
    expected_specificity = true_lower / max(true_lower + false_review_first, 1)
    expected_f1 = (
        2.0 * expected_precision * expected_recall
        / max(expected_precision + expected_recall, np.finfo(float).eps)
    )
    expected_balanced_accuracy = (expected_recall + expected_specificity) / 2.0
    expected_predicted_fraction = predicted_review_first_count / sample_count
    expected_rates = {
        "review_first_precision": expected_precision,
        "review_first_recall": expected_recall,
        "review_first_f1": expected_f1,
        "lower_priority_specificity": expected_specificity,
        "balanced_accuracy": expected_balanced_accuracy,
        "predicted_review_first_fraction": expected_predicted_fraction,
    }
    for key, expected_value in expected_rates.items():
        if not np.isclose(rates[key], expected_value, rtol=0.0, atol=1e-12):
            raise ValueError(f"evaluation metric {key!r} conflicts with confusion_matrix")

    _validate_curve_diagnostics(
        overall,
        roc_auc=rates["roc_auc"],
        positive_prevalence=review_first_count / sample_count,
    )

    captures = overall.get("review_first_capture_by_queue_fraction")
    if not isinstance(captures, list) or len(captures) != 3:
        raise ValueError("top-queue Review First capture metrics are missing")
    expected_fractions = (0.10, 0.25, 0.50)
    found_fractions: list[float] = []
    previous_captured = -1
    for entry in captures:
        if not isinstance(entry, dict):
            raise ValueError("top-queue capture entry must be an object")
        queue_fraction = float(entry["queue_fraction"])
        queue_size = _required_count(entry, "queue_size")
        captured = _required_count(entry, "captured_review_first_count")
        total = _required_count(entry, "total_review_first_count")
        capture_fraction = _required_rate(entry, "capture_fraction")
        expected_queue_size = max(1, int(np.ceil(sample_count * queue_fraction)))
        if queue_size != expected_queue_size:
            raise ValueError("top-queue size is inconsistent with its queue fraction")
        if total != review_first_count or captured > min(total, queue_size):
            raise ValueError("top-queue Review First counts are inconsistent")
        if captured < previous_captured:
            raise ValueError("top-queue capture counts must be non-decreasing")
        expected_capture = captured / max(total, 1)
        if not np.isclose(capture_fraction, expected_capture, rtol=0.0, atol=1e-12):
            raise ValueError("top-queue capture fraction is inconsistent with its counts")
        found_fractions.append(queue_fraction)
        previous_captured = captured
    if not np.allclose(found_fractions, expected_fractions, rtol=0.0, atol=1e-12):
        raise ValueError("top-queue capture fractions must be 10%, 25%, and 50%")

    return report


def get_review_model_status(
    model_dir: str | Path | None = None,
) -> ReviewModelStatus:
    directory = Path(model_dir or _DEFAULT_MODEL_DIR).resolve()
    model_path = directory / "review_priority_head.joblib"
    metadata_path = directory / "metadata.json"
    metrics_path = directory / "metrics.json"
    paths = (model_path, metadata_path)
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        return ReviewModelStatus(
            ready=False,
            model_path=model_path,
            metadata_path=metadata_path,
            metrics_path=metrics_path,
            summary="Experimental priority head is not installed",
            detail=(
                "Local model artifacts are missing; deterministic review-priority "
                "fallback remains available."
            ),
            cache_key="review-head:missing:" + ",".join(sorted(missing)),
            metrics={},
        )
    missing_packages = [
        name
        for name in ("joblib", "sklearn")
        if importlib.util.find_spec(name) is None
    ]
    if missing_packages:
        return ReviewModelStatus(
            ready=False,
            model_path=model_path,
            metadata_path=metadata_path,
            metrics_path=metrics_path,
            summary="Experimental priority head dependencies are missing",
            detail=(
                "Install requirements-training.txt to enable the local prototype head "
                f"({', '.join(missing_packages)} missing)."
            ),
            cache_key="review-head:missing-packages:" + ",".join(missing_packages),
            metrics={},
        )

    cache_key = "review-head:" + ":".join(_artifact_key(path) for path in paths)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(metadata.get("embedding_dimension", -1)) != UNI_EMBEDDING_DIMENSION:
            raise ValueError("metadata has an incompatible embedding dimension")
        if metadata.get("embedding_model") != "MahmoodLab/UNI":
            raise ValueError("metadata does not identify the MahmoodLab/UNI encoder")
        classes = set(metadata.get("model_classes", []))
        if classes != {LOWER_PRIORITY, REVIEW_FIRST}:
            raise ValueError("metadata has incompatible model classes")
        threshold = float(metadata["decision_threshold"])
        if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise ValueError("metadata has an invalid decision threshold")
    except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return ReviewModelStatus(
            ready=False,
            model_path=model_path,
            metadata_path=metadata_path,
            metrics_path=metrics_path,
            summary="Experimental priority head metadata is incompatible",
            detail=f"The local prototype head was not loaded ({exc}).",
            cache_key=cache_key + ":invalid-metadata",
            metrics={},
        )

    shown_metrics: dict[str, float] = {}
    evaluation_report: dict[str, Any] = {}
    evaluation_valid = False
    evaluation_error: str | None = None
    if metrics_path.is_file():
        cache_key += ":" + _artifact_key(metrics_path)
        try:
            raw_report = json.loads(metrics_path.read_text(encoding="utf-8"))
            evaluation_report = _validate_evaluation_report(raw_report, threshold)
            overall = evaluation_report["overall_test_metrics"]
            shown_metrics = {
                key: float(value)
                for key, value in overall.items()
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
                and np.isfinite(float(value))
            }
            evaluation_valid = True
        except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            evaluation_report = {}
            evaluation_error = f"Evaluation metrics unavailable ({exc})."
    else:
        cache_key += ":metrics-missing"
        evaluation_error = "Evaluation metrics unavailable (metrics.json is missing)."

    detail = (
        "This quick local head predicts an MHIST annotator-agreement proxy from "
        "UNI features. It is not clinically validated."
    )
    if evaluation_error:
        detail += f" {evaluation_error}"

    return ReviewModelStatus(
        ready=True,
        model_path=model_path,
        metadata_path=metadata_path,
        metrics_path=metrics_path,
        summary="Experimental priority head is ready",
        detail=detail,
        cache_key=cache_key,
        metrics=shown_metrics,
        decision_threshold=threshold,
        evaluation_report=evaluation_report,
        evaluation_valid=evaluation_valid,
        evaluation_error=evaluation_error,
    )


@lru_cache(maxsize=2)
def _load_review_model(
    model_path: str,
    model_size: int,
    model_mtime_ns: int,
    metadata_path: str,
    metadata_size: int,
    metadata_mtime_ns: int,
) -> tuple[Any, float, tuple[str, ...]]:
    del model_size, model_mtime_ns, metadata_size, metadata_mtime_ns
    import joblib

    model = joblib.load(model_path)
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    threshold = float(metadata["decision_threshold"])
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise ValueError("The local review head does not provide class probabilities.")
    classes = tuple(str(value) for value in model.classes_)
    if set(classes) != {LOWER_PRIORITY, REVIEW_FIRST}:
        raise ValueError("The local review head has unexpected classes.")
    return model, threshold, classes


class LocalPrototypeReviewHead:
    """Predict review order from a finite 1,024-value UNI embedding."""

    def __init__(self, status: ReviewModelStatus | None = None) -> None:
        self.status = status or get_review_model_status()
        if not self.status.ready:
            raise RuntimeError(self.status.detail)

    def predict(self, embedding: tuple[float, ...]) -> ReviewModelPrediction:
        values = np.asarray(embedding, dtype=np.float32)
        if values.shape != (UNI_EMBEDDING_DIMENSION,) or not np.isfinite(values).all():
            raise ValueError("UNI embedding must contain 1,024 finite values.")
        model_stat = self.status.model_path.stat()
        metadata_stat = self.status.metadata_path.stat()
        model, threshold, classes = _load_review_model(
            str(self.status.model_path),
            model_stat.st_size,
            model_stat.st_mtime_ns,
            str(self.status.metadata_path),
            metadata_stat.st_size,
            metadata_stat.st_mtime_ns,
        )
        probabilities = np.asarray(model.predict_proba(values.reshape(1, -1))[0])
        review_first_score = float(probabilities[classes.index(REVIEW_FIRST)])
        if not np.isfinite(review_first_score):
            raise ValueError("The local review head returned a non-finite score.")
        priority = REVIEW_FIRST if review_first_score >= threshold else LOWER_PRIORITY
        return ReviewModelPrediction(
            priority=priority,
            review_first_score=review_first_score,
        )


def get_review_model() -> LocalPrototypeReviewHead:
    """Return the fixed trusted local prototype head or raise clearly."""

    return LocalPrototypeReviewHead(get_review_model_status())
