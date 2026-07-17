"""Train an experimental review-priority head from saved UNI embeddings.

This model predicts an annotator-agreement proxy for review ordering. It does
not predict disease, cancer, or a medical diagnosis. The MHIST majority-vote
disease label is loaded only for audit metrics and is never a model feature.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_DIMENSION = 1024
REVIEW_FIRST = "Review First"
LOWER_PRIORITY = "Lower Priority"
ALLOWED_LABELS = {REVIEW_FIRST, LOWER_PRIORITY}
ALLOWED_PARTITIONS = {"train", "test"}
CLASSIFICATION_THRESHOLD = 0.5
QUEUE_CAPTURE_FRACTIONS = (0.10, 0.25, 0.50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / "data" / "mhist" / "uni_embeddings.sqlite3",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "review_priority_head",
    )
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--max-iterations", type=int, default=3000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_embeddings(
    database: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    if not database.is_file():
        raise FileNotFoundError(f"Embedding database not found: {database}")
    with sqlite3.connect(database) as connection:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        failed = connection.execute(
            "SELECT COUNT(*) FROM embeddings WHERE status != 'complete'"
        ).fetchone()[0]
        rows = connection.execute(
            """
            SELECT partition, proxy_priority, source_label, embedding,
                   embedding_dimension
            FROM embeddings
            WHERE status = 'complete'
            ORDER BY image_name
            """
        ).fetchall()
    if failed:
        raise ValueError(f"Embedding database contains {failed} failed rows.")
    if not rows:
        raise ValueError("Embedding database contains no completed rows.")

    partitions: list[str] = []
    labels: list[str] = []
    audit_labels: list[str] = []
    embeddings: list[np.ndarray] = []
    for index, (partition, label, audit_label, blob, dimension) in enumerate(rows):
        if partition not in ALLOWED_PARTITIONS:
            raise ValueError(f"Row {index} has invalid partition {partition!r}.")
        if label not in ALLOWED_LABELS:
            raise ValueError(f"Row {index} has invalid proxy priority {label!r}.")
        if dimension != EMBEDDING_DIMENSION:
            raise ValueError(f"Row {index} has embedding dimension {dimension!r}.")
        vector = np.frombuffer(blob, dtype="<f4")
        if vector.shape != (EMBEDDING_DIMENSION,) or not np.isfinite(vector).all():
            raise ValueError(f"Row {index} has a malformed or non-finite embedding.")
        partitions.append(partition)
        labels.append(label)
        audit_labels.append(audit_label)
        embeddings.append(vector)

    matrix = np.stack(embeddings).astype(np.float32, copy=False)
    return (
        matrix,
        np.asarray(labels),
        np.asarray(partitions),
        np.asarray(audit_labels),
        metadata,
    )


def metrics_for(
    truth: np.ndarray,
    predicted: np.ndarray,
    review_first_probability: np.ndarray,
) -> dict[str, Any]:
    binary_truth = (truth == REVIEW_FIRST).astype(np.int8)
    actual_by_predicted = confusion_matrix(
        truth, predicted, labels=[LOWER_PRIORITY, REVIEW_FIRST]
    )
    true_lower, false_review_first, false_lower, true_review_first = (
        actual_by_predicted.ravel()
    )
    predicted_review_first = int(np.sum(predicted == REVIEW_FIRST))
    return {
        "sample_count": int(truth.size),
        "review_first_count": int(binary_truth.sum()),
        "lower_priority_count": int(truth.size - binary_truth.sum()),
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "review_first_precision": float(
            precision_score(truth, predicted, pos_label=REVIEW_FIRST, zero_division=0)
        ),
        "review_first_recall": float(
            recall_score(truth, predicted, pos_label=REVIEW_FIRST, zero_division=0)
        ),
        "review_first_f1": float(
            f1_score(truth, predicted, pos_label=REVIEW_FIRST, zero_division=0)
        ),
        "lower_priority_specificity": float(
            true_lower / max(true_lower + false_review_first, 1)
        ),
        "predicted_review_first_count": predicted_review_first,
        "predicted_review_first_fraction": float(
            predicted_review_first / max(truth.size, 1)
        ),
        "roc_auc": float(roc_auc_score(binary_truth, review_first_probability)),
        "average_precision": float(
            average_precision_score(binary_truth, review_first_probability)
        ),
        "brier_score": float(
            brier_score_loss(binary_truth, review_first_probability)
        ),
        "log_loss": float(log_loss(binary_truth, review_first_probability)),
        "confusion_matrix": {
            "row_axis": "predicted",
            "column_axis": "actual",
            "row_labels": [LOWER_PRIORITY, REVIEW_FIRST],
            "column_labels": [LOWER_PRIORITY, REVIEW_FIRST],
            "label_order": [LOWER_PRIORITY, REVIEW_FIRST],
            "values": actual_by_predicted.T.tolist(),
        },
        "review_first_capture_by_queue_fraction": review_first_capture_by_queue_fraction(
            truth, review_first_probability
        ),
    }


def review_first_capture_by_queue_fraction(
    truth: np.ndarray,
    review_first_probability: np.ndarray,
) -> list[dict[str, int | float]]:
    """Report how many positive proxy labels appear near the top of the queue."""

    truth = np.asarray(truth)
    scores = np.asarray(review_first_probability, dtype=np.float64)
    if truth.ndim != 1 or scores.shape != truth.shape or truth.size == 0:
        raise ValueError("Truth and score arrays must be non-empty one-dimensional peers.")
    if not np.isfinite(scores).all():
        raise ValueError("Review First scores must be finite.")
    binary_truth = (truth == REVIEW_FIRST).astype(np.int8)
    total_review_first = int(binary_truth.sum())
    # Stable ordering makes tied scores reproducible across runs.
    ranked_indices = np.argsort(-scores, kind="stable")
    captures: list[dict[str, int | float]] = []
    for fraction in QUEUE_CAPTURE_FRACTIONS:
        queue_size = max(1, int(np.ceil(truth.size * fraction)))
        captured = int(binary_truth[ranked_indices[:queue_size]].sum())
        captures.append(
            {
                "queue_fraction": fraction,
                "queue_size": queue_size,
                "captured_review_first_count": captured,
                "total_review_first_count": total_review_first,
                "capture_fraction": (
                    float(captured / total_review_first)
                    if total_review_first
                    else 0.0
                ),
            }
        )
    return captures


def main() -> int:
    args = parse_args()
    if args.threads < 1 or args.max_iterations < 1:
        raise ValueError("--threads and --max-iterations must be positive.")

    # Limit BLAS/OpenMP use without adding another runtime dependency.
    from threadpoolctl import threadpool_limits

    database = args.database.resolve()
    output_dir = args.output_dir.resolve()
    matrix, labels, partitions, audit_labels, embedding_metadata = load_embeddings(
        database
    )
    train_mask = partitions == "train"
    test_mask = partitions == "test"
    if int(train_mask.sum()) != 2175 or int(test_mask.sum()) != 977:
        raise ValueError(
            "Expected the official MHIST split (2175 train, 977 test), got "
            f"{int(train_mask.sum())} train and {int(test_mask.sum())} test."
        )
    if set(labels[train_mask]) != ALLOWED_LABELS or set(labels[test_mask]) != ALLOWED_LABELS:
        raise ValueError("Both partitions must contain both priority labels.")

    x_train, y_train = matrix[train_mask], labels[train_mask]
    x_test, y_test = matrix[test_mask], labels[test_mask]
    print(
        f"Training samples: {len(y_train)} | Test samples: {len(y_test)} | "
        f"Features: {matrix.shape[1]}",
        flush=True,
    )
    print(
        "Safety: target is pathologist agreement used as an experimental "
        "review-order proxy; disease labels are not model inputs.",
        flush=True,
    )

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=args.max_iterations,
                    random_state=42,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    with threadpool_limits(limits=args.threads):
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)

    class_order = list(model.named_steps["classifier"].classes_)
    review_first_index = class_order.index(REVIEW_FIRST)
    review_first_probability = probabilities[:, review_first_index]
    predictions = np.where(
        review_first_probability >= CLASSIFICATION_THRESHOLD,
        REVIEW_FIRST,
        LOWER_PRIORITY,
    )
    overall_metrics = metrics_for(y_test, predictions, review_first_probability)

    subgroup_metrics: dict[str, Any] = {}
    test_audit_labels = audit_labels[test_mask]
    for audit_label in sorted(set(test_audit_labels)):
        subgroup_mask = test_audit_labels == audit_label
        subgroup_truth = y_test[subgroup_mask]
        if len(set(subgroup_truth)) < 2:
            subgroup_metrics[audit_label] = {
                "sample_count": int(subgroup_mask.sum()),
                "note": "Only one proxy class is present; ranking metrics are undefined.",
            }
        else:
            subgroup_metrics[audit_label] = metrics_for(
                subgroup_truth,
                predictions[subgroup_mask],
                review_first_probability[subgroup_mask],
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "review_priority_head.joblib"
    metrics_path = output_dir / "metrics.json"
    metadata_path = output_dir / "metadata.json"
    joblib.dump(model, model_path)

    report = {
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "overall_test_metrics": overall_metrics,
        "audit_metrics_by_mhist_majority_vote_label": subgroup_metrics,
        "warning": (
            "Performance measures agreement-proxy prediction on MHIST, not clinical "
            "accuracy, diagnosis, cancer detection, or real-world review urgency."
        ),
    }
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": "StandardScaler + class-weighted logistic regression",
        "model_classes": class_order,
        "positive_class": REVIEW_FIRST,
        "decision_threshold": CLASSIFICATION_THRESHOLD,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "embedding_model": embedding_metadata.get("model_id", "unknown"),
        "embedding_database_sha256": sha256(database),
        "training_partition": "official MHIST train partition only",
        "evaluation_partition": "official MHIST test partition only",
        "train_sample_count": int(train_mask.sum()),
        "test_sample_count": int(test_mask.sum()),
        "target_definition": embedding_metadata.get("priority_definition", "unknown"),
        "source_disease_label_used_as_feature": False,
        "group_split_limitation": (
            "The supplied MHIST annotations do not include patient/case/slide group IDs; "
            "patient-level leakage cannot be independently verified."
        ),
        "intended_use": "research and education review-priority experiment only",
        "human_review_required": True,
        "sklearn_version": sklearn.__version__,
    }
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(overall_metrics, indent=2), flush=True)
    print(f"Model saved: {model_path}", flush=True)
    print(f"Metrics saved: {metrics_path}", flush=True)
    print(f"Metadata saved: {metadata_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
