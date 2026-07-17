"""Validate reviewed exports and prepare group-safe UNI training artifacts.

The utility prepares reviewer-assigned review-order labels only.  It excludes
``Needs Better Image`` because that label belongs to the separate image-quality
workflow.  It does not prepare diagnostic, cancer, or disease targets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Sequence

import numpy as np


EMBEDDING_DIMENSION = 1024
FLOAT32_MAX = float(np.finfo(np.float32).max)
EMBEDDING_COLUMNS = tuple(
    f"uni_{index:04d}" for index in range(EMBEDDING_DIMENSION)
)
REVIEW_FIRST = "Review First"
LOWER_PRIORITY = "Lower Priority"
NEEDS_BETTER_IMAGE = "Needs Better Image"
TRAINING_LABELS = (LOWER_PRIORITY, REVIEW_FIRST)
ALLOWED_REVIEWER_LABELS = set(TRAINING_LABELS) | {NEEDS_BETTER_IMAGE}
N_SPLITS = 5
RANDOM_STATE = 42

MANIFEST_FILENAME = "manifest.csv"
EMBEDDINGS_FILENAME = "embeddings.npy"
REPORT_FILENAME = "validation_report.json"

# This validates a de-identified code format, not whether the value is truly
# de-identified.  That remains the reviewer's/process owner's responsibility.
GROUP_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class PreparationError(ValueError):
    """A user-correctable export validation error."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more reviewed-label CSV exports to combine.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for manifest.csv, embeddings.npy, and the report.",
    )
    return parser.parse_args(argv)


def _empty_report(input_paths: Sequence[Path]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "failed",
        "errors": [],
        "input_files": [str(path.resolve()) for path in input_paths],
        "counts": {
            "input_rows": 0,
            "exact_duplicate_rows_removed": 0,
            "needs_better_image_rows_excluded": 0,
            "training_rows": 0,
        },
        "label_balance": {},
        "group_balance": {},
        "folds": {
            "strategy": "StratifiedGroupKFold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
        },
        "deidentification_note": (
            "The utility validates group-ID format only. It cannot determine "
            "whether group IDs or other fields contain identifying information."
        ),
    }


def _write_report(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / REPORT_FILENAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _remove_split_artifacts(output_dir: Path) -> None:
    for filename in (MANIFEST_FILENAME, EMBEDDINGS_FILENAME):
        (output_dir / filename).unlink(missing_ok=True)


def _read_exports(
    input_paths: Sequence[Path], report: dict[str, Any]
) -> tuple[list[dict[str, str]], list[str]]:
    if not input_paths:
        raise PreparationError("At least one input CSV is required.")

    rows: list[dict[str, str]] = []
    field_order: list[str] = []
    field_set: set[str] = set()
    expected_embeddings = set(EMBEDDING_COLUMNS)

    for path in input_paths:
        if not path.is_file():
            raise PreparationError(f"Input CSV not found: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            if not fields:
                raise PreparationError(f"Input CSV has no header: {path}")
            if len(fields) != len(set(fields)):
                raise PreparationError(f"Input CSV has duplicate column names: {path}")
            required = {"image_id", "group_id", "reviewer_priority"}
            missing_required = sorted(required - set(fields))
            if missing_required:
                raise PreparationError(
                    f"Input CSV {path} is missing required columns: "
                    f"{', '.join(missing_required)}."
                )

            actual_embeddings = {field for field in fields if field.startswith("uni_")}
            missing_embeddings = sorted(expected_embeddings - actual_embeddings)
            extra_embeddings = sorted(actual_embeddings - expected_embeddings)
            if missing_embeddings or extra_embeddings:
                details: list[str] = []
                if missing_embeddings:
                    details.append(f"missing {len(missing_embeddings)} expected columns")
                if extra_embeddings:
                    details.append(
                        "unexpected columns " + ", ".join(extra_embeddings[:5])
                    )
                raise PreparationError(
                    f"Input CSV {path} does not contain exactly 1,024 UNI "
                    f"embedding columns ({'; '.join(details)})."
                )

            if "fold" in fields:
                raise PreparationError(
                    f"Input CSV {path} already contains the reserved 'fold' column."
                )
            for field in fields:
                if field not in field_set:
                    field_set.add(field)
                    field_order.append(field)

            for line_number, row in enumerate(reader, start=2):
                if None in row:
                    raise PreparationError(
                        f"Input CSV {path}, line {line_number} has more values than columns."
                    )
                normalized = {field: (value or "") for field, value in row.items()}
                normalized["__source"] = f"{path}:{line_number}"
                rows.append(normalized)

    report["counts"]["input_rows"] = len(rows)
    if not rows:
        raise PreparationError("The supplied exports contain no data rows.")
    return rows, field_order


def _deduplicate_rows(
    rows: Sequence[dict[str, str]],
    field_order: Sequence[str],
    report: dict[str, Any],
) -> list[dict[str, str]]:
    by_image_id: dict[str, tuple[tuple[str, str], ...]] = {}
    unique_rows: list[dict[str, str]] = []
    duplicate_count = 0

    for row in rows:
        image_id = row.get("image_id", "")
        if not image_id or image_id != image_id.strip():
            raise PreparationError(
                f"{row['__source']} has an empty image_id or surrounding whitespace."
            )
        canonical = tuple((field, row.get(field, "")) for field in field_order)
        previous = by_image_id.get(image_id)
        if previous is None:
            by_image_id[image_id] = canonical
            unique_rows.append(row)
        elif previous == canonical:
            duplicate_count += 1
        else:
            raise PreparationError(
                f"Conflicting duplicate image_id {image_id!r} was found at "
                f"{row['__source']}."
            )

    report["counts"]["exact_duplicate_rows_removed"] = duplicate_count
    return unique_rows


def _validate_review_fields(rows: Iterable[dict[str, str]]) -> None:
    for row in rows:
        source = row["__source"]
        group_id = row.get("group_id", "")
        if not GROUP_ID_PATTERN.fullmatch(group_id):
            raise PreparationError(
                f"{source} has invalid group_id {group_id!r}. Use 1-64 characters: "
                "letters, digits, underscore, or hyphen; start with a letter or "
                "digit. Format validation does not prove de-identification."
            )

        label = row.get("reviewer_priority", "")
        if label not in ALLOWED_REVIEWER_LABELS:
            raise PreparationError(
                f"{source} has invalid reviewer_priority {label!r}."
            )

        reviewed = row.get("reviewed")
        if reviewed is not None and reviewed.strip().lower() not in {"true", "1", "yes"}:
            raise PreparationError(f"{source} is not marked as reviewed.")

        suggested = row.get("suggested_priority", "")
        if suggested and suggested not in ALLOWED_REVIEWER_LABELS:
            raise PreparationError(
                f"{source} has invalid suggested_priority {suggested!r}."
            )
        if suggested and suggested != label and not row.get("reviewer_notes", "").strip():
            raise PreparationError(
                f"{source} overrides the suggested priority but has no reviewer notes."
            )
        if "priority_overridden" in row:
            raw_overridden = row["priority_overridden"].strip().lower()
            if raw_overridden not in {"true", "false"}:
                raise PreparationError(
                    f"{source} has invalid priority_overridden "
                    f"{row['priority_overridden']!r}."
                )
            declared_overridden = raw_overridden == "true"
            computed_overridden = bool(suggested and suggested != label)
            if suggested and declared_overridden != computed_overridden:
                raise PreparationError(
                    f"{source} has priority_overridden inconsistent with the "
                    "suggested and reviewer priorities."
                )
            if declared_overridden and not row.get("reviewer_notes", "").strip():
                raise PreparationError(
                    f"{source} declares a priority override but has no reviewer notes."
                )


def _embedding_for(row: dict[str, str]) -> np.ndarray:
    source = row["__source"]
    if "embedding_available" in row and row["embedding_available"].strip().lower() not in {
        "true",
        "1",
        "yes",
    }:
        raise PreparationError(f"{source} is missing a UNI embedding.")
    if "embedding_dimension" in row and row["embedding_dimension"].strip() != str(
        EMBEDDING_DIMENSION
    ):
        raise PreparationError(
            f"{source} reports embedding_dimension "
            f"{row['embedding_dimension']!r}; expected 1024."
        )

    values = np.empty(EMBEDDING_DIMENSION, dtype=np.float32)
    for index, column in enumerate(EMBEDDING_COLUMNS):
        raw_value = row.get(column, "")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise PreparationError(
                f"{source} has a missing or malformed value in {column}."
            ) from exc
        if not math.isfinite(value) or abs(value) > FLOAT32_MAX:
            raise PreparationError(
                f"{source} has a non-finite or non-float32-representable value "
                f"in {column}."
            )
        values[index] = value
    return values


def _assign_folds(
    rows: Sequence[dict[str, str]], report: dict[str, Any]
) -> np.ndarray:
    labels = np.asarray([row["reviewer_priority"] for row in rows])
    groups = np.asarray([row["group_id"] for row in rows])
    group_sets_by_label = {
        label: sorted(set(groups[labels == label])) for label in TRAINING_LABELS
    }
    unique_groups = sorted(set(groups))
    report["label_balance"] = {
        label: int(np.sum(labels == label)) for label in TRAINING_LABELS
    }
    report["group_balance"] = {
        "distinct_training_groups": len(unique_groups),
        "distinct_groups_by_label": {
            label: len(group_sets_by_label[label]) for label in TRAINING_LABELS
        },
        "groups_containing_both_labels": sum(
            1
            for group in unique_groups
            if set(labels[groups == group]) == set(TRAINING_LABELS)
        ),
    }

    if len(unique_groups) < N_SPLITS:
        raise PreparationError(
            f"Five-fold splitting requires at least 5 distinct groups; found "
            f"{len(unique_groups)}. Fold count is never reduced automatically."
        )
    for label in TRAINING_LABELS:
        label_group_count = len(group_sets_by_label[label])
        if label_group_count < N_SPLITS:
            raise PreparationError(
                f"Five-fold splitting requires {label!r} in at least 5 distinct "
                f"groups; found {label_group_count}. Fold count is never reduced "
                "automatically."
            )

    # Import only after dependency-free data validation. This keeps malformed
    # export reporting useful in the base app/CI environment, where the optional
    # training dependencies are intentionally not installed.
    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError as exc:
        raise PreparationError(
            "scikit-learn is required; install requirements-training.txt."
        ) from exc

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    folds = np.full(len(rows), -1, dtype=np.int8)
    dummy_features = np.zeros((len(rows), 1), dtype=np.uint8)
    for fold_index, (_, validation_indices) in enumerate(
        splitter.split(dummy_features, labels, groups)
    ):
        folds[validation_indices] = fold_index

    if np.any(folds < 0):
        raise PreparationError("Fold assignment did not assign every training row.")

    group_to_fold: dict[str, int] = {}
    fold_reports: list[dict[str, Any]] = []
    for fold_index in range(N_SPLITS):
        mask = folds == fold_index
        fold_labels = Counter(labels[mask])
        if set(fold_labels) != set(TRAINING_LABELS):
            raise PreparationError(
                f"Fold {fold_index} does not contain both training labels; no "
                "split-ready artifacts were produced."
            )
        fold_groups = sorted(set(groups[mask]))
        for group in fold_groups:
            previous_fold = group_to_fold.setdefault(group, fold_index)
            if previous_fold != fold_index:
                raise PreparationError(
                    f"Group {group!r} crosses folds {previous_fold} and {fold_index}."
                )
        fold_reports.append(
            {
                "fold": fold_index,
                "rows": int(mask.sum()),
                "groups": len(fold_groups),
                "label_counts": {
                    label: int(fold_labels[label]) for label in TRAINING_LABELS
                },
            }
        )

    if len(group_to_fold) != len(unique_groups):
        raise PreparationError("Not every group received exactly one fold assignment.")
    report["folds"]["assignments"] = fold_reports
    report["folds"]["group_leakage_detected"] = False
    return folds


def _write_success_artifacts(
    output_dir: Path,
    rows: Sequence[dict[str, str]],
    field_order: Sequence[str],
    embeddings: np.ndarray,
    folds: np.ndarray,
) -> None:
    manifest_fields = [
        field for field in field_order if field not in set(EMBEDDING_COLUMNS)
    ] + ["fold"]
    manifest_path = output_dir / MANIFEST_FILENAME
    embeddings_path = output_dir / EMBEDDINGS_FILENAME

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=manifest_fields,
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row, fold in zip(rows, folds, strict=True):
            output_row = {field: row.get(field, "") for field in manifest_fields}
            output_row["fold"] = int(fold)
            writer.writerow(output_row)
    with embeddings_path.open("wb") as handle:
        np.save(handle, embeddings.astype(np.float32, copy=False), allow_pickle=False)


def prepare_review_training_data(
    input_paths: Sequence[Path], output_dir: Path
) -> dict[str, Any]:
    """Prepare artifacts and always return/write a structured validation report."""

    ordered_paths = sorted((Path(path).resolve() for path in input_paths), key=str)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_split_artifacts(output_dir)
    report = _empty_report(ordered_paths)

    try:
        raw_rows, field_order = _read_exports(ordered_paths, report)
        unique_rows = _deduplicate_rows(raw_rows, field_order, report)
        _validate_review_fields(unique_rows)

        excluded_count = sum(
            row["reviewer_priority"] == NEEDS_BETTER_IMAGE for row in unique_rows
        )
        training_rows = [
            row for row in unique_rows if row["reviewer_priority"] in TRAINING_LABELS
        ]
        training_rows.sort(key=lambda row: row["image_id"])
        report["counts"]["needs_better_image_rows_excluded"] = excluded_count
        report["counts"]["training_rows"] = len(training_rows)
        if not training_rows:
            raise PreparationError(
                "No Review First or Lower Priority rows remain after excluding "
                "Needs Better Image."
            )

        embeddings = np.stack([_embedding_for(row) for row in training_rows])
        if embeddings.shape != (len(training_rows), EMBEDDING_DIMENSION):
            raise PreparationError("The combined embedding matrix has an invalid shape.")
        folds = _assign_folds(training_rows, report)
        _write_success_artifacts(
            output_dir,
            training_rows,
            field_order,
            embeddings,
            folds,
        )
        report["status"] = "success"
        report["artifacts"] = {
            "manifest": MANIFEST_FILENAME,
            "embeddings": EMBEDDINGS_FILENAME,
            "embedding_dtype": "float32",
            "embedding_shape": list(embeddings.shape),
            "row_alignment": (
                "Row N in manifest.csv corresponds to row N in embeddings.npy."
            ),
        }
    except Exception as exc:
        _remove_split_artifacts(output_dir)
        report["status"] = "failed"
        report["errors"] = [str(exc)]

    _write_report(output_dir, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = prepare_review_training_data(args.inputs, args.output_dir)
    if report["status"] == "success":
        print(
            f"Prepared {report['counts']['training_rows']} rows in "
            f"{args.output_dir.resolve()}.",
            flush=True,
        )
        return 0
    print(
        "Review-training data preparation failed: " + "; ".join(report["errors"]),
        file=sys.stderr,
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
