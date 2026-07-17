"""Extract resumable UNI embeddings for the local MHIST dataset.

The output is a local SQLite database under data/, which is ignored by Git.
UNI is used only as a feature encoder. The proxy priority in this file is based
on annotator agreement and is not a diagnosis or a validated clinical label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathology_ai.uni_provider import (  # noqa: E402
    UNI_EMBEDDING_DIMENSION,
    UNI_MODEL_ID,
    _DEFAULT_CHECKPOINT,
    _load_uni_model,
    prepare_uni_tensor,
)


IMAGE_COLUMN = "Image Name"
VOTE_COLUMN = "Number of Annotators who Selected SSA (Out of 7)"
PARTITION_COLUMN = "Partition"
SOURCE_LABEL_COLUMN = "Majority Vote Label"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=PROJECT_ROOT / "data" / "mhist" / "annotations.csv",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=PROJECT_ROOT / "data" / "mhist" / "images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "mhist" / "uni_embeddings.sqlite3",
    )
    parser.add_argument("--checkpoint", type=Path, default=_DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry images previously recorded as failed.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many pending images (useful for a smoke test).",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def proxy_priority(ssa_votes: int) -> tuple[str, int]:
    """Map seven-pathologist agreement to an experimental review-order proxy."""

    if not 0 <= ssa_votes <= 7:
        raise ValueError(f"SSA vote count must be from 0 to 7, got {ssa_votes}.")
    majority_agreement = max(ssa_votes, 7 - ssa_votes)
    priority = "Review First" if majority_agreement <= 5 else "Lower Priority"
    return priority, majority_agreement


def read_annotations(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Annotation CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {IMAGE_COLUMN, VOTE_COLUMN, PARTITION_COLUMN, SOURCE_LABEL_COLUMN}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Annotation CSV is missing columns: {sorted(missing)}")
        rows = list(reader)
    names = [row[IMAGE_COLUMN].strip() for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Annotation CSV contains duplicate image names.")
    return rows


def initialize_database(connection: sqlite3.Connection, metadata: dict[str, str]) -> None:
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            image_name TEXT PRIMARY KEY,
            partition TEXT NOT NULL,
            source_label TEXT NOT NULL,
            ssa_votes INTEGER NOT NULL,
            majority_agreement INTEGER NOT NULL,
            proxy_priority TEXT NOT NULL,
            embedding BLOB,
            embedding_dimension INTEGER,
            status TEXT NOT NULL CHECK(status IN ('complete', 'failed')),
            error TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    existing = dict(connection.execute("SELECT key, value FROM metadata"))
    conflicts = {
        key: (existing[key], value)
        for key, value in metadata.items()
        if key in existing and existing[key] != value
    }
    if conflicts:
        shown = ", ".join(
            f"{key}={old!r} (current {new!r})"
            for key, (old, new) in conflicts.items()
        )
        raise RuntimeError(
            f"Output database belongs to different inputs: {shown}. "
            "Choose another --output path or remove the old local output."
        )
    connection.executemany(
        "INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)", metadata.items()
    )
    connection.commit()


def save_failure(
    connection: sqlite3.Connection,
    row: dict[str, str],
    error: str,
) -> None:
    votes = int(row[VOTE_COLUMN])
    priority, agreement = proxy_priority(votes)
    connection.execute(
        """
        INSERT OR REPLACE INTO embeddings(
            image_name, partition, source_label, ssa_votes, majority_agreement,
            proxy_priority, embedding, embedding_dimension, status, error
        ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, 'failed', ?)
        """,
        (
            row[IMAGE_COLUMN].strip(),
            row[PARTITION_COLUMN].strip(),
            row[SOURCE_LABEL_COLUMN].strip(),
            votes,
            agreement,
            priority,
            error[:1000],
        ),
    )


def main() -> int:
    args = parse_args()
    if args.threads < 1 or args.batch_size < 1:
        raise ValueError("--threads and --batch-size must be positive integers.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be a positive integer.")

    annotations = args.annotations.resolve()
    image_dir = args.images.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"UNI checkpoint not found: {checkpoint}")
    rows = read_annotations(annotations)
    output.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "schema_version": "1",
        "model_id": UNI_MODEL_ID,
        "embedding_dimension": str(UNI_EMBEDDING_DIMENSION),
        "checkpoint_size": str(checkpoint.stat().st_size),
        "checkpoint_sha256": file_sha256(checkpoint),
        "annotations_sha256": file_sha256(annotations),
        "priority_definition": (
            "Review First when majority agreement is 4-5 of 7; "
            "Lower Priority when majority agreement is 6-7 of 7"
        ),
        "intended_use": "research and education review-priority experiment only",
    }

    connection = sqlite3.connect(output)
    try:
        initialize_database(connection, metadata)
        completed = {
            name
            for (name,) in connection.execute(
                "SELECT image_name FROM embeddings WHERE status='complete'"
            )
        }
        failed = {
            name
            for (name,) in connection.execute(
                "SELECT image_name FROM embeddings WHERE status='failed'"
            )
        }
        pending = [
            row
            for row in rows
            if row[IMAGE_COLUMN].strip() not in completed
            and (args.retry_failed or row[IMAGE_COLUMN].strip() not in failed)
        ]
        if args.limit is not None:
            pending = pending[: args.limit]
        print(
            f"Dataset: {len(rows)} images | complete: {len(completed)} | "
            f"failed: {len(failed)} | pending this run: {len(pending)}",
            flush=True,
        )
        if not pending:
            print(f"Nothing to do. Output: {output}", flush=True)
            return 0

        import torch

        torch.set_num_threads(args.threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        stat = checkpoint.stat()
        load_started = time.perf_counter()
        model, device = _load_uni_model(
            str(checkpoint), stat.st_size, stat.st_mtime_ns
        )
        print(
            f"Loaded {UNI_MODEL_ID} on {device} in "
            f"{time.perf_counter() - load_started:.1f}s; PyTorch threads={torch.get_num_threads()}",
            flush=True,
        )

        run_started = time.perf_counter()
        processed = 0
        failures_this_run = 0
        for offset in range(0, len(pending), args.batch_size):
            batch_rows = pending[offset : offset + args.batch_size]
            valid_rows: list[dict[str, str]] = []
            tensors = []
            for row in batch_rows:
                image_path = image_dir / row[IMAGE_COLUMN].strip()
                try:
                    with Image.open(image_path) as image:
                        image.load()
                        tensors.append(prepare_uni_tensor(image.convert("RGB")))
                    valid_rows.append(row)
                except Exception as exc:  # Continue and record individual bad inputs.
                    save_failure(connection, row, f"{type(exc).__name__}: {exc}")
                    failures_this_run += 1

            if tensors:
                tensor_batch = torch.stack(tensors).to(device)
                try:
                    with torch.inference_mode():
                        tokens = model.forward_features(tensor_batch)
                        batch_embeddings = (
                            model.forward_head(tokens, pre_logits=True)
                            .float()
                            .cpu()
                            .numpy()
                        )
                    expected = (len(valid_rows), UNI_EMBEDDING_DIMENSION)
                    if batch_embeddings.shape != expected:
                        raise RuntimeError(
                            f"Unexpected embedding shape {batch_embeddings.shape}; expected {expected}."
                        )
                    if not np.isfinite(batch_embeddings).all():
                        raise RuntimeError("UNI returned non-finite embedding values.")
                    for row, embedding in zip(valid_rows, batch_embeddings, strict=True):
                        votes = int(row[VOTE_COLUMN])
                        priority, agreement = proxy_priority(votes)
                        array = np.asarray(embedding, dtype="<f4")
                        connection.execute(
                            """
                            INSERT OR REPLACE INTO embeddings(
                                image_name, partition, source_label, ssa_votes,
                                majority_agreement, proxy_priority, embedding,
                                embedding_dimension, status, error
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'complete', NULL)
                            """,
                            (
                                row[IMAGE_COLUMN].strip(),
                                row[PARTITION_COLUMN].strip(),
                                row[SOURCE_LABEL_COLUMN].strip(),
                                votes,
                                agreement,
                                priority,
                                sqlite3.Binary(array.tobytes()),
                                UNI_EMBEDDING_DIMENSION,
                            ),
                        )
                except Exception as exc:
                    for row in valid_rows:
                        save_failure(connection, row, f"{type(exc).__name__}: {exc}")
                    failures_this_run += len(valid_rows)

            connection.commit()
            processed += len(batch_rows)
            elapsed = time.perf_counter() - run_started
            rate = processed / max(elapsed, 1e-9)
            remaining = (len(pending) - processed) / max(rate, 1e-9)
            if processed == len(pending) or processed % max(args.batch_size * 10, 1) == 0:
                print(
                    f"Progress: {processed}/{len(pending)} | {rate:.2f} images/s | "
                    f"ETA {remaining / 60:.1f} min | failures {failures_this_run}",
                    flush=True,
                )
    except KeyboardInterrupt:
        connection.commit()
        print("\nStopped safely. Run the same command to resume.", flush=True)
        return 130
    finally:
        connection.close()

    print(f"Finished. Embeddings saved to: {output}", flush=True)
    final_complete = connection_count = 0
    # Reopen read-only through a fresh connection because the main connection
    # has been closed by the finally block above.
    with sqlite3.connect(output) as summary_connection:
        final_complete = summary_connection.execute(
            "SELECT COUNT(*) FROM embeddings WHERE status='complete'"
        ).fetchone()[0]
        connection_count = summary_connection.execute(
            "SELECT COUNT(*) FROM embeddings WHERE status='failed'"
        ).fetchone()[0]
    print(
        f"Summary: complete={final_complete}, failed={connection_count}, "
        f"expected={len(rows)}",
        flush=True,
    )
    return 1 if connection_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
