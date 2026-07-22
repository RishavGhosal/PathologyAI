"""Pure data transforms for dashboard visual diagnostics.

The Streamlit layer owns rendering.  This module keeps the expensive and
testable numerical work independent from UI state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


MIN_TSNE_SAMPLES = 8


class ProjectionUnavailable(ValueError):
    """Raised when a statistically useful t-SNE view cannot be produced."""


@dataclass(frozen=True)
class EmbeddingProjection:
    """Two-dimensional coordinates and the settings used to create them."""

    coordinates: tuple[tuple[float, float], ...]
    method: str
    sample_count: int
    input_dimension: int
    reduced_dimension: int
    perplexity: float


def build_tsne_projection(
    embeddings: Sequence[Sequence[float]],
    *,
    random_state: int = 42,
) -> EmbeddingProjection:
    """Build a deterministic PCA-initialized t-SNE projection.

    Fewer than eight observations are deliberately rejected: a tiny scatter
    would imply cluster structure that the current batch cannot support.
    """

    try:
        matrix = np.asarray(embeddings, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProjectionUnavailable("Model embeddings could not be converted to numbers.") from exc
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ProjectionUnavailable("Each model embedding must contain at least two values.")
    sample_count = int(matrix.shape[0])
    if sample_count < MIN_TSNE_SAMPLES:
        raise ProjectionUnavailable(
            f"At least {MIN_TSNE_SAMPLES} valid model embeddings are required for the "
            f"t-SNE view; this batch has {sample_count}."
        )
    if not np.isfinite(matrix).all():
        raise ProjectionUnavailable("Model embeddings must contain only finite values.")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= np.finfo(np.float32).eps):
        raise ProjectionUnavailable("Model embeddings must have a finite non-zero norm.")
    matrix = matrix / norms[:, np.newaxis]
    if not np.any(np.ptp(matrix, axis=0) > 0.0):
        raise ProjectionUnavailable(
            "The available model embeddings are identical, so a 2D projection would be misleading."
        )

    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ProjectionUnavailable(
            "The t-SNE view requires scikit-learn from requirements-training.txt."
        ) from exc

    reduced_dimension = min(50, sample_count - 1, int(matrix.shape[1]))
    reduced = PCA(
        n_components=reduced_dimension,
        random_state=random_state,
        svd_solver="full",
    ).fit_transform(matrix)
    perplexity = min(30.0, max(2.0, (sample_count - 1) / 3.0))
    coordinates = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        method="barnes_hut",
        n_jobs=1,
        random_state=random_state,
    ).fit_transform(reduced)
    if coordinates.shape != (sample_count, 2) or not np.isfinite(coordinates).all():
        raise ProjectionUnavailable("t-SNE returned malformed or non-finite coordinates.")

    return EmbeddingProjection(
        coordinates=tuple(
            (float(point[0]), float(point[1])) for point in coordinates
        ),
        method="L2-normalized, PCA-initialized t-SNE",
        sample_count=sample_count,
        input_dimension=int(matrix.shape[1]),
        reduced_dimension=reduced_dimension,
        perplexity=float(perplexity),
    )


__all__ = [
    "EmbeddingProjection",
    "MIN_TSNE_SAMPLES",
    "ProjectionUnavailable",
    "build_tsne_projection",
]
