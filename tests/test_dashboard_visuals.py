"""Tests for pure dashboard visual transforms."""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from pathology_ai.dashboard_visuals import (
    MIN_TSNE_SAMPLES,
    ProjectionUnavailable,
    build_tsne_projection,
)


SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None


class EmbeddingProjectionValidationTests(unittest.TestCase):
    def test_small_batch_is_not_presented_as_tsne_structure(self) -> None:
        embeddings = tuple(tuple([float(index)] * 1024) for index in range(3))

        with self.assertRaisesRegex(
            ProjectionUnavailable, f"At least {MIN_TSNE_SAMPLES}"
        ):
            build_tsne_projection(embeddings)

    def test_too_few_dimensions_and_nonfinite_values_are_rejected(self) -> None:
        with self.assertRaisesRegex(ProjectionUnavailable, "at least two"):
            build_tsne_projection(tuple((0.0,) for _ in range(8)))

        bad = np.zeros((8, 1024), dtype=np.float32)
        bad[0, 0] = np.nan
        with self.assertRaisesRegex(ProjectionUnavailable, "finite"):
            build_tsne_projection(bad)

    def test_identical_embeddings_are_not_randomly_separated(self) -> None:
        identical = np.ones((8, 1024), dtype=np.float32)

        with self.assertRaisesRegex(ProjectionUnavailable, "identical"):
            build_tsne_projection(identical)

    def test_zero_norm_embedding_is_rejected(self) -> None:
        embeddings = np.ones((8, 1024), dtype=np.float32)
        embeddings[0] = 0.0

        with self.assertRaisesRegex(ProjectionUnavailable, "non-zero norm"):
            build_tsne_projection(embeddings)

    @unittest.skipUnless(SKLEARN_AVAILABLE, "t-SNE requires optional scikit-learn")
    def test_projection_is_finite_deterministic_and_uses_valid_perplexity(self) -> None:
        generator = np.random.default_rng(42)
        embeddings = generator.normal(size=(8, 1024)).astype(np.float32)

        first = build_tsne_projection(embeddings)
        second = build_tsne_projection(embeddings)

        first_coordinates = np.asarray(first.coordinates)
        second_coordinates = np.asarray(second.coordinates)
        self.assertEqual(first_coordinates.shape, (8, 2))
        self.assertTrue(np.isfinite(first_coordinates).all())
        np.testing.assert_allclose(first_coordinates, second_coordinates, rtol=0.0, atol=0.0)
        self.assertLess(first.perplexity, first.sample_count)
        self.assertEqual(first.input_dimension, 1024)
        self.assertEqual(first.reduced_dimension, 7)


if __name__ == "__main__":
    unittest.main()
