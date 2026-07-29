from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app import Workspace, _projection_json
from pathology_ai.dashboard_visuals import EmbeddingProjection


class DashboardProjectionApiTests(unittest.TestCase):
    def test_serializes_only_the_approved_frontend_point_fields(self) -> None:
        space = Workspace()
        records = []
        for index in range(8):
            record = SimpleNamespace(
                image_id=f"id-{index}",
                display_name=f"image-{index}.png",
                triage=SimpleNamespace(suggested_priority="Review First"),
                attention=SimpleNamespace(
                    embedding_model="MahmoodLab/UNI",
                    embedding=(float(index), 1.0),
                ),
            )
            if index == 0:
                record.proxy_label = "HP"
            records.append(record)
            space.reviews[record.image_id] = {"reviewed": index == 1}
        space.batch = SimpleNamespace(records=records)
        projection = EmbeddingProjection(
            coordinates=tuple((float(index), float(index + 1)) for index in range(8)),
            method="test t-SNE",
            sample_count=8,
            input_dimension=2,
            reduced_dimension=2,
            perplexity=2.0,
        )

        with patch("app.build_tsne_projection", return_value=projection):
            result = _projection_json(space)

        self.assertTrue(result["available"])
        self.assertEqual(set(result["points"][0]), {
            "id", "name", "x", "y", "proxy_label", "suggested_priority",
            "embedding_model", "reviewed",
        })
        self.assertEqual(result["points"][0]["proxy_label"], "HP")
        self.assertIsNone(result["points"][1]["proxy_label"])
        self.assertNotIn("embedding", result["points"][0])


if __name__ == "__main__":
    unittest.main()
