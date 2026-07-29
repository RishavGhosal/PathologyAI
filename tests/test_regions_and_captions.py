"""Tests for deterministic region computation and caption safety behavior."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from pathology_ai.captions import CaptionInput, VALIDATED_SYSTEM_PROMPT, VisionCaptionService, validate_caption
from pathology_ai.regions import analyze_variation_map, rank_agreement


class RegionAnalysisTests(unittest.TestCase):
    def test_returns_at_most_three_contiguous_regions_with_normalized_boxes(self) -> None:
        values = np.zeros((20, 20), dtype=np.float32)
        values[2:6, 3:8] = 0.95
        values[12:17, 12:18] = 0.8
        analysis = analyze_variation_map(values, source="fixture")

        self.assertEqual(len(analysis.regions), 2)
        self.assertEqual(analysis.regions[0].location, "upper-left")
        self.assertGreater(analysis.regions[0].contribution_percentage, 0)
        for region in analysis.regions:
            self.assertGreaterEqual(region.x, 0)
            self.assertLessEqual(region.x + region.width, 1)
            self.assertGreaterEqual(region.y, 0)
            self.assertLessEqual(region.y + region.height, 1)

    def test_edge_regions_are_removed_before_next_highest_region_is_selected(self) -> None:
        values = np.zeros((12, 12), dtype=np.float32)
        values[4:7, 0:3] = 1.0
        values[4:7, 7:10] = 0.92

        analysis = analyze_variation_map(
            values,
            source="fixture",
            exclude_edge_regions=True,
        )

        self.assertEqual(len(analysis.regions), 1)
        self.assertGreater(analysis.regions[0].x, 0.4)

    def test_rank_agreement_is_high_for_same_ranking_and_low_for_reverse(self) -> None:
        values = np.asarray([[0.1, 0.2], [0.8, 0.4]], dtype=np.float32)
        self.assertAlmostEqual(rank_agreement(values, values), 1.0)
        self.assertAlmostEqual(rank_agreement(values, 1.0 - values), 0.0)


class CaptionSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.region = CaptionInput(1, 42.0, 0.82, 0.3, {"blur_score": 100.0, "contrast": 40.0}, "upper-right")
        self.image = Image.new("RGB", (64, 64), (120, 80, 140))

    def test_valid_caption_is_returned_without_retry_and_prompt_is_exact(self) -> None:
        calls: list[dict] = []

        def request(body: dict) -> dict:
            calls.append(body)
            return {
                "priority_reason": "This region contributes the largest share (42%) of this image's priority score, which is why it is flagged first.",
                "visual_description": "Denser, darker texture with irregular boundary lines and stronger local contrast than the surrounding area.",
                "workflow_guidance": "UNI and Hibou-B show low agreement (0.30); consider routing this image to a second reviewer.",
                "fallback_triggered": False,
            }

        output = VisionCaptionService(request).generate(self.image, self.region)
        self.assertFalse(output["fallback_triggered"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["messages"][0]["content"], VALIDATED_SYSTEM_PROMPT)

    def test_github_token_selects_github_models_defaults(self) -> None:
        with patch.dict(
            "os.environ",
            {"GITHUB_TOKEN": "github_pat_test", "PATHOLOGYAI_CAPTION_MODEL": ""},
            clear=True,
        ):
            token, endpoint, model = VisionCaptionService._configuration()

        self.assertEqual(token, "github_pat_test")
        self.assertEqual(endpoint, "https://models.github.ai/inference/chat/completions")
        self.assertEqual(model, "openai/gpt-4.1")

    def test_openai_token_remains_supported_when_github_token_is_absent(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=True):
            token, endpoint, model = VisionCaptionService._configuration()

        self.assertEqual(token, "sk-test")
        self.assertEqual(endpoint, "https://api.openai.com/v1/chat/completions")
        self.assertEqual(model, "gpt-4o")

    def test_unsafe_output_retries_once_then_falls_back_without_sanitizing(self) -> None:
        calls = 0

        def request(_body: dict) -> dict:
            nonlocal calls
            calls += 1
            return {
                "priority_reason": "This region contributes the largest share (42%) of this image's priority score, which is why it is flagged first.",
                "visual_description": "This area appears abnormal and concerning.",
                "workflow_guidance": "This may indicate a problem worth investigating.",
                "fallback_triggered": False,
            }

        output = VisionCaptionService(request).generate(self.image, self.region)
        self.assertEqual(calls, 2)
        self.assertEqual(output, {
            "priority_reason": "This region contributes the largest share (42%) of this image's priority score, which is why it is flagged first.",
            "visual_description": None,
            "workflow_guidance": None,
            "fallback_triggered": True,
        })

    def test_validator_rejects_null_optional_fields_without_fallback(self) -> None:
        payload = {
            "priority_reason": "This region contributes 42% of this image's priority score and is included in the priority queue.",
            "visual_description": None,
            "workflow_guidance": "Route to a second reviewer.",
            "fallback_triggered": False,
        }
        self.assertIsNone(validate_caption(payload, self.region))


if __name__ == "__main__":
    unittest.main()
