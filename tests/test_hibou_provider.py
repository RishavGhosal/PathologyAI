"""Readiness and safe fallback tests for the optional Hibou-B provider."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pathology_ai.attention import DeterministicDemoAttentionProvider, get_attention_provider
from pathology_ai.hibou_provider import get_hibou_provider_status


class HibouProviderTests(unittest.TestCase):
    def test_missing_local_snapshot_is_not_ready_and_explains_fallback(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            model_dir = Path(temporary_directory) / "hibou-b"
            status = get_hibou_provider_status(model_dir)

        self.assertFalse(status.ready)
        self.assertIn("not found", status.summary)
        self.assertIn("deterministic fallback", status.summary)
        self.assertEqual(status.cache_key, "hibou:missing-local-snapshot")

    def test_requested_hibou_without_local_snapshot_uses_deterministic_provider(self) -> None:
        with patch(
            "pathology_ai.hibou_provider.get_hibou_provider_status",
            return_value=get_hibou_provider_status(Path("missing-hibou-snapshot")),
        ):
            provider = get_attention_provider(provider_kind="hibou")

        self.assertIsInstance(provider, DeterministicDemoAttentionProvider)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
