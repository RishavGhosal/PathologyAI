from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "extract_uni_embeddings.py"
SPEC = importlib.util.spec_from_file_location("extract_uni_embeddings", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MHISTEmbeddingScriptTests(unittest.TestCase):
    def test_proxy_priority_boundaries(self):
        cases = [
        (0, "Lower Priority", 7),
        (1, "Lower Priority", 6),
        (2, "Review First", 5),
        (3, "Review First", 4),
        (4, "Review First", 4),
        (5, "Review First", 5),
        (6, "Lower Priority", 6),
        (7, "Lower Priority", 7),
        ]
        for votes, expected_priority, expected_agreement in cases:
            with self.subTest(votes=votes):
                self.assertEqual(
                    MODULE.proxy_priority(votes),
                    (expected_priority, expected_agreement),
                )

    def test_proxy_priority_rejects_invalid_vote_count(self):
        with self.assertRaises(ValueError):
            MODULE.proxy_priority(8)
