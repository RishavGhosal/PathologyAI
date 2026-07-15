"""Regression tests for PathologyAI's deterministic processing pipeline.

The fixtures are generated in memory so the suite stays small, offline, and
repeatable.  ``unittest`` is the only test framework used.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image, ImageDraw

from pathology_ai.attention import (
    DEMO_PROVIDER_NAME,
    AttentionResult,
    DeterministicDemoAttentionProvider,
    get_attention_provider,
)
from pathology_ai.pipeline import UploadPayload, process_uploads
from pathology_ai.quality import QualityAssessment, assess_image_quality
from pathology_ai.triage import (
    LOWER_PRIORITY,
    NEEDS_BETTER_IMAGE,
    PRIORITIES,
    REVIEW_FIRST,
    assign_review_priority,
    priority_sort_key,
)
from pathology_ai.uni_provider import _letterbox, get_uni_provider_status


def _encode_image(image: Image.Image, image_format: str) -> bytes:
    stream = BytesIO()
    save_options = {"quality": 95} if image_format == "JPEG" else {}
    image.save(stream, format=image_format, **save_options)
    return stream.getvalue()


def _reviewable_image(marker: int = 255, size: int = 256) -> Image.Image:
    """Return a sharp, nonuniform image with visible background at each edge."""

    image = Image.new("RGB", (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    margin = size // 6
    tile = max(4, size // 24)
    for y in range(margin, size - margin, tile):
        for x in range(margin, size - margin, tile):
            color = (45, 35, 75) if ((x // tile) + (y // tile)) % 2 == 0 else (225, 150, 185)
            draw.rectangle(
                (x, y, min(x + tile - 1, size - margin - 1), min(y + tile - 1, size - margin - 1)),
                fill=color,
            )
    # A one-pixel marker lets a test provider select a score without materially
    # changing the quality assessment.
    image.putpixel((0, 0), (marker, marker, marker))
    return image


def _small_image() -> Image.Image:
    return _reviewable_image(size=64)


def _smooth_gradient() -> Image.Image:
    image = Image.new("L", (256, 256))
    image.putdata([x for _y in range(256) for x in range(256)])
    return image.convert("RGB")


def _cropped_looking_image() -> Image.Image:
    image = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(image)
    # The colored area meets the top and left frame edges while a substantial
    # white background remains visible, matching the conservative crop rule.
    draw.rectangle((0, 0, 169, 149), fill=(95, 35, 120))
    # Add internal detail so blur/uniformity do not confound the crop signal.
    for x in range(8, 168, 16):
        draw.line((x, 0, x, 149), fill=(215, 125, 175), width=3)
    return image


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return stream.getvalue()


class _MarkerAttentionProvider:
    """Test provider: black corner means high score; white means low score."""

    def analyze(self, image: Image.Image) -> AttentionResult:
        score = 0.90 if image.convert("RGB").getpixel((0, 0))[0] < 128 else 0.10
        preview = image.convert("RGB").copy()
        return AttentionResult(
            overlay=preview.copy(),
            heatmap=Image.new("RGB", preview.size, "black"),
            explanation="Deterministic test attention.",
            visual_complexity_score=score,
            provider_name="Test provider",
            is_demonstration=True,
        )


class _FailingAttentionProvider:
    def analyze(self, image: Image.Image) -> AttentionResult:
        raise RuntimeError("model artifact unavailable")


class FileValidationTests(unittest.TestCase):
    def test_valid_png_jpeg_and_tiff_are_decoded_with_metadata(self) -> None:
        image = _reviewable_image()
        payloads = [
            UploadPayload("sample.png", _encode_image(image, "PNG"), "image/png"),
            UploadPayload("sample.jpg", _encode_image(image, "JPEG"), "image/jpeg"),
            UploadPayload("sample.tiff", _encode_image(image, "TIFF"), "image/tiff"),
        ]

        result = process_uploads(payloads)

        self.assertEqual(result.uploaded_count, 3)
        self.assertEqual(len(result.records), 3)
        self.assertEqual(result.skipped, [])
        by_name = {record.file_name: record for record in result.records}
        self.assertEqual(by_name["sample.png"].file_type, "PNG")
        self.assertEqual(by_name["sample.jpg"].file_type, "JPEG")
        self.assertEqual(by_name["sample.tiff"].file_type, "TIFF")
        self.assertEqual(by_name["sample.png"].mime_type, "image/png")
        self.assertEqual(by_name["sample.jpg"].mime_type, "image/jpeg")
        self.assertEqual(by_name["sample.tiff"].mime_type, "image/tiff")
        for record in result.records:
            self.assertEqual((record.width, record.height), (256, 256))
            self.assertGreater(record.size_bytes, 0)

    def test_unsupported_empty_and_corrupt_uploads_are_rejected(self) -> None:
        result = process_uploads(
            [
                UploadPayload("notes.pdf", b"%PDF-1.7"),
                UploadPayload("empty.png", b""),
                UploadPayload("broken.jpg", b"this is not image data"),
            ]
        )

        self.assertEqual(result.records, [])
        self.assertEqual(len(result.skipped), 3)
        reasons = {item.file_name: item.reason for item in result.skipped}
        self.assertIn("Unsupported file type", reasons["notes.pdf"])
        self.assertIn("empty", reasons["empty.png"].lower())
        self.assertIn("corrupted or unreadable image", reasons["broken.jpg"].lower())

    def test_corrupt_zip_is_rejected(self) -> None:
        result = process_uploads([UploadPayload("broken.zip", b"not a zip archive")])

        self.assertEqual(result.records, [])
        self.assertEqual(len(result.skipped), 1)
        self.assertIn("corrupted or unreadable zip", result.skipped[0].reason.lower())

    def test_top_level_batch_decoded_pixel_budget_skips_excess_images(self) -> None:
        image_bytes = _encode_image(_reviewable_image(), "PNG")
        payloads = [
            UploadPayload("first.png", image_bytes),
            UploadPayload("second.png", image_bytes),
        ]

        with patch("pathology_ai.pipeline.MAX_BATCH_DECODED_PIXELS", 70_000):
            result = process_uploads(payloads)

        self.assertEqual(len(result.records), 1)
        self.assertEqual(len(result.skipped), 1)
        self.assertIn("decoded-image limit", result.skipped[0].reason)


class QualityAssessmentTests(unittest.TestCase):
    def test_sharp_reviewable_fixture_passes_quality_checks(self) -> None:
        quality = assess_image_quality(_reviewable_image())

        self.assertTrue(quality.adequate, quality.reasons)
        self.assertEqual(quality.reasons, ())

    def test_blurry_image_is_detected(self) -> None:
        quality = assess_image_quality(_smooth_gradient())

        self.assertFalse(quality.adequate)
        self.assertTrue(
            any("blurred or out of focus" in reason for reason in quality.reasons),
            quality.reasons,
        )

    def test_very_small_image_is_detected(self) -> None:
        quality = assess_image_quality(_small_image())

        self.assertFalse(quality.adequate)
        self.assertTrue(any("Very small dimensions" in reason for reason in quality.reasons))

    def test_excessive_darkness_and_brightness_are_detected(self) -> None:
        cases = (
            (Image.new("RGB", (256, 256), (5, 5, 5)), "excessively dark"),
            (Image.new("RGB", (256, 256), (250, 250, 250)), "excessively bright"),
        )
        for image, expected_text in cases:
            with self.subTest(expected_text=expected_text):
                quality = assess_image_quality(image)
                self.assertFalse(quality.adequate)
                self.assertTrue(
                    any(expected_text in reason for reason in quality.reasons),
                    quality.reasons,
                )

    def test_blank_or_nearly_uniform_image_is_detected(self) -> None:
        quality = assess_image_quality(Image.new("RGB", (256, 256), (128, 128, 128)))

        self.assertFalse(quality.adequate)
        self.assertTrue(
            any("blank or nearly uniform" in reason for reason in quality.reasons),
            quality.reasons,
        )

    def test_possible_crop_is_detected_conservatively(self) -> None:
        quality = assess_image_quality(_cropped_looking_image())

        self.assertFalse(quality.adequate)
        self.assertTrue(
            any("Possible edge truncation" in reason for reason in quality.reasons),
            quality.reasons,
        )

    def test_inadequate_quality_always_maps_to_needs_better_image(self) -> None:
        quality = assess_image_quality(_small_image())

        triage = assign_review_priority(quality, visual_complexity_score=1.0)

        self.assertEqual(triage.suggested_priority, "Needs Better Image")
        self.assertIn("clearer or more complete image", triage.explanation.lower())


class ZipProcessingTests(unittest.TestCase):
    def test_mixed_zip_processes_valid_root_and_nested_images_and_skips_bad_members(self) -> None:
        png = _encode_image(_reviewable_image(), "PNG")
        jpg = _encode_image(_reviewable_image(), "JPEG")
        payload = UploadPayload(
            "mixed.zip",
            _zip_bytes(
                {
                    "root.png": png,
                    "nested/inner.jpg": jpg,
                    "nested/readme.txt": b"not an image",
                    "broken.tiff": b"corrupt tiff bytes",
                    "empty.png": b"",
                    "../escape.png": png,
                    "__MACOSX/.DS_Store": b"metadata",
                }
            ),
            "application/zip",
        )

        result = process_uploads([payload])

        self.assertEqual(result.uploaded_count, 1)
        self.assertEqual(
            {record.file_name for record in result.records},
            {"root.png", "nested/inner.jpg"},
        )
        reasons = {item.file_name: item.reason for item in result.skipped}
        self.assertIn("Unsupported file type", reasons["nested/readme.txt"])
        self.assertIn("Corrupted or unreadable image", reasons["broken.tiff"])
        self.assertIn("empty", reasons["empty.png"].lower())
        self.assertIn("Unsafe archive path", reasons["../escape.png"])
        self.assertIn("System metadata", reasons["__MACOSX/.DS_Store"])

    def test_zip_with_no_supported_images_reports_every_skipped_file(self) -> None:
        payload = UploadPayload(
            "documents.zip",
            _zip_bytes(
                {
                    "readme.txt": b"hello",
                    "nested/table.csv": b"a,b\n1,2\n",
                }
            ),
        )

        result = process_uploads([payload])

        self.assertEqual(result.records, [])
        self.assertEqual(len(result.skipped), 2)
        self.assertTrue(
            all("Unsupported file type" in item.reason for item in result.skipped)
        )

    def test_empty_zip_reports_no_files(self) -> None:
        payload = UploadPayload("empty.zip", _zip_bytes({}))

        result = process_uploads([payload])

        self.assertEqual(result.records, [])
        self.assertEqual(len(result.skipped), 1)
        self.assertIn("contains no files", result.skipped[0].reason.lower())

    def test_batch_decoded_pixel_budget_skips_excess_images(self) -> None:
        image_bytes = _encode_image(_reviewable_image(), "PNG")
        payload = UploadPayload(
            "many-images.zip",
            _zip_bytes(
                {
                    "first.png": image_bytes,
                    "second.png": image_bytes,
                }
            ),
        )

        with patch("pathology_ai.pipeline.MAX_BATCH_DECODED_PIXELS", 70_000):
            result = process_uploads([payload])

        self.assertEqual(len(result.records), 1)
        self.assertEqual(len(result.skipped), 1)
        self.assertIn("decoded-image limit", result.skipped[0].reason)


class AttentionAndTriageTests(unittest.TestCase):
    def test_demonstration_attention_is_byte_for_byte_deterministic(self) -> None:
        provider = DeterministicDemoAttentionProvider()
        image = _reviewable_image()

        first = provider.analyze(image)
        second = provider.analyze(image.copy())

        self.assertEqual(first.provider_name, DEMO_PROVIDER_NAME)
        self.assertTrue(first.is_demonstration)
        self.assertEqual(first.visual_complexity_score, second.visual_complexity_score)
        self.assertEqual(first.explanation, second.explanation)
        self.assertEqual(first.overlay.size, image.size)
        self.assertEqual(first.overlay.tobytes(), second.overlay.tobytes())
        self.assertEqual(first.heatmap.tobytes(), second.heatmap.tobytes())
        self.assertIn("not a learned pathology finding", first.explanation)

    def test_failing_optional_provider_uses_deterministic_fallback(self) -> None:
        payload = UploadPayload(
            "image.png",
            _encode_image(_reviewable_image(), "PNG"),
        )

        result = process_uploads([payload], provider=_FailingAttentionProvider())

        self.assertEqual(len(result.records), 1)
        record = result.records[0]
        self.assertEqual(record.attention.provider_name, DEMO_PROVIDER_NAME)
        self.assertTrue(record.attention.is_demonstration)
        self.assertTrue(
            any("fallback was used (RuntimeError)" in note for note in record.metadata_notes),
            record.metadata_notes,
        )

    def test_all_three_exact_priority_labels_appear_and_sort_safely(self) -> None:
        provider = _MarkerAttentionProvider()
        payloads = [
            UploadPayload(
                "z_lower.png",
                _encode_image(_reviewable_image(marker=255), "PNG"),
            ),
            UploadPayload(
                "b_needs_better.png",
                _encode_image(Image.new("RGB", (256, 256), (128, 128, 128)), "PNG"),
            ),
            UploadPayload(
                "a_review_first.png",
                _encode_image(_reviewable_image(marker=0), "PNG"),
            ),
            UploadPayload(
                "A_lower.png",
                _encode_image(_reviewable_image(marker=255), "PNG"),
            ),
        ]

        result = process_uploads(payloads, provider=provider)

        labels = [record.triage.suggested_priority for record in result.records]
        self.assertEqual(
            labels,
            [REVIEW_FIRST, NEEDS_BETTER_IMAGE, LOWER_PRIORITY, LOWER_PRIORITY],
        )
        self.assertEqual(
            [record.file_name for record in result.records],
            ["a_review_first.png", "b_needs_better.png", "A_lower.png", "z_lower.png"],
        )
        self.assertEqual(set(labels), {"Review First", "Needs Better Image", "Lower Priority"})
        self.assertEqual(
            PRIORITIES,
            ("Review First", "Needs Better Image", "Lower Priority"),
        )
        self.assertGreater(priority_sort_key("unexpected label"), priority_sort_key(LOWER_PRIORITY))

    def test_triage_thresholds_produce_the_three_contract_labels(self) -> None:
        adequate = QualityAssessment(True, (), {})
        inadequate = QualityAssessment(False, ("fixture failed quality",), {})

        labels = {
            assign_review_priority(adequate, 0.90).suggested_priority,
            assign_review_priority(adequate, 0.10).suggested_priority,
            assign_review_priority(inadequate, 0.90).suggested_priority,
        }

        self.assertEqual(labels, set(PRIORITIES))


class UNIProviderConfigurationTests(unittest.TestCase):
    def test_uni_is_opt_in_and_default_provider_stays_lightweight(self) -> None:
        provider = get_attention_provider(prefer_uni=False)

        self.assertIsInstance(provider, DeterministicDemoAttentionProvider)

    def test_missing_local_checkpoint_has_clear_offline_fallback_status(self) -> None:
        with TemporaryDirectory() as directory:
            missing = Path(directory) / "pytorch_model.bin"
            status = get_uni_provider_status(missing)

        self.assertFalse(status.ready)
        self.assertIn("checkpoint not found", status.summary.lower())
        self.assertIn("will not download", status.detail.lower())

    def test_letterbox_preserves_aspect_ratio_and_reports_content_box(self) -> None:
        square, content_box = _letterbox(Image.new("RGB", (400, 200), "white"))

        self.assertEqual(square.size, (224, 224))
        left, top, right, bottom = content_box
        self.assertEqual((right - left, bottom - top), (224, 112))
        self.assertEqual((left, top), (0, 56))


if __name__ == "__main__":
    unittest.main()
