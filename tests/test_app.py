"""Streamlit UI smoke tests using only the installed Streamlit test API."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import unittest
import zipfile

from PIL import Image, ImageDraw
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
EXPECTED_DISCLAIMER = (
    "This research and education prototype provides review-priority suggestions only. "
    "It does not provide a medical diagnosis and does not replace review by a qualified "
    "pathologist."
)


def _image_bytes(size: tuple[int, int] = (256, 256), image_format: str = "PNG") -> bytes:
    image = Image.new("RGB", size, (224, 184, 205))
    draw = ImageDraw.Draw(image)
    tile = max(4, min(size) // 16)
    for y in range(0, size[1], tile):
        for x in range(0, size[0], tile):
            if ((x // tile) + (y // tile)) % 2:
                draw.rectangle(
                    (x, y, min(x + tile - 1, size[0] - 1), min(y + tile - 1, size[1] - 1)),
                    fill=(92, 42, 118),
                )
    output = BytesIO()
    image.save(output, format=image_format)
    return output.getvalue()


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return output.getvalue()


def _metrics(app: AppTest) -> dict[str, str]:
    return {metric.label: metric.value for metric in app.metric}


class StreamlitAppSmokeTests(unittest.TestCase):
    def test_app_starts_with_visible_safety_language(self) -> None:
        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.file_uploader), 1)
        self.assertTrue(any(item.value == EXPECTED_DISCLAIMER for item in app.warning))
        self.assertTrue(any("Research/Education Prototype" in item.value for item in app.caption))
        self.assertTrue(
            any(
                "No trained review-priority classifier is loaded" in item.value
                for item in app.caption
            )
        )

    def test_mixed_upload_and_no_supported_zip_render_without_errors(self) -> None:
        valid_png = _image_bytes()
        small_png = _image_bytes((64, 64))
        mixed_zip = _zip_bytes(
            {
                "nested/valid.png": valid_png,
                "broken.jpg": b"not an image",
                "notes.txt": b"unsupported",
            }
        )
        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        app.file_uploader[0].set_value(
            [
                ("valid.png", valid_png, "image/png"),
                ("small.png", small_png, "image/png"),
                ("unsupported.txt", b"hello", "text/plain"),
                ("corrupt.jpg", b"not an image", "image/jpeg"),
                ("mixed.zip", mixed_zip, "application/zip"),
            ]
        )
        app.run()

        values = _metrics(app)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(values["Total files uploaded"], "5")
        self.assertEqual(values["Valid images"], "3")
        self.assertEqual(values["Skipped or failed files"], "4")

        unsupported_zip = _zip_bytes(
            {"readme.txt": b"none", "table.csv": b"a,b\n1,2"}
        )
        empty_app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        empty_app.file_uploader[0].set_value(
            [("no-images.zip", unsupported_zip, "application/zip")]
        )
        empty_app.run()
        empty_values = _metrics(empty_app)
        self.assertEqual(len(empty_app.exception), 0)
        self.assertEqual(empty_values["Valid images"], "0")
        self.assertEqual(empty_values["Skipped or failed files"], "2")
        self.assertEqual(len(empty_app.error), 1)

    def test_viewer_configuration_and_manual_review_state(self) -> None:
        valid_png = _image_bytes()
        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        app.file_uploader[0].set_value([("valid.png", valid_png, "image/png")])
        app.run()

        charts = app.get("plotly_chart")
        self.assertEqual(len(charts), 2)
        config = json.loads(charts[0].proto.config)
        figure_spec = json.loads(charts[0].proto.spec)
        self.assertTrue(config["displayModeBar"])
        self.assertTrue(config["scrollZoom"])
        self.assertEqual(figure_spec["layout"]["dragmode"], "pan")

        app.text_area[0].input("student review note")
        app.selectbox[1].set_value("Lower Priority")
        app.checkbox[0].check()
        app.run()

        values = _metrics(app)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(values["Reviewed images"], "1")
        self.assertEqual(values["Lower Priority"], "1")
        self.assertEqual(app.text_area[0].value, "student review note")
        self.assertEqual(app.selectbox[1].value, "Lower Priority")
        self.assertTrue(app.checkbox[0].value)


if __name__ == "__main__":
    unittest.main()
