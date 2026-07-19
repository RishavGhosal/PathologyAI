"""Streamlit UI smoke tests using only the installed Streamlit test API."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import unittest
import zipfile

from PIL import Image, ImageDraw
from streamlit.testing.v1 import AppTest

from pathology_ai.pipeline import _stable_image_id
from pathology_ai.review_model import get_review_model_status
from pathology_ai.triage import LOWER_PRIORITY, REVIEW_FIRST


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


def _element_with_label(elements, label: str):
    return next(element for element in elements if element.label == label)


def _dataframe_with_column(app: AppTest, column: str):
    return next(frame.value for frame in app.dataframe if column in frame.value.columns)


class StreamlitAppSmokeTests(unittest.TestCase):
    def test_app_starts_with_visible_safety_language(self) -> None:
        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.file_uploader), 1)
        self.assertTrue(any(item.value == EXPECTED_DISCLAIMER for item in app.warning))
        self.assertTrue(any("Research/Education Prototype" in item.value for item in app.caption))
        status_messages = [
            item.value for group in (app.success, app.warning, app.caption) for item in group
        ]
        self.assertTrue(
            any("Experimental priority head" in message for message in status_messages)
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
        if app.toggle:
            app.toggle[0].set_value(False)
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
        self.assertEqual(
            [tab.label for tab in app.tabs],
            ["Review Queue", "Operational Dashboard", "Model Evaluation & Limits"],
        )
        self.assertEqual(values["Total files uploaded"], "5")
        self.assertEqual(values["Valid images"], "3")
        self.assertEqual(values["Corrupted or skipped files"], "4")
        self.assertEqual(values["UNI embedding successes"], "0")
        self.assertEqual(values["UNI embedding failures"], "0")
        self.assertEqual(values["UNI not attempted"], "3")
        self.assertEqual(values["Unknown or other tissue"], "3")

        unsupported_zip = _zip_bytes(
            {"readme.txt": b"none", "table.csv": b"a,b\n1,2"}
        )
        empty_app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        if empty_app.toggle:
            empty_app.toggle[0].set_value(False)
        empty_app.file_uploader[0].set_value(
            [("no-images.zip", unsupported_zip, "application/zip")]
        )
        empty_app.run()
        empty_values = _metrics(empty_app)
        self.assertEqual(len(empty_app.exception), 0)
        self.assertEqual(empty_values["Valid images"], "0")
        self.assertEqual(empty_values["Corrupted or skipped files"], "2")
        self.assertEqual(len(empty_app.error), 1)

    def test_viewer_configuration_and_manual_review_state(self) -> None:
        valid_png = _image_bytes()
        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        if app.toggle:
            app.toggle[0].set_value(False)
        app.file_uploader[0].set_value([("valid.png", valid_png, "image/png")])
        app.run()

        charts = app.get("plotly_chart")
        self.assertGreaterEqual(len(charts), 2)
        config = json.loads(charts[0].proto.config)
        figure_spec = json.loads(charts[0].proto.spec)
        self.assertTrue(config["displayModeBar"])
        self.assertTrue(config["scrollZoom"])
        self.assertEqual(figure_spec["layout"]["dragmode"], "pan")

        _element_with_label(app.selectbox, "Review status").set_value("All")
        _element_with_label(
            app.text_area, "Reviewer notes (kept in this browser session)"
        ).input("student review note")
        _element_with_label(
            app.text_input,
            "De-identified case/slide group ID (required to mark reviewed)",
        ).input("slide-group-001")
        _element_with_label(
            app.selectbox, "Confirm or override the suggested priority"
        ).set_value("Lower Priority")
        _element_with_label(app.button, "Save review").click()
        app.run()

        values = _metrics(app)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(values["Reviewed images"], "1")
        self.assertEqual(values["Lower Priority"], "1")
        self.assertEqual(
            _element_with_label(
                app.text_area, "Reviewer notes (kept in this browser session)"
            ).value,
            "student review note",
        )
        self.assertEqual(
            _element_with_label(
                app.text_input,
                "De-identified case/slide group ID (required to mark reviewed)",
            ).value,
            "slide-group-001",
        )
        self.assertEqual(
            _element_with_label(
                app.selectbox, "Confirm or override the suggested priority"
            ).value,
            "Lower Priority",
        )
        self.assertTrue(any("Reviewed for this Streamlit session" in x.value for x in app.success))
        per_image = _dataframe_with_column(app, "Effective reviewer priority")
        self.assertEqual(
            list(per_image.columns),
            [
                "Filename",
                "File type",
                "Dimensions",
                "File size",
                "Image Quality",
                "Attention source",
                "Priority source",
                "Suggested priority",
                "Effective reviewer priority",
                "Queue sort key",
                "Experimental agreement-proxy score",
                "Quality flags/codes",
                "Review status",
                "Case/slide group ID",
                "Override status",
            ],
        )
        self.assertEqual(per_image.iloc[0]["Review status"], "Reviewed")
        self.assertEqual(per_image.iloc[0]["Case/slide group ID"], "slide-group-001")

    def test_review_requires_group_id(self) -> None:
        valid_png = _image_bytes()
        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        if app.toggle:
            app.toggle[0].set_value(False)
        app.file_uploader[0].set_value([("valid.png", valid_png, "image/png")])
        app.run()

        _element_with_label(app.button, "Save review").click()
        app.run()

        self.assertTrue(any("group ID is required" in item.value for item in app.error))
        self.assertEqual(_metrics(app)["Reviewed images"], "0")

    def test_review_override_requires_notes(self) -> None:
        valid_png = _image_bytes()
        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        if app.toggle:
            app.toggle[0].set_value(False)
        app.file_uploader[0].set_value([("valid.png", valid_png, "image/png")])
        app.run()

        priority = _element_with_label(
            app.selectbox, "Confirm or override the suggested priority"
        )
        override = LOWER_PRIORITY if priority.value == REVIEW_FIRST else REVIEW_FIRST
        _element_with_label(
            app.text_input,
            "De-identified case/slide group ID (required to mark reviewed)",
        ).input("slide-group-001")
        priority.set_value(override)
        _element_with_label(app.button, "Save review").click()
        app.run()

        self.assertTrue(
            any("notes are required" in item.value for item in app.error)
        )
        self.assertEqual(_metrics(app)["Reviewed images"], "0")

    def test_bulk_grouping_fills_only_blanks_from_same_uploaded_source(self) -> None:
        valid_png = _image_bytes()
        batch_zip = _zip_bytes({"one.png": valid_png, "two.png": valid_png})
        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        if app.toggle:
            app.toggle[0].set_value(False)
        app.file_uploader[0].set_value(
            [
                ("batch.zip", batch_zip, "application/zip"),
                ("outside.png", valid_png, "image/png"),
            ]
        )
        app.run()

        one_id = _stable_image_id("batch.zip", "one.png", valid_png)
        two_id = _stable_image_id("batch.zip", "two.png", valid_png)
        outside_id = _stable_image_id("outside.png", "outside.png", valid_png)
        reviews = dict(app.session_state.filtered_state["reviews"])
        reviews[two_id] = dict(reviews[two_id])
        reviews[two_id]["group_id"] = "existing-group"
        app.session_state["reviews"] = reviews
        app.session_state[f"group_{two_id}"] = "existing-group"
        app.run()

        _element_with_label(
            app.selectbox, "Select an image for detailed review"
        ).set_value(one_id)
        app.run()
        _element_with_label(
            app.text_input,
            "De-identified case/slide group ID (required to mark reviewed)",
        ).input("new-group")
        _element_with_label(
            app.button,
            "Apply this ID to ungrouped images from the same uploaded source",
        ).click()
        app.run()

        reviews = app.session_state.filtered_state["reviews"]
        self.assertEqual(reviews[one_id]["group_id"], "new-group")
        self.assertEqual(reviews[two_id]["group_id"], "existing-group")
        self.assertEqual(reviews[outside_id]["group_id"], "")

    def test_save_next_only_targets_images_matching_current_filters(self) -> None:
        valid_png = _image_bytes()
        small_png = _image_bytes((64, 64))
        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        if app.toggle:
            app.toggle[0].set_value(False)
        app.file_uploader[0].set_value(
            [
                ("valid.png", valid_png, "image/png"),
                ("small.png", small_png, "image/png"),
            ]
        )
        app.run()

        selected_priority = _element_with_label(
            app.selectbox, "Confirm or override the suggested priority"
        ).value
        _element_with_label(app.selectbox, "Review status").set_value("All")
        _element_with_label(app.multiselect, "Review priorities").set_value(
            [selected_priority]
        )
        app.run()

        save_next = _element_with_label(app.button, "Save & next unreviewed")
        self.assertTrue(save_next.disabled)
        self.assertIn("current queue filters", save_next.help)

    def test_previous_and_next_navigation_stay_bounded_with_stale_events(self) -> None:
        valid_png = _image_bytes()
        names = ("01.png", "02.png", "03.png")
        image_ids = tuple(
            _stable_image_id(name, name, valid_png) for name in names
        )
        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        if app.toggle:
            app.toggle[0].set_value(False)
        app.file_uploader[0].set_value(
            [(name, valid_png, "image/png") for name in names]
        )
        app.run()

        selector = _element_with_label(
            app.selectbox, "Select an image for detailed review"
        )
        self.assertEqual(selector.value, image_ids[0])

        # Simulate a queued Next event whose button was rendered before the
        # selection reached the upper boundary.  It must read current state at
        # callback time rather than applying a stale precomputed neighbor.
        stale_next = _element_with_label(app.button, "Next")
        app.session_state["selected_image_id"] = image_ids[2]
        stale_next.click()
        app.run()
        selector = _element_with_label(
            app.selectbox, "Select an image for detailed review"
        )
        self.assertEqual(selector.value, image_ids[2])
        self.assertTrue(_element_with_label(app.button, "Next").disabled)

        # The symmetric stale Previous event must remain at the lower boundary.
        stale_previous = _element_with_label(app.button, "Previous")
        app.session_state["selected_image_id"] = image_ids[0]
        stale_previous.click()
        app.run()
        selector = _element_with_label(
            app.selectbox, "Select an image for detailed review"
        )
        self.assertEqual(selector.value, image_ids[0])
        self.assertTrue(_element_with_label(app.button, "Previous").disabled)

        # Normal navigation remains reversible after visiting either boundary.
        _element_with_label(app.button, "Next").click()
        app.run()
        self.assertEqual(
            _element_with_label(
                app.selectbox, "Select an image for detailed review"
            ).value,
            image_ids[1],
        )
        _element_with_label(app.button, "Next").click()
        app.run()
        self.assertEqual(
            _element_with_label(
                app.selectbox, "Select an image for detailed review"
            ).value,
            image_ids[2],
        )
        _element_with_label(app.button, "Previous").click()
        app.run()
        self.assertEqual(
            _element_with_label(
                app.selectbox, "Select an image for detailed review"
            ).value,
            image_ids[1],
        )
        _element_with_label(app.button, "Next").click()
        app.run()
        self.assertEqual(
            _element_with_label(
                app.selectbox, "Select an image for detailed review"
            ).value,
            image_ids[2],
        )
        self.assertEqual(len(app.exception), 0)

    def test_local_evaluation_dashboard_threshold_and_confusion_orientation(self) -> None:
        model_status = get_review_model_status()
        if not model_status.ready or not model_status.evaluation_valid:
            self.skipTest("Local validated review-head evaluation artifact is unavailable")

        valid_png = _image_bytes()
        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        if app.toggle:
            app.toggle[0].set_value(False)
        app.file_uploader[0].set_value([("valid.png", valid_png, "image/png")])
        app.run()

        metrics = _metrics(app)
        overall = model_status.evaluation_report["overall_test_metrics"]
        self.assertEqual(
            metrics["Classification threshold used"],
            f"{model_status.decision_threshold:.3f}",
        )
        self.assertEqual(metrics["Held-out test images"], str(overall["sample_count"]))
        self.assertEqual(
            metrics["Balanced accuracy"], f"{overall['balanced_accuracy']:.3f}"
        )

        heatmaps = []
        chart_axis_titles = []
        for chart in app.get("plotly_chart"):
            spec = json.loads(chart.proto.spec)
            chart_axis_titles.append(
                str(
                    spec.get("layout", {})
                    .get("xaxis", {})
                    .get("title", {})
                    .get("text", "")
                )
            )
            if spec.get("data", [{}])[0].get("type") == "heatmap":
                heatmaps.append(spec)
        self.assertEqual(len(heatmaps), 1)
        heatmap = heatmaps[0]["data"][0]
        matrix = overall["confusion_matrix"]
        self.assertEqual(heatmap["z"], matrix["values"])
        self.assertEqual(
            heatmap["x"],
            [f"Actual {label}" for label in matrix["column_labels"]],
        )
        self.assertEqual(
            heatmap["y"],
            [f"Predicted {label}" for label in matrix["row_labels"]],
        )
        if "roc_curve" in overall:
            self.assertIn("False-positive rate", chart_axis_titles)
            self.assertIn("Recall", chart_axis_titles)


if __name__ == "__main__":
    unittest.main()
