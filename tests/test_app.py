"""HTTP tests for the PathologyAI API and Vite production frontend."""

from __future__ import annotations

from http.cookiejar import CookieJar
from http.server import ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import re
from threading import Thread
import unittest
from urllib.error import HTTPError
from urllib.request import HTTPCookieProcessor, Request, build_opener
from zipfile import ZipFile

from PIL import Image, ImageDraw

from app import AppHandler, SESSIONS, _server_settings


def _image_bytes() -> bytes:
    image = Image.new("RGB", (256, 256), (224, 184, 205))
    drawing = ImageDraw.Draw(image)
    for x in range(0, 256, 16):
        drawing.line((x, 0, 255 - x, 255), fill=(92, 42, 118), width=5)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _multipart(
    files: tuple[tuple[str, bytes, str], ...] | None = None,
) -> tuple[bytes, str]:
    boundary = "PathologyAIBoundary"
    files = files or (("sample.png", _image_bytes(), "image/png"),)
    lines = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"provider_kind\"\r\n\r\ndeterministic\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"use_review_model\"\r\n\r\nfalse\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"domain_context\"\r\n\r\nunknown_or_other\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"screening_seconds\"\r\n\r\n30\r\n".encode(),
    ]
    for filename, content, content_type in files:
        lines.extend(
            (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n".encode(),
                content,
                b"\r\n",
            )
        )
    lines.append(f"--{boundary}--\r\n".encode())
    return b"".join(lines), boundary


def _zip_bytes() -> bytes:
    output = BytesIO()
    with ZipFile(output, "w") as archive:
        archive.writestr("one.png", _image_bytes())
        archive.writestr("two.png", _image_bytes())
    return output.getvalue()


class StandaloneAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        SESSIONS.clear()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), AppHandler)
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self) -> None:
        SESSIONS.clear()
        self.cookie_jar = CookieJar()
        self.opener = build_opener(HTTPCookieProcessor(self.cookie_jar))

    def request_json(self, path: str, payload: dict | None = None) -> dict:
        body = None if payload is None else json.dumps(payload).encode()
        request = Request(self.url + path, data=body, method="POST" if body else "GET")
        if body:
            request.add_header("Content-Type", "application/json")
        with self.opener.open(request) as response:
            return json.loads(response.read())

    def test_vite_index_assets_and_session_cookie(self) -> None:
        with self.opener.open(self.url + "/") as response:
            page = response.read().decode()
            self.assertEqual(response.headers.get_content_type(), "text/html")
            self.assertIn("pathology_ai_session=", response.headers["Set-Cookie"])
        self.assertIn('id="root"', page)
        with self.opener.open(self.url + "/index.html") as response:
            self.assertEqual(response.headers.get_content_type(), "text/html")
            self.assertEqual(response.read().decode(), page)

        asset_paths = re.findall(r'(?:src|href)="(/assets/[^"]+)"', page)
        script_paths = [path for path in asset_paths if path.endswith(".js")]
        style_paths = [path for path in asset_paths if path.endswith(".css")]
        self.assertTrue(script_paths, "The Vite index must reference a built JavaScript asset.")
        self.assertTrue(style_paths, "The Vite index must reference a built CSS asset.")
        for path in script_paths + style_paths:
            self.assertRegex(path, r"^/assets/.+-[A-Za-z0-9_-]+\.(?:js|css)$")
            with self.opener.open(self.url + path) as response:
                expected = "text/javascript" if path.endswith(".js") else "text/css"
                self.assertEqual(response.headers.get_content_type(), expected)
                self.assertGreater(len(response.read()), 0)
                self.assertIsNone(response.headers.get("Set-Cookie"))

        request = Request(self.url + "/api/status")
        with self.opener.open(request) as response:
            status = json.loads(response.read())
            self.assertIsNone(response.headers.get("Set-Cookie"))
        self.assertIsNone(status["batch"])
        self.assertIn("does not provide a medical diagnosis", status["disclaimer"])
        self.assertEqual(len(self.cookie_jar), 1)

    def test_health_check_does_not_create_a_session(self) -> None:
        with self.opener.open(self.url + "/healthz") as response:
            self.assertEqual(response.headers.get_content_type(), "application/json")
            self.assertEqual(json.loads(response.read()), {"status": "ok"})
            self.assertIsNone(response.headers.get("Set-Cookie"))
        self.assertEqual(SESSIONS, {})

    def test_missing_and_traversal_static_paths_are_rejected(self) -> None:
        for path in (
            "/assets/does-not-exist.js",
            "/%2e%2e/app.py",
            "/assets/%2e%2e/index.html",
            "/assets%5c..%5cindex.html",
        ):
            with self.subTest(path=path), self.assertRaises(HTTPError) as raised:
                self.opener.open(self.url + path)
            self.assertEqual(raised.exception.code, 404)
            self.assertEqual(
                json.loads(raised.exception.read()),
                {"error": "Route not found."},
            )

    def test_upload_review_images_and_export(self) -> None:
        body, boundary = _multipart()
        request = Request(self.url + "/api/upload", data=body, method="POST")
        request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        with self.opener.open(request) as response:
            created = json.loads(response.read())
        record = created["batch"]["records"][0]
        self.assertEqual(record["name"], "sample.png")
        self.assertEqual(record["attention"]["provider_name"], "Deterministic demonstration attention")
        for kind in ("original", "overlay", "heatmap"):
            with self.opener.open(self.url + record["images"][kind]) as response:
                self.assertEqual(response.headers["Content-Type"], "image/png")
                self.assertTrue(response.read().startswith(b"\x89PNG"))

        settings = self.request_json(
            "/api/settings",
            {"domain_context": "mhist_like_colorectal_polyp", "screening_seconds": 45},
        )
        self.assertEqual(settings["settings"]["domain_context"], "mhist_like_colorectal_polyp")
        self.assertEqual(settings["settings"]["screening_seconds"], 45)

        reviewed = self.request_json(
            f"/api/reviews/{record['id']}",
            {"priority": record["triage"]["suggested_priority"], "notes": "Looks ready", "group_id": "batch_01"},
        )
        self.assertEqual(reviewed["batch"]["metrics"]["reviewed_count"], 1)
        self.assertTrue(reviewed["batch"]["records"][0]["review"]["reviewed"])

        reopened = self.request_json(f"/api/reviews/{record['id']}/reopen", {})
        self.assertFalse(reopened["batch"]["records"][0]["review"]["reviewed"])
        self.assertEqual(reopened["batch"]["metrics"]["reviewed_count"], 0)
        reviewed = self.request_json(
            f"/api/reviews/{record['id']}",
            {"priority": record["triage"]["suggested_priority"], "notes": "Looks ready", "group_id": "batch_01"},
        )
        grouped = self.request_json(f"/api/groups/{record['id']}", {"group_id": "case_02"})
        self.assertEqual(grouped["batch"]["records"][0]["review"]["group_id"], "case_02")

        with self.opener.open(self.url + "/api/export") as response:
            csv = response.read().decode()
            self.assertEqual(response.headers.get_content_type(), "text/csv")
            self.assertIn("pathologyai_review_labels.csv", response.headers["Content-Disposition"])
        self.assertIn("reviewer_notes", csv)
        self.assertIn("Looks ready", csv)

        reset = self.request_json("/api/reset", {})
        self.assertIsNone(reset["batch"])

    def test_group_applies_to_every_record_from_same_upload_source(self) -> None:
        body, boundary = _multipart(
            (("batch.zip", _zip_bytes(), "application/zip"),)
        )
        request = Request(self.url + "/api/upload", data=body, method="POST")
        request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        with self.opener.open(request) as response:
            created = json.loads(response.read())

        records = created["batch"]["records"]
        self.assertEqual(len(records), 2)
        self.assertEqual({record["source_name"] for record in records}, {"batch.zip"})
        grouped = self.request_json(
            f"/api/groups/{records[0]['id']}", {"group_id": "case_02"}
        )
        self.assertEqual(
            [record["review"]["group_id"] for record in grouped["batch"]["records"]],
            ["case_02", "case_02"],
        )

    def test_source_file_has_no_streamlit_dependency(self) -> None:
        source = Path(__file__).resolve().parents[1] / "app.py"
        self.assertNotIn("import streamlit", source.read_text(encoding="utf-8"))


class ServerSettingsTests(unittest.TestCase):
    def test_local_defaults_open_the_browser(self) -> None:
        self.assertEqual(_server_settings({}), ("127.0.0.1", 8501, True))

    def test_render_uses_public_host_and_platform_port(self) -> None:
        self.assertEqual(
            _server_settings({"RENDER": "true", "PORT": "10000"}),
            ("0.0.0.0", 10000, False),
        )

    def test_explicit_local_settings_are_preserved(self) -> None:
        self.assertEqual(
            _server_settings({"PATHOLOGYAI_HOST": "localhost", "PATHOLOGYAI_PORT": "8504"}),
            ("localhost", 8504, True),
        )

    def test_invalid_port_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 1 and 65535"):
            _server_settings({"PORT": "70000"})
