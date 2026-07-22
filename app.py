"""Local PathologyAI API server with a Vite-built React frontend."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import mimetypes
import os
from pathlib import Path
from threading import RLock, Timer
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4
import webbrowser

from pathology_ai.attention import get_attention_provider
from pathology_ai.dashboard_metrics import (
    DEFAULT_SCREENING_SECONDS_PER_IMAGE,
    MHIST_LIKE_DOMAIN,
    UNKNOWN_OR_OTHER_DOMAIN,
    build_operational_metrics,
)
from pathology_ai.hibou_provider import get_hibou_provider_status
from pathology_ai.pipeline import BatchResult, UploadPayload, format_file_size, process_uploads
from pathology_ai.review_export import build_review_export_csv, validate_optional_group_id, validate_review_fields
from pathology_ai.review_model import get_review_model, get_review_model_status
from pathology_ai.triage import PRIORITIES, priority_sort_key
from pathology_ai.uni_provider import get_uni_provider_status


HOST, PORT = "127.0.0.1", int(os.getenv("PATHOLOGYAI_PORT", "8501"))
PROJECT_DIR = Path(__file__).resolve().parent
DIST_DIR = PROJECT_DIR / "dist"
INDEX_PATH = DIST_DIR / "index.html"
DISCLAIMER = (
    "This research and education prototype provides review-priority suggestions only. "
    "It does not provide a medical diagnosis and does not replace review by a qualified "
    "pathologist."
)
DOMAIN_VALUES = {UNKNOWN_OR_OTHER_DOMAIN: "unknown_or_other", MHIST_LIKE_DOMAIN: "mhist_like_colorectal_polyp"}
MAX_REQUEST_BYTES = 110 * 1024 * 1024
STATIC_MIME_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".map": "application/json; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".wasm": "application/wasm",
    ".webmanifest": "application/manifest+json",
}


@dataclass
class Workspace:
    batch: BatchResult | None = None
    reviews: dict[str, dict[str, Any]] = field(default_factory=dict)
    domain_context: str = "unknown_or_other"
    screening_seconds: float = DEFAULT_SCREENING_SECONDS_PER_IMAGE
    provider_kind: str = "deterministic"
    use_review_model: bool = False


SESSIONS: dict[str, Workspace] = {}
LOCK = RLock()


def _static_file(url_path: str) -> Path | None:
    """Resolve a URL path to a regular file contained by the Vite output."""

    try:
        decoded = unquote(url_path, errors="strict")
    except UnicodeDecodeError:
        return None
    if decoded in {"", "/", "/index.html"}:
        candidate = INDEX_PATH
    else:
        if not decoded.startswith("/") or "\x00" in decoded:
            return None
        parts = decoded[1:].replace("\\", "/").split("/")
        if any(part in {"", ".", ".."} for part in parts):
            return None
        candidate = DIST_DIR.joinpath(*parts)

    try:
        resolved = candidate.resolve()
        resolved.relative_to(DIST_DIR.resolve())
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved if resolved.is_file() else None


def _static_content_type(path: Path) -> str:
    """Return deterministic browser-safe MIME types for Vite output files."""

    suffix = path.suffix.lower()
    if suffix in STATIC_MIME_TYPES:
        return STATIC_MIME_TYPES[suffix]
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _review_defaults(record: Any) -> dict[str, Any]:
    return {
        "priority": record.triage.suggested_priority,
        "notes": "",
        "group_id": "",
        "reviewed": False,
        "reviewed_at_utc": "",
    }


def _providers() -> dict[str, Any]:
    uni, hibou, head = get_uni_provider_status(), get_hibou_provider_status(), get_review_model_status()
    return {
        "uni": {"ready": uni.ready, "summary": uni.summary, "detail": uni.detail},
        "hibou": {"ready": hibou.ready, "summary": hibou.summary, "detail": hibou.detail},
        "review_model": {
            "ready": head.ready,
            "summary": head.summary,
            "detail": head.detail,
            "metrics": head.metrics,
            "evaluation_valid": head.evaluation_valid,
            "evaluation_error": head.evaluation_error,
        },
    }


def _record_json(record: Any, review: dict[str, Any]) -> dict[str, Any]:
    quality, triage, attention = record.quality, record.triage, record.attention
    return {
        "id": record.image_id,
        "name": record.display_name,
        "source_name": record.source_name,
        "file_type": record.file_type,
        "size": format_file_size(record.size_bytes),
        "dimensions": [record.width, record.height],
        "quality": {"adequate": quality.adequate, "reasons": quality.reasons, "advisories": quality.advisories, "metrics": quality.metrics, "issue_codes": quality.issue_codes, "advisory_codes": quality.advisory_codes},
        "triage": {"suggested_priority": triage.suggested_priority, "explanation": triage.explanation, "priority_source": triage.priority_source, "priority_method": triage.priority_method, "review_first_score": triage.review_first_score, "fallback_reason": triage.fallback_reason},
        "attention": {"provider_name": attention.provider_name, "explanation": attention.explanation, "overlay_caption": attention.overlay_caption, "embedding_model": attention.embedding_model, "embedding_available": attention.embedding is not None},
        "metadata_notes": record.metadata_notes,
        "review": review,
        "images": {kind: f"/api/images/{record.image_id}/{kind}" for kind in ("original", "overlay", "heatmap")},
    }


def _workspace_json(space: Workspace) -> dict[str, Any]:
    batch = space.batch
    if batch is None:
        return {"disclaimer": DISCLAIMER, "providers": _providers(), "batch": None}
    metrics = build_operational_metrics(
        batch, space.reviews,
        domain_declaration=(MHIST_LIKE_DOMAIN if space.domain_context == "mhist_like_colorectal_polyp" else UNKNOWN_OR_OTHER_DOMAIN),
        screening_seconds_per_image=space.screening_seconds,
        embedding_expected=space.provider_kind != "deterministic",
    )
    return {
        "disclaimer": DISCLAIMER,
        "providers": _providers(),
        "settings": {"domain_context": space.domain_context, "screening_seconds": space.screening_seconds, "provider_kind": space.provider_kind, "use_review_model": space.use_review_model},
        "batch": {
            "uploaded_count": batch.uploaded_count,
            "records": [_record_json(record, space.reviews[record.image_id]) for record in batch.records],
            "skipped": [asdict(item) for item in batch.skipped],
            "metrics": asdict(metrics),
        },
    }


def _image_bytes(image: Any) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _multipart(request: BaseHTTPRequestHandler) -> tuple[dict[str, str], list[UploadPayload]]:
    content_type = request.headers.get("Content-Type", "")
    length = int(request.headers.get("Content-Length", "0"))
    if length <= 0 or length > MAX_REQUEST_BYTES or "multipart/form-data" not in content_type:
        raise ValueError("Upload must be a multipart request within the 110 MB limit.")
    message = BytesParser(policy=default).parsebytes(
        f"Content-Type: {content_type}\r\n\r\n".encode() + request.rfile.read(length)
    )
    fields: dict[str, str] = {}
    files: list[UploadPayload] = []
    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue
        name = part.get_param("name", header="content-disposition") or ""
        filename = part.get_filename()
        value = part.get_payload(decode=True) or b""
        if filename:
            files.append(UploadPayload(filename, value, part.get_content_type()))
        elif name:
            fields[name] = value.decode("utf-8", errors="replace")
    return fields, files


def _snapshot_review(record: Any, review: dict[str, Any]) -> None:
    review.update({
        "reviewed": True,
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "group_id_format_validated": bool(str(review.get("group_id", "")).strip()),
        "suggested_priority_at_review": record.triage.suggested_priority,
        "priority_source_at_review": record.triage.priority_source,
        "priority_method_at_review": record.triage.priority_method,
        "priority_fallback_reason_at_review": record.triage.fallback_reason or "",
        "review_first_proxy_score_at_review": record.triage.review_first_score,
        "attention_provider_at_review": record.attention.provider_name,
        "embedding_model_at_review": record.attention.embedding_model or "",
        "embedding_at_review": record.attention.embedding,
    })


class AppHandler(BaseHTTPRequestHandler):
    server_version = "PathologyAI/2"

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _space(self) -> tuple[Workspace, str | None]:
        cookie = SimpleCookie(self.headers.get("Cookie"))
        session_id = cookie.get("pathology_ai_session")
        key = session_id.value if session_id else uuid4().hex
        with LOCK:
            return SESSIONS.setdefault(key, Workspace()), (None if session_id else key)

    def _send(self, status: int, data: bytes = b"", content_type: str = "application/json", session: str | None = None, download: str | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        if session:
            self.send_header("Set-Cookie", f"pathology_ai_session={session}; Path=/; SameSite=Lax; HttpOnly")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        self.wfile.write(data)

    def _json(self, payload: Any, status: int = HTTPStatus.OK, session: str | None = None) -> None:
        self._send(status, json.dumps(payload, default=list).encode(), session=session)

    def _error(self, message: str, status: int = HTTPStatus.BAD_REQUEST, session: str | None = None) -> None:
        self._json({"error": message}, status, session)

    def _record(self, space: Workspace, image_id: str) -> Any:
        if not space.batch:
            raise KeyError("Upload images before using the review workspace.")
        return next(record for record in space.batch.records if record.image_id == image_id)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        space, session = self._space()
        try:
            if path == "/api/status":
                with LOCK:
                    self._json(_workspace_json(space), session=session)
            elif path.startswith("/api/images/"):
                _, _, _, image_id, kind = path.split("/", 4)
                record = self._record(space, image_id)
                image = {"original": record.image, "overlay": record.attention.overlay, "heatmap": record.attention.heatmap}.get(kind)
                if image is None:
                    raise KeyError("Unknown image view.")
                self._send(HTTPStatus.OK, _image_bytes(image), "image/png", session)
            elif path == "/api/export":
                if not space.batch:
                    raise KeyError("There is no batch to export.")
                data = build_review_export_csv(space.batch.records, space.reviews, space.domain_context)
                self._send(HTTPStatus.OK, data, "text/csv; charset=utf-8", session, "pathologyai_review_labels.csv")
            else:
                static_file = _static_file(path)
                if static_file is None:
                    self._error("Route not found.", HTTPStatus.NOT_FOUND, session)
                else:
                    self._send(
                        HTTPStatus.OK,
                        static_file.read_bytes(),
                        _static_content_type(static_file),
                        session,
                    )
        except (KeyError, ValueError, StopIteration) as exc:
            self._error(str(exc), HTTPStatus.BAD_REQUEST, session)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        space, session = self._space()
        try:
            if path == "/api/upload":
                fields, files = _multipart(self)
                if not files:
                    raise ValueError("Choose at least one image or ZIP file.")
                kind = fields.get("provider_kind", "deterministic")
                if kind not in {"deterministic", "uni", "hibou"}:
                    raise ValueError("Unsupported feature provider.")
                provider = get_attention_provider(provider_kind=kind)
                use_head = fields.get("use_review_model") == "true" and kind == "uni" and get_review_model_status().ready
                space.batch = process_uploads(files, provider, get_review_model() if use_head else None)
                space.reviews = {record.image_id: _review_defaults(record) for record in space.batch.records}
                space.provider_kind, space.use_review_model = kind, use_head
                space.domain_context = fields.get("domain_context", "unknown_or_other") if fields.get("domain_context") in DOMAIN_VALUES.values() else "unknown_or_other"
                space.screening_seconds = max(0.0, min(600.0, float(fields.get("screening_seconds", DEFAULT_SCREENING_SECONDS_PER_IMAGE))))
                self._json(_workspace_json(space), HTTPStatus.CREATED, session)
            elif path == "/api/settings":
                payload = self._body_json()
                space.domain_context = payload.get("domain_context", space.domain_context) if payload.get("domain_context", space.domain_context) in DOMAIN_VALUES.values() else space.domain_context
                space.screening_seconds = max(0.0, min(600.0, float(payload.get("screening_seconds", space.screening_seconds))))
                self._json(_workspace_json(space), session=session)
            elif path.startswith("/api/reviews/") and path.endswith("/reopen"):
                record = self._record(space, path.split("/")[3])
                review = space.reviews[record.image_id]
                review["reviewed"], review["reviewed_at_utc"] = False, ""
                for key in tuple(review):
                    if key.endswith("_at_review"):
                        review.pop(key)
                self._json(_workspace_json(space), session=session)
            elif path.startswith("/api/reviews/"):
                record = self._record(space, path.split("/")[3])
                payload, review = self._body_json(), space.reviews[record.image_id]
                review.update({key: payload.get(key, review.get(key, "")) for key in ("priority", "notes", "group_id")})
                validate_review_fields(record, review)
                _snapshot_review(record, review)
                self._json(_workspace_json(space), session=session)
            elif path.startswith("/api/groups/"):
                record = self._record(space, path.split("/")[3])
                group_id = validate_optional_group_id(self._body_json().get("group_id", ""))
                for item in space.batch.records if space.batch else []:
                    if item.source_name == record.source_name:
                        space.reviews[item.image_id]["group_id"] = group_id
                self._json(_workspace_json(space), session=session)
            elif path == "/api/reset":
                space.batch = None
                space.reviews.clear()
                space.domain_context = "unknown_or_other"
                space.screening_seconds = DEFAULT_SCREENING_SECONDS_PER_IMAGE
                space.provider_kind = "deterministic"
                space.use_review_model = False
                self._json(_workspace_json(space), session=session)
            else:
                self._error("Route not found.", HTTPStatus.NOT_FOUND, session)
        except (KeyError, ValueError, StopIteration, json.JSONDecodeError) as exc:
            self._error(str(exc), HTTPStatus.BAD_REQUEST, session)

    def _body_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1_000_000:
            raise ValueError("Invalid JSON request body.")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object.")
        return value


def main() -> None:
    if not INDEX_PATH.is_file():
        raise FileNotFoundError(
            "dist/index.html is missing. Run `npm.cmd run build` before `python app.py`."
        )
    server = ThreadingHTTPServer((HOST, PORT), AppHandler)
    url = f"http://{HOST}:{PORT}"
    print(f"PathologyAI is running at {url}")
    Timer(0.35, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
