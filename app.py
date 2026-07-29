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
from typing import Any, Mapping
from urllib.parse import unquote, urlparse
from uuid import uuid4
import webbrowser

from pathology_ai.attention import get_attention_provider
from pathology_ai.captions import CaptionInput, VisionCaptionService
from pathology_ai.dashboard_metrics import (
    DEFAULT_SCREENING_SECONDS_PER_IMAGE,
    MHIST_LIKE_DOMAIN,
    UNKNOWN_OR_OTHER_DOMAIN,
    build_operational_metrics,
)
from pathology_ai.dashboard_visuals import ProjectionUnavailable, build_tsne_projection
from pathology_ai.hibou_provider import get_hibou_provider_status
from pathology_ai.modal_provider import get_modal_provider_status
from pathology_ai.pipeline import BatchResult, UploadPayload, format_file_size, process_uploads
from pathology_ai.regions import (
    RegionAnalysis,
    analyze_variation_map,
    rank_agreement,
    resize_map,
    variation_map_from_heatmap,
)
from pathology_ai.review_export import build_review_export_csv, validate_optional_group_id, validate_review_fields
from pathology_ai.review_model import get_review_model, get_review_model_status
from pathology_ai.triage import PRIORITIES, priority_sort_key
from pathology_ai.uni_provider import get_uni_provider_status



def _server_settings(environ: Mapping[str, str] | None = None) -> tuple[str, int, bool]:
    """Return host, port, and local-browser behavior for local or hosted runs."""

    values = os.environ if environ is None else environ
    hosted = values.get("RENDER", "").lower() == "true" or bool(values.get("PORT"))
    host = values.get("PATHOLOGYAI_HOST", "0.0.0.0" if hosted else "127.0.0.1")
    port = int(values.get("PORT") or values.get("PATHOLOGYAI_PORT") or "8501")
    if not 1 <= port <= 65535:
        raise ValueError("The server port must be between 1 and 65535.")
    open_browser = not hosted and host in {"127.0.0.1", "localhost", "::1"}
    return host, port, open_browser


HOST, PORT, OPEN_BROWSER = _server_settings()
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
    provider_kind: str = field(default_factory=lambda: _preferred_provider_kind())
    use_review_model: bool = False
    computed_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    caption_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_map_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    projection_cache: dict[str, Any] | None = None


SESSIONS: dict[str, Workspace] = {}
LOCK = RLock()


def _preferred_provider_kind() -> str:
    """Prefer hosted Modal encoders, then local encoders, then the safety fallback."""

    if get_modal_provider_status("uni").ready:
        return "modal_uni"
    if get_modal_provider_status("hibou").ready:
        return "modal_hibou"
    if get_uni_provider_status().ready:
        return "uni"
    if get_hibou_provider_status().ready:
        return "hibou"
    return "deterministic"


def _static_file(url_path: str) -> Path | None:
    """Resolve a URL path to a regular file contained by the Vite output."""

    try:
        decoded = unquote(url_path, errors="strict")
    except UnicodeDecodeError:
        return None
    if decoded in {"", "/", "/index.html", "/app", "/app/"}:
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
    uni, hibou = get_uni_provider_status(), get_hibou_provider_status()
    modal_uni, modal_hibou = get_modal_provider_status("uni"), get_modal_provider_status("hibou")
    head = get_review_model_status()
    return {
        "uni": {"ready": uni.ready, "summary": uni.summary, "detail": uni.detail},
        "hibou": {"ready": hibou.ready, "summary": hibou.summary, "detail": hibou.detail},
        "modal_uni": {"ready": modal_uni.ready, "summary": modal_uni.summary, "detail": modal_uni.detail},
        "modal_hibou": {"ready": modal_hibou.ready, "summary": modal_hibou.summary, "detail": modal_hibou.detail},
        "review_model": {
            "ready": head.ready,
            "summary": head.summary,
            "detail": head.detail,
            "metrics": head.metrics,
            "evaluation_valid": head.evaluation_valid,
            "evaluation_error": head.evaluation_error,
        },
    }


def _provider_for_agreement(kind: str) -> Any | None:
    """Return a configured second encoder without silently using a fallback."""

    if kind == "uni":
        status = get_uni_provider_status()
        if status.ready:
            return get_attention_provider(provider_kind="uni")
        modal = get_modal_provider_status("uni")
        if modal.ready:
            from pathology_ai.modal_provider import ModalFeatureProvider

            return ModalFeatureProvider("uni")
    if kind == "hibou":
        status = get_hibou_provider_status()
        if status.ready:
            return get_attention_provider(provider_kind="hibou")
        modal = get_modal_provider_status("hibou")
        if modal.ready:
            from pathology_ai.modal_provider import ModalFeatureProvider

            return ModalFeatureProvider("hibou")
    return None


def _model_kind(model_name: str | None) -> str | None:
    lowered = (model_name or "").lower()
    if "uni" in lowered:
        return "uni"
    if "hibou" in lowered:
        return "hibou"
    return None


def _computed_features(space: Workspace, record: Any) -> dict[str, Any]:
    cached = space.computed_cache.get(record.image_id)
    if cached is not None:
        return cached

    attention = record.attention
    primary_map = attention.variation_map
    if primary_map is None:
        primary_map = variation_map_from_heatmap(attention.heatmap)
    primary_map = primary_map.astype("float32", copy=False)
    primary_kind = _model_kind(attention.embedding_model)
    maps = space.model_map_cache.setdefault(record.image_id, {})
    if primary_kind:
        maps[primary_kind] = primary_map
    analysis: RegionAnalysis = analyze_variation_map(
        primary_map,
        source=getattr(attention, "embedding_model", None) or "feature_variation",
        exclude_edge_regions="possible_edge_truncation" in getattr(record.quality, "advisory_codes", ()),
    )

    model_agreement_score: float | None = None
    model_agreement_method: str | None = None
    if primary_kind in {"uni", "hibou"}:
        other_kind = "hibou" if primary_kind == "uni" else "uni"
        if other_kind not in maps:
            try:
                provider = _provider_for_agreement(other_kind)
                if provider is not None:
                    other = provider.analyze(record.image)
                    other_map = other.variation_map
                    if other_map is None:
                        other_map = variation_map_from_heatmap(other.heatmap)
                    maps[other_kind] = other_map.astype("float32", copy=False)
            except Exception:
                maps.pop(other_kind, None)
        if other_kind in maps:
            common_shape = (
                max(8, min(primary_map.shape[0], maps[other_kind].shape[0])),
                max(8, min(primary_map.shape[1], maps[other_kind].shape[1])),
            )
            model_agreement_score = rank_agreement(
                resize_map(primary_map, common_shape),
                resize_map(maps[other_kind], common_shape),
            )
            model_agreement_method = "Shared-grid feature-variation rank agreement"

    quality_signal = {
        key: float(record.quality.metrics[key])
        for key in ("blur_score", "contrast")
        if key in record.quality.metrics
    }
    result = {
        "regions": [region.as_json() for region in analysis.regions],
        "priority_score": round(float(analysis.image_priority_score), 6),
        "summary": analysis.summary,
        "source": analysis.source,
        "model_agreement_score": None if model_agreement_score is None else round(float(model_agreement_score), 6),
        "model_agreement_available": model_agreement_score is not None,
        "model_agreement_method": model_agreement_method,
        "quality_signal": quality_signal,
    }
    space.computed_cache[record.image_id] = result
    return result


def _crop_region(image: Any, region: Mapping[str, Any]) -> Any:
    width, height = image.size
    left = max(0, min(width - 1, int(float(region["x"]) * width)))
    top = max(0, min(height - 1, int(float(region["y"]) * height)))
    right = max(left + 1, min(width, int((float(region["x"]) + float(region["width"])) * width)))
    bottom = max(top + 1, min(height, int((float(region["y"]) + float(region["height"])) * height)))
    return image.crop((left, top, right, bottom))


def _region_captions(space: Workspace, record: Any) -> dict[str, Any]:
    cached = space.caption_cache.get(record.image_id)
    if cached is not None:
        return cached
    computed = _computed_features(space, record)
    service = VisionCaptionService()
    rendered_regions: list[dict[str, Any]] = []
    for region in computed["regions"]:
        inputs = CaptionInput(
            region_id=int(region["region_id"]),
            contribution_percentage=float(region["contribution_percentage"]),
            priority_score=float(computed["priority_score"]),
            model_agreement_score=computed["model_agreement_score"],
            quality_signal=computed["quality_signal"],
            location=str(region["location"]),
        )
        rendered_regions.append({
            **region,
            "caption": service.generate(_crop_region(record.image, region), inputs),
        })
    result = {
        "image_id": record.image_id,
        "computed": computed,
        "regions": rendered_regions,
    }
    space.caption_cache[record.image_id] = result
    return result


def _record_json(space: Workspace, record: Any, review: dict[str, Any]) -> dict[str, Any]:
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
        "computed": _computed_features(space, record),
        "metadata_notes": record.metadata_notes,
        "review": review,
        "images": {kind: f"/api/images/{record.image_id}/{kind}" for kind in ("original", "overlay", "heatmap")},
    }


def _projection_json(space: Workspace) -> dict[str, Any]:
    """Return a browser-safe UNI projection without shipping raw embeddings."""

    records = space.batch.records if space.batch is not None else []
    eligible = [
        record for record in records
        if _model_kind(getattr(record.attention, "embedding_model", None)) == "uni"
        and getattr(record.attention, "embedding", None) is not None
    ]
    cache_key = "|".join(
        f"{record.image_id}:{len(record.attention.embedding or ())}"
        for record in eligible
    )
    if space.projection_cache is not None and space.projection_cache.get("key") == cache_key:
        cached = space.projection_cache["value"]
    else:
        if not eligible:
            value = {"available": False, "points": [], "method": None, "sample_count": 0, "full_count": 0, "error": "No UNI embeddings are available for this batch."}
        else:
            try:
                projection = build_tsne_projection(
                    tuple(record.attention.embedding for record in eligible),
                )
                coordinates = projection.coordinates
                x_values = [point[0] for point in coordinates]
                y_values = [point[1] for point in coordinates]
                x_min, x_max = min(x_values), max(x_values)
                y_min, y_max = min(y_values), max(y_values)
                x_span = x_max - x_min or 1.0
                y_span = y_max - y_min or 1.0
                points = []
                for record, (x, y) in zip(eligible, coordinates):
                    proxy_label = getattr(record, "proxy_label", None)
                    if proxy_label not in {"HP", "SSA"}:
                        proxy_label = None
                    review = space.reviews[record.image_id]
                    points.append({
                        "id": record.image_id,
                        "name": record.display_name,
                        "x": round((x - x_min) / x_span, 6),
                        "y": round((y - y_min) / y_span, 6),
                        "proxy_label": proxy_label,
                        "suggested_priority": record.triage.suggested_priority,
                        "embedding_model": record.attention.embedding_model,
                        "reviewed": bool(review.get("reviewed", False)),
                    })
                value = {"available": True, "points": points, "method": projection.method, "sample_count": projection.sample_count, "full_count": len(points), "error": None}
            except ProjectionUnavailable as exc:
                value = {"available": False, "points": [], "method": None, "sample_count": 0, "full_count": len(eligible), "error": str(exc)}
        space.projection_cache = {"key": cache_key, "value": value}
        cached = value

    if cached.get("available"):
        points = [
            {**point, "reviewed": bool(space.reviews.get(point["id"], {}).get("reviewed", point["reviewed"]))}
            for point in cached["points"]
        ]
        return {**cached, "points": points}
    return cached


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
            "records": [_record_json(space, record, space.reviews[record.image_id]) for record in batch.records],
            "skipped": [asdict(item) for item in batch.skipped],
            "metrics": asdict(metrics),
            "embedding_projection": _projection_json(space),
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
        if path == "/healthz":
            self._json({"status": "ok"})
            return
        space, session = self._space()
        try:
            if path == "/api/status":
                with LOCK:
                    self._json(_workspace_json(space), session=session)
            elif path.startswith("/api/images/") and path.endswith("/region-captions"):
                image_id = path.split("/")[3]
                record = self._record(space, image_id)
                with LOCK:
                    self._json(_region_captions(space, record), session=session)
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
                if kind not in {"deterministic", "uni", "hibou", "modal_uni", "modal_hibou"}:
                    raise ValueError("Unsupported feature provider.")
                requested_head = fields.get("use_review_model") == "true"
                if kind == "deterministic" and (
                    get_modal_provider_status("uni").ready or get_modal_provider_status("hibou").ready
                ) and not (get_uni_provider_status().ready or get_hibou_provider_status().ready):
                    kind = _preferred_provider_kind()
                if kind == "uni" and not get_uni_provider_status().ready and get_modal_provider_status("uni").ready:
                    kind = "modal_uni"
                elif kind == "hibou" and not get_hibou_provider_status().ready and get_modal_provider_status("hibou").ready:
                    kind = "modal_hibou"
                use_head = requested_head and kind == "uni" and get_review_model_status().ready
                remote_kind = {"modal_uni": "uni", "modal_hibou": "hibou"}.get(kind)
                if remote_kind:
                    from pathology_ai.modal_provider import ModalFeatureProvider

                    provider = ModalFeatureProvider(remote_kind, use_review_model=requested_head)
                else:
                    provider = get_attention_provider(provider_kind=kind)
                space.batch = process_uploads(files, provider, get_review_model() if use_head else None)
                space.reviews = {record.image_id: _review_defaults(record) for record in space.batch.records}
                space.computed_cache.clear()
                space.caption_cache.clear()
                space.model_map_cache.clear()
                space.projection_cache = None
                space.provider_kind, space.use_review_model = kind, requested_head and (use_head or remote_kind == "uni")
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
                space.computed_cache.clear()
                space.caption_cache.clear()
                space.model_map_cache.clear()
                space.projection_cache = None
                space.domain_context = "unknown_or_other"
                space.screening_seconds = DEFAULT_SCREENING_SECONDS_PER_IMAGE
                space.provider_kind = _preferred_provider_kind()
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
    if OPEN_BROWSER:
        Timer(0.35, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
