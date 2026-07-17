"""Safe file validation, in-memory ZIP handling, and image processing."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import PurePosixPath
import warnings
import zipfile

from PIL import Image, ImageOps, UnidentifiedImageError

from .attention import (
    AttentionProvider,
    AttentionResult,
    DeterministicDemoAttentionProvider,
    get_attention_provider,
)
from .quality import QualityAssessment, assess_image_quality
from .triage import TriageResult, assign_review_priority, priority_sort_key
from .review_model import LocalPrototypeReviewHead


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
SUPPORTED_UPLOAD_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | {".zip"}
SUPPORTED_IMAGE_FORMATS = {"PNG", "JPEG", "TIFF"}
FORMAT_TO_MIME = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "TIFF": "image/tiff",
}
EXTENSION_TO_FORMAT = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}

MAX_IMAGE_BYTES = 40 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_BATCH_DECODED_PIXELS = 60_000_000
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 300
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 250 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200.0


@dataclass(frozen=True)
class UploadPayload:
    name: str
    data: bytes
    mime_type: str = ""


@dataclass(frozen=True)
class SkippedFile:
    source_name: str
    file_name: str
    reason: str


@dataclass
class ImageRecord:
    image_id: str
    source_name: str
    display_name: str
    file_name: str
    file_type: str
    mime_type: str
    width: int
    height: int
    size_bytes: int
    image: Image.Image
    quality: QualityAssessment
    attention: AttentionResult
    triage: TriageResult
    metadata_notes: tuple[str, ...] = ()


@dataclass
class BatchResult:
    uploaded_count: int
    records: list[ImageRecord]
    skipped: list[SkippedFile]


def _extension(file_name: str) -> str:
    normalized = file_name.replace("\\", "/")
    base_name = normalized.rsplit("/", 1)[-1]
    if "." not in base_name:
        return ""
    return f".{base_name.rsplit('.', 1)[-1].lower()}"


def _unsupported_reason(file_name: str) -> str:
    ext = _extension(file_name)
    shown = ext or "no extension"
    return (
        f"Unsupported file type ({shown}). Supported types are PNG, JPG/JPEG, "
        "TIFF, and ZIP."
    )


def _stable_image_id(source_name: str, file_name: str, data: bytes) -> str:
    digest = sha256()
    digest.update(source_name.encode("utf-8", errors="replace"))
    digest.update(b"\0")
    digest.update(file_name.encode("utf-8", errors="replace"))
    digest.update(b"\0")
    digest.update(data)
    return digest.hexdigest()


def _safe_archive_member(member_name: str) -> bool:
    normalized = member_name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or normalized.startswith("/"):
        return False
    if any(part == ".." for part in path.parts):
        return False
    if path.parts and ":" in path.parts[0]:
        return False
    return True


def _metadata_file(member_name: str) -> bool:
    parts = PurePosixPath(member_name.replace("\\", "/")).parts
    return "__MACOSX" in parts or (parts and parts[-1] == ".DS_Store")


def _decode_image(data: bytes) -> tuple[Image.Image, str, int, tuple[str, ...]]:
    metadata_notes: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(BytesIO(data)) as probe:
            detected_format = (probe.format or "").upper()
            if detected_format not in SUPPORTED_IMAGE_FORMATS:
                raise ValueError(
                    f"Detected image format {detected_format or 'unknown'} is not supported."
                )
            width, height = probe.size
            if width <= 0 or height <= 0:
                raise ValueError("Image dimensions are invalid.")
            if width * height > MAX_IMAGE_PIXELS:
                raise ValueError(
                    f"Image is too large for this MVP ({width} × {height} px; "
                    f"limit is {MAX_IMAGE_PIXELS:,} pixels)."
                )
            frame_count = int(getattr(probe, "n_frames", 1))
            probe.verify()

        with Image.open(BytesIO(data)) as decoded:
            if int(getattr(decoded, "n_frames", 1)) > 1:
                decoded.seek(0)
                metadata_notes.append(
                    f"Multi-page {detected_format} detected; this MVP previews the first frame."
                )
            decoded.load()
            oriented = ImageOps.exif_transpose(decoded)
            image = oriented.convert("RGB").copy()

    return image, detected_format, frame_count, tuple(metadata_notes)


def _process_image_bytes(
    data: bytes,
    file_name: str,
    source_name: str,
    display_name: str,
    provider: AttentionProvider,
    review_model: LocalPrototypeReviewHead | None = None,
) -> tuple[ImageRecord | None, SkippedFile | None]:
    if not data:
        return None, SkippedFile(source_name, file_name, "File is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        return None, SkippedFile(
            source_name,
            file_name,
            f"Image file is too large for this MVP (limit {MAX_IMAGE_BYTES // (1024 * 1024)} MB).",
        )

    try:
        image, detected_format, _frame_count, metadata_notes = _decode_image(data)
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        detail = str(exc).strip()
        reason = "Corrupted or unreadable image."
        if detail:
            reason = f"{reason} {detail}"
        return None, SkippedFile(source_name, file_name, reason)

    expected_format = EXTENSION_TO_FORMAT.get(_extension(file_name))
    notes = list(metadata_notes)
    if expected_format and expected_format != detected_format:
        notes.append(
            f"Filename suggests {expected_format}, but the decoded image is {detected_format}."
        )

    quality = assess_image_quality(image)
    fallback_reason: str | None = None
    try:
        attention = provider.analyze(image)
    except Exception as exc:  # A future optional model must not break basic review.
        fallback = DeterministicDemoAttentionProvider()
        attention = fallback.analyze(image)
        notes.append(
            "The configured local model provider could not process this image; the "
            f"deterministic demonstration fallback was used ({type(exc).__name__})."
        )
        fallback_reason = f"attention_provider_error:{type(exc).__name__}"
    prediction = None
    if review_model is not None and quality.adequate:
        if attention.embedding is None:
            notes.append(
                "The experimental priority head could not run because a UNI embedding "
                "was unavailable; the deterministic review-priority fallback was used."
            )
            fallback_reason = fallback_reason or "uni_embedding_unavailable"
        else:
            try:
                prediction = review_model.predict(attention.embedding)
            except Exception as exc:
                notes.append(
                    "The experimental priority head could not process this image; the "
                    f"deterministic review-priority fallback was used ({type(exc).__name__})."
                )
                fallback_reason = f"experimental_head_error:{type(exc).__name__}"
    triage = assign_review_priority(
        quality,
        attention.visual_complexity_score,
        attention.priority_score_source if prediction is None else prediction.source,
        experimental_priority=None if prediction is None else prediction.priority,
        review_first_score=None if prediction is None else prediction.review_first_score,
        fallback_reason=fallback_reason,
    )

    return (
        ImageRecord(
            image_id=_stable_image_id(source_name, file_name, data),
            source_name=source_name,
            display_name=display_name,
            file_name=file_name,
            file_type=detected_format,
            mime_type=FORMAT_TO_MIME[detected_format],
            width=image.width,
            height=image.height,
            size_bytes=len(data),
            image=image,
            quality=quality,
            attention=attention,
            triage=triage,
            metadata_notes=tuple(notes),
        ),
        None,
    )


def _process_zip(
    payload: UploadPayload,
    provider: AttentionProvider,
    decoded_pixel_budget: int,
    review_model: LocalPrototypeReviewHead | None = None,
) -> tuple[list[ImageRecord], list[SkippedFile]]:
    records: list[ImageRecord] = []
    skipped: list[SkippedFile] = []
    used_decoded_pixels = 0

    if not payload.data:
        return records, [SkippedFile(payload.name, payload.name, "ZIP file is empty.")]
    if len(payload.data) > MAX_ARCHIVE_BYTES:
        return records, [
            SkippedFile(
                payload.name,
                payload.name,
                f"ZIP file is too large for this MVP (limit {MAX_ARCHIVE_BYTES // (1024 * 1024)} MB).",
            )
        ]

    try:
        stream = BytesIO(payload.data)
        if not zipfile.is_zipfile(stream):
            raise zipfile.BadZipFile("The file does not contain a readable ZIP archive.")
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            if not members:
                return records, [
                    SkippedFile(payload.name, payload.name, "ZIP contains no files.")
                ]
            if len(members) > MAX_ARCHIVE_MEMBERS:
                return records, [
                    SkippedFile(
                        payload.name,
                        payload.name,
                        f"ZIP contains too many files (limit {MAX_ARCHIVE_MEMBERS}).",
                    )
                ]
            total_size = sum(item.file_size for item in members)
            if total_size > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                return records, [
                    SkippedFile(
                        payload.name,
                        payload.name,
                        "ZIP expands beyond the safe in-memory processing limit.",
                    )
                ]

            for item in members:
                member_name = item.filename
                if not _safe_archive_member(member_name):
                    skipped.append(
                        SkippedFile(
                            payload.name,
                            member_name,
                            "Unsafe archive path was skipped; no files were extracted.",
                        )
                    )
                    continue
                if _metadata_file(member_name):
                    skipped.append(
                        SkippedFile(payload.name, member_name, "System metadata file was ignored.")
                    )
                    continue
                if _extension(member_name) not in SUPPORTED_IMAGE_EXTENSIONS:
                    skipped.append(
                        SkippedFile(payload.name, member_name, _unsupported_reason(member_name))
                    )
                    continue
                if item.flag_bits & 0x1:
                    skipped.append(
                        SkippedFile(payload.name, member_name, "Encrypted ZIP member is unreadable.")
                    )
                    continue
                if item.file_size == 0:
                    skipped.append(SkippedFile(payload.name, member_name, "File is empty."))
                    continue
                if item.file_size > MAX_IMAGE_BYTES:
                    skipped.append(
                        SkippedFile(
                            payload.name,
                            member_name,
                            f"Image file is too large for this MVP (limit {MAX_IMAGE_BYTES // (1024 * 1024)} MB).",
                        )
                    )
                    continue
                compression_ratio = item.file_size / max(item.compress_size, 1)
                if item.file_size > 1024 * 1024 and compression_ratio > MAX_COMPRESSION_RATIO:
                    skipped.append(
                        SkippedFile(
                            payload.name,
                            member_name,
                            "ZIP member has a suspicious compression ratio and was skipped.",
                        )
                    )
                    continue

                try:
                    with archive.open(item, "r") as member_stream:
                        member_data = member_stream.read(MAX_IMAGE_BYTES + 1)
                except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                    skipped.append(
                        SkippedFile(
                            payload.name,
                            member_name,
                            f"ZIP member could not be read ({type(exc).__name__}).",
                        )
                    )
                    continue
                if len(member_data) > MAX_IMAGE_BYTES:
                    skipped.append(
                        SkippedFile(payload.name, member_name, "Image exceeds the safe read limit.")
                    )
                    continue

                record, failure = _process_image_bytes(
                    member_data,
                    member_name,
                    payload.name,
                    f"{payload.name} / {member_name}",
                    provider,
                    review_model,
                )
                if record is not None:
                    decoded_pixels = record.width * record.height
                    if decoded_pixels > decoded_pixel_budget - used_decoded_pixels:
                        skipped.append(
                            SkippedFile(
                                payload.name,
                                member_name,
                                "Batch decoded-image limit reached "
                                f"({MAX_BATCH_DECODED_PIXELS:,} total pixels); this image "
                                "was skipped to keep the app responsive.",
                            )
                        )
                    else:
                        records.append(record)
                        used_decoded_pixels += decoded_pixels
                if failure is not None:
                    skipped.append(failure)
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        detail = str(exc).strip()
        reason = "Corrupted or unreadable ZIP file."
        if detail:
            reason = f"{reason} {detail}"
        skipped.append(SkippedFile(payload.name, payload.name, reason))

    return records, skipped


def process_uploads(
    payloads: list[UploadPayload],
    provider: AttentionProvider | None = None,
    review_model: LocalPrototypeReviewHead | None = None,
) -> BatchResult:
    """Validate and process top-level uploads and safe in-memory ZIP members."""

    active_provider = provider or get_attention_provider()
    records: list[ImageRecord] = []
    skipped: list[SkippedFile] = []
    remaining_decoded_pixels = MAX_BATCH_DECODED_PIXELS

    for payload in payloads:
        extension = _extension(payload.name)
        if extension not in SUPPORTED_UPLOAD_EXTENSIONS:
            skipped.append(
                SkippedFile(payload.name, payload.name, _unsupported_reason(payload.name))
            )
            continue
        if extension == ".zip":
            zip_records, zip_skipped = _process_zip(
                payload,
                active_provider,
                remaining_decoded_pixels,
                review_model,
            )
            records.extend(zip_records)
            skipped.extend(zip_skipped)
            remaining_decoded_pixels -= sum(
                item.width * item.height for item in zip_records
            )
            continue

        record, failure = _process_image_bytes(
            payload.data,
            payload.name,
            payload.name,
            payload.name,
            active_provider,
            review_model,
        )
        if record is not None:
            decoded_pixels = record.width * record.height
            if decoded_pixels > remaining_decoded_pixels:
                skipped.append(
                    SkippedFile(
                        payload.name,
                        payload.name,
                        "Batch decoded-image limit reached "
                        f"({MAX_BATCH_DECODED_PIXELS:,} total pixels); this image was "
                        "skipped to keep the app responsive.",
                    )
                )
            else:
                records.append(record)
                remaining_decoded_pixels -= decoded_pixels
        if failure is not None:
            skipped.append(failure)

    records.sort(
        key=lambda item: (
            priority_sort_key(item.triage.suggested_priority),
            item.display_name.casefold(),
        )
    )
    return BatchResult(
        uploaded_count=len(payloads),
        records=records,
        skipped=skipped,
    )


def format_file_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size_bytes} B"
