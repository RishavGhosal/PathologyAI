"""Deterministic, non-diagnostic image-quality checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


MIN_DIMENSION = 128
BLUR_SCORE_THRESHOLD = 18.0
DARKNESS_THRESHOLD = 28.0
BRIGHTNESS_THRESHOLD = 232.0
UNIFORMITY_THRESHOLD = 6.0

# Stable machine-readable codes used by dashboards and reviewer exports. Keep
# these values independent from human-readable wording so copy changes do not
# break aggregation.
ISSUE_SMALL_DIMENSIONS = "small_dimensions"
ISSUE_EXCESSIVE_DARKNESS = "excessive_darkness"
ISSUE_EXCESSIVE_BRIGHTNESS = "excessive_brightness"
ISSUE_BLANK_OR_NEARLY_UNIFORM = "blank_or_nearly_uniform"
ISSUE_BLUR = "blur"
ADVISORY_POSSIBLE_EDGE_TRUNCATION = "possible_edge_truncation"


@dataclass(frozen=True)
class QualityAssessment:
    """The result of presentation-quality checks on a decoded image."""

    adequate: bool
    reasons: tuple[str, ...]
    metrics: dict[str, float]
    advisories: tuple[str, ...] = ()
    issue_codes: tuple[str, ...] = ()
    advisory_codes: tuple[str, ...] = ()


def _analysis_array(image: Image.Image, max_side: int = 768) -> np.ndarray:
    sample = image.convert("RGB").copy()
    if max(sample.size) > max_side:
        sample.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return np.asarray(sample, dtype=np.float32)


def _laplacian_variance(gray: np.ndarray) -> float:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    center = gray[1:-1, 1:-1]
    laplacian = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - (4.0 * center)
    )
    return float(np.var(laplacian))


def _possible_edge_truncation(rgb: np.ndarray) -> tuple[bool, int]:
    """Conservatively flag foreground that meets several frame edges.

    This is intentionally limited to images that contain a visible background;
    full-field tissue patches are not automatically treated as cropped.
    """

    height, width, _ = rgb.shape
    if min(height, width) < MIN_DIMENSION:
        return False, 0

    corner = max(3, min(height, width) // 24)
    corner_pixels = np.concatenate(
        [
            rgb[:corner, :corner].reshape(-1, 3),
            rgb[:corner, -corner:].reshape(-1, 3),
            rgb[-corner:, :corner].reshape(-1, 3),
            rgb[-corner:, -corner:].reshape(-1, 3),
        ],
        axis=0,
    )
    background = np.median(corner_pixels, axis=0)
    distance = np.linalg.norm(rgb - background, axis=2)
    foreground = distance > 32.0
    foreground_fraction = float(np.mean(foreground))

    # A mostly full frame is commonly intentional for microscopy patches.
    if not 0.06 <= foreground_fraction <= 0.82:
        return False, 0

    band = max(2, min(height, width) // 40)
    contacts = (
        float(np.mean(foreground[:band, :])),
        float(np.mean(foreground[-band:, :])),
        float(np.mean(foreground[:, :band])),
        float(np.mean(foreground[:, -band:])),
    )
    contact_count = sum(value >= 0.32 for value in contacts)
    return contact_count >= 2, contact_count


def assess_image_quality(image: Image.Image) -> QualityAssessment:
    """Check whether an image is usable for this review-priority prototype."""

    width, height = image.size
    rgb = _analysis_array(image)
    gray = (
        (0.299 * rgb[:, :, 0])
        + (0.587 * rgb[:, :, 1])
        + (0.114 * rgb[:, :, 2])
    )
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur_score = _laplacian_variance(gray)
    possible_crop, edge_contacts = _possible_edge_truncation(rgb)

    reasons: list[str] = []
    issue_codes: list[str] = []
    advisories: list[str] = []
    advisory_codes: list[str] = []
    if min(width, height) < MIN_DIMENSION:
        reasons.append(
            f"Very small dimensions ({width} × {height} px); use an image at least "
            f"{MIN_DIMENSION} px on each side."
        )
        issue_codes.append(ISSUE_SMALL_DIMENSIONS)
    if brightness <= DARKNESS_THRESHOLD:
        reasons.append("Image is excessively dark, so important visual detail may be hidden.")
        issue_codes.append(ISSUE_EXCESSIVE_DARKNESS)
    elif brightness >= BRIGHTNESS_THRESHOLD:
        reasons.append("Image is excessively bright, so important visual detail may be washed out.")
        issue_codes.append(ISSUE_EXCESSIVE_BRIGHTNESS)

    nearly_uniform = contrast <= UNIFORMITY_THRESHOLD
    if nearly_uniform:
        reasons.append("Image is blank or nearly uniform (very little tonal variation).")
        issue_codes.append(ISSUE_BLANK_OR_NEARLY_UNIFORM)
    elif blur_score < BLUR_SCORE_THRESHOLD:
        reasons.append(
            "Image appears blurred or out of focus based on its low edge sharpness."
        )
        issue_codes.append(ISSUE_BLUR)

    if possible_crop:
        advisories.append(
            "Possible edge truncation: image content touches multiple frame edges while "
            "background remains visible. This can be normal for microscopy fields; "
            "manual verification is recommended."
        )
        advisory_codes.append(ADVISORY_POSSIBLE_EDGE_TRUNCATION)

    return QualityAssessment(
        adequate=not reasons,
        reasons=tuple(reasons),
        metrics={
            "brightness": brightness,
            "contrast": contrast,
            "blur_score": blur_score,
            "edge_contacts": float(edge_contacts),
        },
        advisories=tuple(advisories),
        issue_codes=tuple(issue_codes),
        advisory_codes=tuple(advisory_codes),
    )
