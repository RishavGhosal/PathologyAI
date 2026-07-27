"""Deterministic feature-variation regions and image-level routing signals.

This module deliberately knows nothing about tissue or diagnosis.  It turns a
provider's numeric feature-variation map into contiguous, normalized regions
that can be drawn by the UI and passed as structured context to a captioning
model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class FeatureRegion:
    """One contiguous high-variation region in normalized image coordinates."""

    region_id: int
    x: float
    y: float
    width: float
    height: float
    peak_intensity: float
    mean_intensity: float
    area_fraction: float
    contribution_percentage: float
    location: str

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegionAnalysis:
    """Computed signals shared by the API, queue, and captioning path."""

    regions: tuple[FeatureRegion, ...]
    image_priority_score: float
    summary: str
    source: str

    def as_json(self) -> dict[str, Any]:
        return {
            "regions": [region.as_json() for region in self.regions],
            "priority_score": self.image_priority_score,
            "summary": self.summary,
            "source": self.source,
        }


def position_name(x: float, y: float) -> str:
    horizontal = "left" if x < 0.34 else "right" if x > 0.66 else "center"
    vertical = "upper" if y < 0.34 else "lower" if y > 0.66 else "middle"
    if horizontal == "center" and vertical == "middle":
        return "central"
    if horizontal == "center":
        return f"{vertical}-center"
    if vertical == "middle":
        return f"middle-{horizontal}"
    return f"{vertical}-{horizontal}"


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 8-connected pixel coordinates for each true component."""

    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for start_y, start_x in zip(*np.where(mask)):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        pixels: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for next_y in range(max(0, y - 1), min(height, y + 2)):
                for next_x in range(max(0, x - 1), min(width, x + 2)):
                    if not mask[next_y, next_x] or visited[next_y, next_x]:
                        continue
                    visited[next_y, next_x] = True
                    stack.append((next_y, next_x))
        components.append(np.asarray(pixels, dtype=np.int32))
    return components


def _fallback_component(values: np.ndarray) -> np.ndarray:
    y, x = np.unravel_index(int(np.argmax(values)), values.shape)
    return np.asarray([[y, x]], dtype=np.int32)


def analyze_variation_map(
    variation_map: np.ndarray,
    *,
    source: str,
    max_regions: int = 3,
) -> RegionAnalysis:
    """Find the top one to three contiguous high-intensity regions.

    The threshold is percentile-based, so it is stable across encoders with
    different raw score ranges.  Coordinates are normalized to the source
    image and therefore remain usable for responsive overlay rendering.
    """

    values = np.asarray(variation_map, dtype=np.float32)
    if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
        values = np.zeros((8, 8), dtype=np.float32)
    values = np.clip(values, 0.0, None)
    peak = float(values.max())
    if peak <= 1e-8:
        return RegionAnalysis((), 0.0, "No high-variation region was detected", source)
    normalized = np.clip(values / peak, 0.0, 1.0)
    threshold = max(float(np.percentile(normalized, 90.0)), 0.45)
    mask = normalized >= threshold
    components = [component for component in _connected_components(mask) if len(component) >= max(2, values.size // 500)]
    if not components:
        components = [_fallback_component(normalized)]

    scored: list[tuple[float, np.ndarray]] = []
    for component in components:
        component_values = normalized[component[:, 0], component[:, 1]]
        score = float(np.mean(component_values) * (0.7 + 0.3 * min(len(component) / max(values.size * 0.08, 1.0), 1.0)))
        scored.append((score, component))
    scored.sort(key=lambda item: (item[0], float(np.max(normalized[item[1][:, 0], item[1][:, 1]]))), reverse=True)
    selected = scored[:max_regions]

    total_mass = max(float(np.sum(normalized)), 1e-8)
    regions: list[FeatureRegion] = []
    for region_id, (_score, component) in enumerate(selected, start=1):
        ys, xs = component[:, 0], component[:, 1]
        min_x, max_x = int(xs.min()), int(xs.max())
        min_y, max_y = int(ys.min()), int(ys.max())
        component_values = normalized[ys, xs]
        mass = float(np.sum(component_values))
        center_x = float(np.mean(xs)) / max(values.shape[1] - 1, 1)
        center_y = float(np.mean(ys)) / max(values.shape[0] - 1, 1)
        regions.append(
            FeatureRegion(
                region_id=region_id,
                x=min_x / values.shape[1],
                y=min_y / values.shape[0],
                width=(max_x + 1 - min_x) / values.shape[1],
                height=(max_y + 1 - min_y) / values.shape[0],
                peak_intensity=float(np.max(component_values)),
                mean_intensity=float(np.mean(component_values)),
                area_fraction=float(len(component) / values.size),
                contribution_percentage=float(100.0 * mass / total_mass),
                location=position_name(center_x, center_y),
            )
        )

    # A raw within-image peak is always one after encoder normalization.  Use
    # its contribution to the image-wide variation mass to make the score
    # comparable across images while retaining the peak-intensity signal.
    image_priority_score = float(
        max(
            region.peak_intensity * min(region.contribution_percentage / 20.0, 1.0)
            for region in regions
        )
    )
    highest = regions[0]
    summary = (
        f"Feature variation concentrated in {highest.location}, "
        f"high intensity ({highest.peak_intensity:.2f})"
    )
    return RegionAnalysis(tuple(regions), image_priority_score, summary, source)


def variation_map_from_heatmap(heatmap: Image.Image) -> np.ndarray:
    """Recover a monotonic intensity proxy for legacy providers without maps."""

    rgb = np.asarray(heatmap.convert("RGB"), dtype=np.float32) / 255.0
    # The application colormap moves black -> red -> yellow -> white.  This
    # luminance proxy is monotonic enough for region geometry and fallback use.
    return np.max(rgb, axis=2).astype(np.float32)


def resize_map(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.uint8(np.clip(values, 0.0, 1.0) * 255.0), mode="L")
    resized = image.resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def rank_agreement(left: np.ndarray, right: np.ndarray) -> float | None:
    """Return a [0, 1] rank-agreement score for two shared-grid maps."""

    if left.size < 2 or right.size < 2:
        return None
    left_values = np.asarray(left, dtype=np.float64).ravel()
    right_values = np.asarray(right, dtype=np.float64).ravel()
    if not np.isfinite(left_values).all() or not np.isfinite(right_values).all():
        return None
    left_ranks = np.argsort(np.argsort(left_values)).astype(np.float64)
    right_ranks = np.argsort(np.argsort(right_values)).astype(np.float64)
    left_ranks -= left_ranks.mean()
    right_ranks -= right_ranks.mean()
    denominator = float(np.linalg.norm(left_ranks) * np.linalg.norm(right_ranks))
    if denominator <= 1e-8:
        return None
    correlation = float(np.dot(left_ranks, right_ranks) / denominator)
    return float(np.clip((correlation + 1.0) / 2.0, 0.0, 1.0))
