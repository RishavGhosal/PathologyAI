"""Explainable visual-attention providers for PathologyAI.

The deterministic provider always remains available. A separately installed
local UNI adapter can provide an exploratory feature-variation visualization,
but UNI is not a review-priority classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image, ImageFilter


DEMO_PROVIDER_NAME = "Deterministic demonstration attention"


@dataclass(frozen=True)
class AttentionResult:
    overlay: Image.Image
    heatmap: Image.Image
    explanation: str
    visual_complexity_score: float
    provider_name: str
    is_demonstration: bool
    uses_trained_encoder: bool = False
    priority_score_source: str = "Deterministic visual-complexity heuristic"
    overlay_caption: str = "Deterministic visual-salience demonstration overlay."
    embedding: tuple[float, ...] | None = None
    embedding_model: str | None = None


class AttentionProvider(Protocol):
    """Interface for a future local trained-model attention adapter."""

    def analyze(self, image: Image.Image) -> AttentionResult:
        """Return an overlay, explanation, and review-order feature score."""


def _normalized(values: np.ndarray) -> np.ndarray:
    upper = float(np.percentile(values, 97.0))
    if upper <= 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(values / upper, 0.0, 1.0).astype(np.float32)


def _position_name(x: float, y: float) -> str:
    horizontal = "left" if x < 0.34 else "right" if x > 0.66 else "center"
    vertical = "upper" if y < 0.34 else "lower" if y > 0.66 else "middle"
    if horizontal == "center" and vertical == "middle":
        return "central"
    if horizontal == "center":
        return f"{vertical}-center"
    if vertical == "middle":
        return f"middle-{horizontal}"
    return f"{vertical}-{horizontal}"


class DeterministicDemoAttentionProvider:
    """Create a repeatable heatmap from contrast, color, and edges only."""

    max_preview_side = 1200

    def analyze(self, image: Image.Image) -> AttentionResult:
        preview = image.convert("RGB").copy()
        if max(preview.size) > self.max_preview_side:
            preview.thumbnail(
                (self.max_preview_side, self.max_preview_side),
                Image.Resampling.LANCZOS,
            )

        rgb = np.asarray(preview, dtype=np.float32) / 255.0
        gray = (
            (0.299 * rgb[:, :, 0])
            + (0.587 * rgb[:, :, 1])
            + (0.114 * rgb[:, :, 2])
        )
        gradient_y, gradient_x = np.gradient(gray)
        gradient = np.hypot(gradient_x, gradient_y)

        blur_radius = max(2.0, min(preview.size) / 80.0)
        gray_image = Image.fromarray(np.uint8(np.clip(gray * 255.0, 0, 255)), mode="L")
        local_mean = np.asarray(
            gray_image.filter(ImageFilter.GaussianBlur(radius=blur_radius)),
            dtype=np.float32,
        ) / 255.0
        local_contrast = np.abs(gray - local_mean)
        saturation = np.max(rgb, axis=2) - np.min(rgb, axis=2)

        salience = (
            (0.52 * _normalized(gradient))
            + (0.33 * _normalized(local_contrast))
            + (0.15 * _normalized(saturation))
        )
        salience_image = Image.fromarray(
            np.uint8(np.clip(salience * 255.0, 0, 255)), mode="L"
        ).filter(ImageFilter.GaussianBlur(radius=max(1.0, min(preview.size) / 300.0)))
        salience = np.asarray(salience_image, dtype=np.float32) / 255.0
        salience = _normalized(salience)

        # A compact black-red-yellow-white map without matplotlib.
        heat_rgb = np.stack(
            [
                np.clip(3.0 * salience, 0.0, 1.0),
                np.clip((3.0 * salience) - 1.0, 0.0, 1.0),
                np.clip((3.0 * salience) - 2.0, 0.0, 1.0),
            ],
            axis=2,
        )
        alpha = (0.62 * np.power(salience, 0.72))[:, :, None]
        overlay_rgb = (rgb * (1.0 - alpha)) + (heat_rgb * alpha)

        heatmap = Image.fromarray(np.uint8(np.clip(heat_rgb * 255.0, 0, 255)), mode="RGB")
        overlay = Image.fromarray(
            np.uint8(np.clip(overlay_rgb * 255.0, 0, 255)), mode="RGB"
        )

        edge_density = float(np.mean(gradient > 0.035))
        mean_gradient = float(np.mean(gradient))
        global_contrast = float(np.std(gray))
        complexity = (
            (0.50 * min(edge_density / 0.20, 1.0))
            + (0.30 * min(mean_gradient / 0.08, 1.0))
            + (0.20 * min(global_contrast / 0.24, 1.0))
        )

        if float(np.max(salience)) <= 1e-8:
            region_text = "No single region dominates the visual-salience map"
        else:
            cutoff = float(np.percentile(salience, 90.0))
            weights = np.where(salience >= cutoff, salience, 0.0)
            if float(np.sum(weights)) <= 1e-8:
                weights = salience
            yy, xx = np.indices(salience.shape, dtype=np.float32)
            x_center = float(np.sum(xx * weights) / np.sum(weights)) / max(
                salience.shape[1] - 1, 1
            )
            y_center = float(np.sum(yy * weights) / np.sum(weights)) / max(
                salience.shape[0] - 1, 1
            )
            region_text = (
                f"The demonstration highlights the {_position_name(x_center, y_center)} "
                "area most strongly because it contains comparatively stronger edges, "
                "color variation, or local contrast"
            )

        explanation = (
            f"{region_text}. This visual-salience cue is deterministic and is not a "
            "learned pathology finding or medical conclusion."
        )
        return AttentionResult(
            overlay=overlay,
            heatmap=heatmap,
            explanation=explanation,
            visual_complexity_score=float(np.clip(complexity, 0.0, 1.0)),
            provider_name=DEMO_PROVIDER_NAME,
            is_demonstration=True,
            uses_trained_encoder=False,
            priority_score_source="Deterministic visual-complexity heuristic",
            overlay_caption=(
                "Deterministic demonstration overlay based on edges, contrast, and "
                "color variation."
            ),
        )


def get_attention_provider(prefer_uni: bool = False) -> AttentionProvider:
    """Return a local provider without downloading weights.

    UNI is opt-in because its ViT-L encoder is large and can be slow on CPU. If
    the checkpoint or optional packages are unavailable, the deterministic
    provider keeps the basic application functional.
    """

    if prefer_uni:
        try:
            from .uni_provider import LocalUNIFeatureProvider, get_uni_provider_status

            status = get_uni_provider_status()
            if status.ready:
                return LocalUNIFeatureProvider(status.checkpoint_path)
        except Exception:
            pass
    return DeterministicDemoAttentionProvider()
