"""Optional local UNI feature-visualization provider.

UNI is a pretrained pathology image encoder, not a diagnosis model and not a
review-priority classifier. This adapter uses its patch-token representations
to show relative within-image feature variation. It never downloads weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib.util
from pathlib import Path
from threading import Lock
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageFilter

from .attention import AttentionResult, DeterministicDemoAttentionProvider
from .regions import analyze_variation_map


UNI_PROVIDER_NAME = "Local UNI feature-variation demonstration"
UNI_MODEL_ID = "MahmoodLab/UNI"
UNI_INPUT_SIZE = 224
UNI_EMBEDDING_DIMENSION = 1024
_DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[1] / "models" / "uni" / "pytorch_model.bin"
)
_INFERENCE_LOCK = Lock()


@dataclass(frozen=True)
class UNIProviderStatus:
    ready: bool
    checkpoint_path: Path
    summary: str
    detail: str
    cache_key: str


def get_uni_provider_status(checkpoint_path: str | Path | None = None) -> UNIProviderStatus:
    """Inspect local readiness without loading the 1.2 GB checkpoint."""

    path = Path(checkpoint_path or _DEFAULT_CHECKPOINT).resolve()
    missing_packages = [
        package
        for package in ("torch", "torchvision", "timm")
        if importlib.util.find_spec(package) is None
    ]
    if not path.is_file():
        return UNIProviderStatus(
            ready=False,
            checkpoint_path=path,
            summary="UNI checkpoint not found; deterministic fallback available",
            detail=(
                "Place the approved pytorch_model.bin file in models/uni/. "
                "The app will not download it automatically."
            ),
            cache_key="uni:missing-checkpoint",
        )
    stat = path.stat()
    artifact_key = f"{stat.st_size}:{stat.st_mtime_ns}"
    if missing_packages:
        shown = ", ".join(missing_packages)
        return UNIProviderStatus(
            ready=False,
            checkpoint_path=path,
            summary="UNI checkpoint found; optional inference packages are missing",
            detail=f"Install the optional UNI requirements ({shown}).",
            cache_key=f"uni:missing-packages:{shown}:{artifact_key}",
        )
    return UNIProviderStatus(
        ready=True,
        checkpoint_path=path,
        summary="Local UNI encoder is ready",
        detail=(
            "UNI can be enabled for an exploratory feature-variation overlay. "
            "No trained review-priority classifier is loaded."
        ),
        cache_key=f"uni:ready:{artifact_key}",
    )


@lru_cache(maxsize=2)
def _load_uni_model(
    checkpoint_path: str,
    checkpoint_size: int,
    checkpoint_mtime_ns: int,
) -> tuple[Any, str]:
    """Load the official ViT-L/16 architecture strictly from local weights."""

    del checkpoint_size, checkpoint_mtime_ns  # They intentionally invalidate the cache.
    import torch
    import timm

    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=UNI_INPUT_SIZE,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, device


def _letterbox(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Fit an image into a square without changing its aspect ratio."""

    resized = image.convert("RGB").copy()
    resized.thumbnail((UNI_INPUT_SIZE, UNI_INPUT_SIZE), Image.Resampling.LANCZOS)
    left = (UNI_INPUT_SIZE - resized.width) // 2
    top = (UNI_INPUT_SIZE - resized.height) // 2
    square = Image.new("RGB", (UNI_INPUT_SIZE, UNI_INPUT_SIZE), (255, 255, 255))
    square.paste(resized, (left, top))
    return square, (left, top, left + resized.width, top + resized.height)


def prepare_uni_tensor(image: Image.Image) -> Any:
    """Return one normalized CHW tensor using the app's UNI preprocessing."""

    import torch

    square, _ = _letterbox(image)
    values = np.asarray(square, dtype=np.float32) / 255.0
    values = (values - np.array((0.485, 0.456, 0.406), dtype=np.float32)) / np.array(
        (0.229, 0.224, 0.225), dtype=np.float32
    )
    return torch.from_numpy(values.transpose(2, 0, 1))


def _normalized(values: np.ndarray) -> np.ndarray:
    low = float(np.percentile(values, 5.0))
    high = float(np.percentile(values, 95.0))
    if high - low <= 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


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


def _render_feature_overlay(
    image: Image.Image,
    patch_scores: np.ndarray,
    content_box: tuple[int, int, int, int],
) -> tuple[Image.Image, Image.Image, str, np.ndarray]:
    preview = image.convert("RGB").copy()
    if max(preview.size) > 1200:
        preview.thumbnail((1200, 1200), Image.Resampling.LANCZOS)

    score_image = Image.fromarray(
        np.uint8(np.clip(_normalized(patch_scores) * 255.0, 0, 255)), mode="L"
    ).resize((UNI_INPUT_SIZE, UNI_INPUT_SIZE), Image.Resampling.BILINEAR)
    score_image = score_image.crop(content_box).resize(
        preview.size, Image.Resampling.BILINEAR
    )
    score_image = score_image.filter(
        ImageFilter.GaussianBlur(radius=max(2.0, min(preview.size) / 80.0))
    )
    salience = _normalized(np.asarray(score_image, dtype=np.float32))

    heat_rgb = np.stack(
        [
            np.clip(3.0 * salience, 0.0, 1.0),
            np.clip((3.0 * salience) - 1.0, 0.0, 1.0),
            np.clip((3.0 * salience) - 2.0, 0.0, 1.0),
        ],
        axis=2,
    )
    rgb = np.asarray(preview, dtype=np.float32) / 255.0
    alpha = (0.58 * np.power(salience, 0.75))[:, :, None]
    overlay_rgb = (rgb * (1.0 - alpha)) + (heat_rgb * alpha)

    if float(np.max(salience)) <= 1e-8:
        region = "No single region dominates this feature-variation map"
    else:
        cutoff = float(np.percentile(salience, 90.0))
        weights = np.where(salience >= cutoff, salience, 0.0)
        yy, xx = np.indices(salience.shape, dtype=np.float32)
        total = max(float(np.sum(weights)), 1e-8)
        x_center = float(np.sum(xx * weights) / total) / max(salience.shape[1] - 1, 1)
        y_center = float(np.sum(yy * weights) / total) / max(salience.shape[0] - 1, 1)
        region = f"The {_position_name(x_center, y_center)} area is highlighted most strongly"

    heatmap = Image.fromarray(
        np.uint8(np.clip(heat_rgb * 255.0, 0, 255)), mode="RGB"
    )
    overlay = Image.fromarray(
        np.uint8(np.clip(overlay_rgb * 255.0, 0, 255)), mode="RGB"
    )
    return overlay, heatmap, region, salience


class LocalUNIFeatureProvider:
    """Create an exploratory map from local UNI patch-token variation."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_loader: Callable[[str, int, int], tuple[Any, str]] = _load_uni_model,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self._model_loader = model_loader

    def analyze(self, image: Image.Image) -> AttentionResult:
        import torch

        stat = self.checkpoint_path.stat()
        model, device = self._model_loader(
            str(self.checkpoint_path), stat.st_size, stat.st_mtime_ns
        )
        square, content_box = _letterbox(image)
        tensor = prepare_uni_tensor(image).unsqueeze(0).to(device)

        with _INFERENCE_LOCK, torch.inference_mode():
            tokens = model.forward_features(tensor)
            embedding_tensor = model.forward_head(tokens, pre_logits=True).float().cpu()
        if tokens.ndim != 3 or tokens.shape[0] != 1:
            raise RuntimeError("UNI returned an unexpected token tensor.")
        prefix_tokens = int(getattr(model, "num_prefix_tokens", 1))
        patches = tokens[0, prefix_tokens:, :]
        patch_count = int(patches.shape[0])
        grid_side = int(round(patch_count ** 0.5))
        if grid_side * grid_side != patch_count or int(patches.shape[1]) != UNI_EMBEDDING_DIMENSION:
            raise RuntimeError("UNI returned an unexpected patch-token layout.")

        normalized = torch.nn.functional.normalize(patches.float(), dim=1)
        centroid = torch.nn.functional.normalize(normalized.mean(dim=0, keepdim=True), dim=1)
        scores = 1.0 - torch.sum(normalized * centroid, dim=1)
        patch_scores = scores.reshape(grid_side, grid_side).cpu().numpy()
        if tuple(embedding_tensor.shape) != (1, UNI_EMBEDDING_DIMENSION):
            raise RuntimeError("UNI returned an unexpected global embedding shape.")
        if not bool(torch.isfinite(embedding_tensor).all()):
            raise RuntimeError("UNI returned a non-finite global embedding.")
        embedding = tuple(float(value) for value in embedding_tensor[0].tolist())
        overlay, heatmap, region, variation_map = _render_feature_overlay(
            image, patch_scores, content_box
        )
        region_analysis = analyze_variation_map(
            variation_map,
            source="uni_feature_variation",
        )

        # Review priority deliberately stays on the existing deterministic rule.
        demo_result = DeterministicDemoAttentionProvider().analyze(image)
        explanation = (
            f"{region} because its representation differs comparatively more from other "
            "regions in this resized image under the local UNI encoder. This is an "
            "exploratory feature-variation visualization, not a validated clinical "
            "attention map, pathology finding, or medical conclusion. UNI alone does "
            "not assign review priority."
        )
        return AttentionResult(
            overlay=overlay,
            heatmap=heatmap,
            explanation=explanation,
            visual_complexity_score=demo_result.visual_complexity_score,
            provider_name=f"{UNI_PROVIDER_NAME} ({device.upper()})",
            is_demonstration=True,
            uses_trained_encoder=True,
            priority_score_source="Deterministic visual-complexity heuristic (not UNI)",
            overlay_caption=(
                "Exploratory local UNI feature-variation overlay; not a diagnostic "
                "explanation."
            ),
            embedding=embedding,
            embedding_model=UNI_MODEL_ID,
            variation_map=variation_map,
            image_priority_score=region_analysis.image_priority_score,
        )
