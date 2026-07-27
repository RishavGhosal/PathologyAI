"""Optional local Hibou-B feature provider for broad pathology exploration.

Hibou-B is a pretrained pathology feature encoder, not a diagnosis or a
review-priority classifier.  The application loads a previously downloaded
local snapshot only; it never fetches weights or remote code while serving a
user upload.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib.util
from pathlib import Path
from threading import Lock
from typing import Any, Callable

import numpy as np
from PIL import Image

from .attention import AttentionResult, DeterministicDemoAttentionProvider
from .regions import analyze_variation_map
from .uni_provider import _letterbox, _render_feature_overlay


HIBOU_PROVIDER_NAME = "Local Hibou-B feature-variation exploration"
HIBOU_MODEL_ID = "histai/hibou-b"
HIBOU_INPUT_SIZE = 224
HIBOU_EMBEDDING_DIMENSION = 768
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "hibou-b"
_INFERENCE_LOCK = Lock()


@dataclass(frozen=True)
class HibouProviderStatus:
    ready: bool
    model_dir: Path
    summary: str
    detail: str
    cache_key: str


def get_hibou_provider_status(model_dir: str | Path | None = None) -> HibouProviderStatus:
    """Inspect local Hibou-B readiness without network access or model loading."""

    directory = Path(model_dir or _DEFAULT_MODEL_DIR).resolve()
    required = (directory / "config.json",)
    missing = [path.name for path in required if not path.is_file()]
    has_weights = any(directory.glob("*.safetensors")) or any(directory.glob("*.bin"))
    if missing or not has_weights:
        return HibouProviderStatus(
            ready=False,
            model_dir=directory,
            summary="Hibou-B local snapshot not found; deterministic fallback available",
            detail=(
                "After accepting the model terms, download the complete histai/hibou-b "
                "snapshot into models/hibou-b/. The app will not download it at runtime."
            ),
            cache_key="hibou:missing-local-snapshot",
        )
    missing_packages = [
        package
        for package in ("torch", "transformers")
        if importlib.util.find_spec(package) is None
    ]
    artifact_key = ":".join(
        f"{path.name}:{path.stat().st_size}:{path.stat().st_mtime_ns}"
        for path in sorted(directory.glob("*"))
        if path.is_file()
    )
    if missing_packages:
        shown = ", ".join(missing_packages)
        return HibouProviderStatus(
            ready=False,
            model_dir=directory,
            summary="Hibou-B snapshot found; optional packages are missing",
            detail=f"Install requirements-hibou.txt ({shown} missing).",
            cache_key=f"hibou:missing-packages:{shown}:{artifact_key}",
        )
    return HibouProviderStatus(
        ready=True,
        model_dir=directory,
        summary="Local Hibou-B encoder is ready",
        detail=(
            "CPU-only Hibou-B feature exploration is available. It does not assign "
            "tissue, disease, diagnosis, or review priority."
        ),
        cache_key=f"hibou:ready:{artifact_key}",
    )


@lru_cache(maxsize=1)
def _load_hibou_model(
    model_dir: str,
    artifact_key: str,
) -> tuple[Any, Any]:
    """Load only the audited, locally stored model snapshot on CPU."""

    del artifact_key
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=False,
    )
    model = AutoModel.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=True
    )
    model.to("cpu")
    model.eval()
    return model, processor


class LocalHibouFeatureProvider:
    """Create an exploratory local-feature variation map from Hibou-B tokens."""

    def __init__(
        self,
        model_dir: str | Path,
        model_loader: Callable[[str, str], tuple[Any, Any]] = _load_hibou_model,
    ) -> None:
        self.model_dir = Path(model_dir).resolve()
        self._model_loader = model_loader

    def analyze(self, image: Image.Image) -> AttentionResult:
        import torch

        status = get_hibou_provider_status(self.model_dir)
        if not status.ready:
            raise RuntimeError(status.summary)
        model, processor = self._model_loader(str(self.model_dir), status.cache_key)
        inputs = processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {name: value.to("cpu") for name, value in inputs.items()}
        with _INFERENCE_LOCK, torch.inference_mode():
            outputs = model(**inputs)
        tokens = getattr(outputs, "last_hidden_state", None)
        if tokens is None and isinstance(outputs, tuple) and outputs:
            tokens = outputs[0]
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3 or tokens.shape[0] != 1:
            raise RuntimeError("Hibou-B returned an unexpected token tensor.")
        embedding_tensor = tokens[:, 0, :].float().cpu()
        if tuple(embedding_tensor.shape) != (1, HIBOU_EMBEDDING_DIMENSION):
            raise RuntimeError("Hibou-B returned an unexpected global embedding shape.")
        if not bool(torch.isfinite(embedding_tensor).all()):
            raise RuntimeError("Hibou-B returned a non-finite global embedding.")

        prefix_tokens = 1 + int(getattr(model.config, "num_register_tokens", 0))
        patches = tokens[0, prefix_tokens:, :].float()
        patch_count = int(patches.shape[0])
        grid_side = int(round(patch_count**0.5))
        if grid_side * grid_side != patch_count:
            raise RuntimeError("Hibou-B returned an unexpected patch-token layout.")
        normalized = torch.nn.functional.normalize(patches, dim=1)
        centroid = torch.nn.functional.normalize(normalized.mean(dim=0, keepdim=True), dim=1)
        scores = 1.0 - torch.sum(normalized * centroid, dim=1)
        patch_scores = scores.reshape(grid_side, grid_side).cpu().numpy()
        _square, content_box = _letterbox(image)
        overlay, heatmap, region, variation_map = _render_feature_overlay(image, patch_scores, content_box)
        region_analysis = analyze_variation_map(
            variation_map,
            source="hibou_b_feature_variation",
        )

        demo_result = DeterministicDemoAttentionProvider().analyze(image)
        embedding = tuple(float(value) for value in embedding_tensor[0].tolist())
        return AttentionResult(
            overlay=overlay,
            heatmap=heatmap,
            explanation=(
                f"{region} because its Hibou-B patch representation differs comparatively "
                "more from other regions in this image. This is exploratory feature "
                "variation, not a diagnostic attention map or pathology conclusion."
            ),
            visual_complexity_score=demo_result.visual_complexity_score,
            provider_name=f"{HIBOU_PROVIDER_NAME} (CPU)",
            is_demonstration=True,
            uses_trained_encoder=True,
            priority_score_source="Deterministic visual-complexity heuristic (not Hibou-B)",
            overlay_caption="Exploratory local Hibou-B feature-variation overlay; not diagnostic.",
            embedding=embedding,
            embedding_model=HIBOU_MODEL_ID,
            variation_map=variation_map,
            image_priority_score=region_analysis.image_priority_score,
        )


__all__ = [
    "HIBOU_EMBEDDING_DIMENSION",
    "HIBOU_MODEL_ID",
    "HibouProviderStatus",
    "LocalHibouFeatureProvider",
    "get_hibou_provider_status",
]
