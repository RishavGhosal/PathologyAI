"""Remote UNI/Hibou providers backed by a Modal HTTP endpoint."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import os
from io import BytesIO
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image

from .attention import AttentionResult
from .review_model import ReviewModelPrediction


MODAL_UNI_PROVIDER_NAME = "Modal UNI feature exploration"
MODAL_HIBOU_PROVIDER_NAME = "Modal Hibou-B feature exploration"
_DEFAULT_TIMEOUT_SECONDS = 300


@dataclass(frozen=True)
class ModalProviderStatus:
    ready: bool
    summary: str
    detail: str


def get_modal_provider_status(
    provider_kind: str = "hibou",
    environ: dict[str, str] | None = None,
) -> ModalProviderStatus:
    values = os.environ if environ is None else environ
    url = values.get("PATHOLOGYAI_MODAL_URL", "").strip()
    label = "UNI" if provider_kind == "uni" else "Hibou-B"
    if not url:
        return ModalProviderStatus(
            ready=False,
            summary=f"Modal {label} API is not configured",
            detail="Set PATHOLOGYAI_MODAL_URL to the deployed Modal endpoint to enable remote GPU inference.",
        )
    if not url.startswith(("https://", "http://")):
        return ModalProviderStatus(
            ready=False,
            summary="Modal URL is invalid",
            detail="PATHOLOGYAI_MODAL_URL must be an http:// or https:// URL.",
        )
    return ModalProviderStatus(
        ready=True,
        summary=f"Modal {label} API is configured",
        detail=f"Images are sent to the configured Modal GPU endpoint for remote {label} feature extraction.",
    )


def _decode_png(value: str, label: str) -> Image.Image:
    try:
        raw = base64.b64decode(value, validate=True)
        return Image.open(BytesIO(raw)).convert("RGB")
    except (ValueError, TypeError, base64.binascii.Error) as exc:
        raise RuntimeError(f"Modal returned an invalid {label} image.") from exc


class ModalFeatureProvider:
    """Call Modal for either UNI or Hibou-B and adapt the response locally."""

    def __init__(
        self,
        provider_kind: str,
        url: str | None = None,
        use_review_model: bool = False,
        timeout: int = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if provider_kind not in {"uni", "hibou"}:
            raise ValueError(f"Unsupported Modal provider: {provider_kind}")
        self.provider_kind = provider_kind
        self.url = (url or os.environ.get("PATHOLOGYAI_MODAL_URL", "")).strip()
        self.use_review_model = use_review_model and provider_kind == "uni"
        self.timeout = timeout
        self.last_review_prediction: ReviewModelPrediction | None = None

    def analyze(self, image: Image.Image) -> AttentionResult:
        if not self.url:
            raise RuntimeError("PATHOLOGYAI_MODAL_URL is not configured.")
        self.last_review_prediction = None
        output = BytesIO()
        image.convert("RGB").save(output, format="JPEG", quality=92, optimize=True)
        payload = json.dumps(
            {
                "provider_kind": self.provider_kind,
                "use_review_model": self.use_review_model,
                "image_base64": base64.b64encode(output.getvalue()).decode("ascii"),
            }
        ).encode()
        headers = {"Content-Type": "application/json"}
        modal_key, modal_secret = os.environ.get("MODAL_KEY"), os.environ.get("MODAL_SECRET")
        if modal_key and modal_secret:
            headers.update({"Modal-Key": modal_key, "Modal-Secret": modal_secret})
        request = Request(self.url, data=payload, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result: Any = json.loads(response.read())
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Modal inference request failed: {type(exc).__name__}.") from exc
        if not isinstance(result, dict):
            raise RuntimeError("Modal returned an invalid response object.")

        review_priority = result.get("review_priority")
        review_score = result.get("review_first_score")
        if review_priority is not None and review_score is not None:
            self.last_review_prediction = ReviewModelPrediction(
                priority=str(review_priority),
                review_first_score=float(review_score),
                source=str(result.get("review_priority_source", "Experimental MHIST annotator-agreement proxy (Modal)")),
            )
        embedding = result.get("embedding")
        if embedding is not None:
            embedding = tuple(float(value) for value in embedding)
        provider_name = MODAL_UNI_PROVIDER_NAME if self.provider_kind == "uni" else MODAL_HIBOU_PROVIDER_NAME
        return AttentionResult(
            overlay=_decode_png(str(result["overlay_png"]), "overlay"),
            heatmap=_decode_png(str(result["heatmap_png"]), "heatmap"),
            explanation=str(result.get("explanation", f"{provider_name} feature variation.")),
            visual_complexity_score=float(result.get("visual_complexity_score", 0.0)),
            provider_name=str(result.get("provider_name", provider_name)),
            is_demonstration=True,
            uses_trained_encoder=True,
            priority_score_source=str(result.get("priority_score_source", "Deterministic visual-complexity heuristic")),
            overlay_caption=str(result.get("overlay_caption", "Exploratory remote feature-variation overlay; not diagnostic.")),
            embedding=embedding,
            embedding_model=result.get("embedding_model"),
            image_priority_score=(
                float(result["image_priority_score"])
                if result.get("image_priority_score") is not None
                else None
            ),
        )


class ModalHibouFeatureProvider(ModalFeatureProvider):
    """Backward-compatible Hibou-specific alias."""

    def __init__(self, url: str | None = None, timeout: int = _DEFAULT_TIMEOUT_SECONDS) -> None:
        super().__init__("hibou", url=url, timeout=timeout)


__all__ = [
    "MODAL_HIBOU_PROVIDER_NAME",
    "MODAL_UNI_PROVIDER_NAME",
    "ModalFeatureProvider",
    "ModalHibouFeatureProvider",
    "ModalProviderStatus",
    "get_modal_provider_status",
]
