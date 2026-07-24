"""Deploy the complete remote PathologyAI model worker on Modal.

The endpoint accepts ``provider_kind`` values ``uni`` or ``hibou`` and an
optional ``use_review_model`` flag. Model weights are downloaded into a private
Modal Volume using the ``huggingface`` Modal Secret. The local app never needs
to receive the Hugging Face token.

Commands:
    modal serve modal_app.py
    modal deploy modal_app.py
"""

from __future__ import annotations

import base64
from io import BytesIO
import os
from pathlib import Path

import modal


APP_NAME = "pathologyai-model-api"
MODEL_VOLUME = modal.Volume.from_name("pathologyai-hibou-models", create_if_missing=True)
MODEL_ROOT = Path("/models")
UNI_DIR = MODEL_ROOT / "uni"
HIBOU_DIR = MODEL_ROOT / "hibou-b"
REVIEW_HEAD_DIR = Path("/review_priority_head")
REVIEW_HEAD_REMOTE_DIR = "/review_priority_head"
LOCAL_REVIEW_HEAD_DIR = Path(__file__).resolve().parent / "models" / "review_priority_head"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]",
        "numpy",
        "Pillow",
        "torch",
        "torchvision",
        "transformers==4.57.6",
        "huggingface_hub==0.36.0",
        "timm==1.0.28",
        "joblib",
        "scikit-learn",
    )
    .add_local_python_source("pathology_ai", copy=True)
)
if LOCAL_REVIEW_HEAD_DIR.is_dir():
    image = image.add_local_dir(str(LOCAL_REVIEW_HEAD_DIR), REVIEW_HEAD_REMOTE_DIR, copy=True)

app = modal.App(APP_NAME)


def _ensure_snapshot(repo_id: str, target: Path) -> Path:
    if not (target / "config.json").is_file() and not (target / "pytorch_model.bin").is_file():
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target),
            token=os.environ.get("HF_TOKEN") or None,
        )
        MODEL_VOLUME.commit()
    return target


def _ensure_model(provider_kind: str) -> Path:
    """Download only the model needed for this request into the mounted volume."""

    if provider_kind == "uni":
        return _ensure_snapshot("MahmoodLab/UNI", UNI_DIR)
    return _ensure_snapshot("histai/hibou-b", HIBOU_DIR)


def _png_base64(image) -> str:
    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    return base64.b64encode(output.getvalue()).decode("ascii")


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/models": MODEL_VOLUME},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=600,
    scaledown_window=120,
)
@modal.fastapi_endpoint(method="POST", docs=True, requires_proxy_auth=True)
def infer(payload: dict) -> dict:
    """Run remote UNI/Hibou-B inference and optionally the MHIST head."""

    import base64 as _base64
    from io import BytesIO as _BytesIO
    from PIL import Image

    from pathology_ai.hibou_provider import LocalHibouFeatureProvider
    from pathology_ai.review_model import LocalPrototypeReviewHead, get_review_model_status
    from pathology_ai.uni_provider import LocalUNIFeatureProvider

    if not isinstance(payload, dict):
        raise ValueError("The request body must be an object.")
    provider_kind = payload.get("provider_kind", "hibou")
    if provider_kind not in {"uni", "hibou"}:
        raise ValueError("provider_kind must be 'uni' or 'hibou'.")
    encoded = payload.get("image_base64")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("image_base64 is required.")
    try:
        image = Image.open(_BytesIO(_base64.b64decode(encoded, validate=True))).convert("RGB")
    except Exception as exc:
        raise ValueError("image_base64 must contain a valid image.") from exc

    model_dir = _ensure_model(provider_kind)
    if provider_kind == "uni":
        result = LocalUNIFeatureProvider(model_dir / "pytorch_model.bin").analyze(image)
    else:
        result = LocalHibouFeatureProvider(model_dir).analyze(image)

    response = {
        "overlay_png": _png_base64(result.overlay),
        "heatmap_png": _png_base64(result.heatmap),
        "provider_name": result.provider_name.replace("Local ", "Modal "),
        "explanation": result.explanation.replace("local ", "remote "),
        "visual_complexity_score": result.visual_complexity_score,
        "priority_score_source": result.priority_score_source,
        "overlay_caption": result.overlay_caption.replace("local ", "remote "),
        "embedding_model": result.embedding_model,
        "embedding": list(result.embedding) if result.embedding is not None else None,
    }

    if payload.get("use_review_model") and provider_kind == "uni":
        status = get_review_model_status(REVIEW_HEAD_DIR)
        if status.ready:
            prediction = LocalPrototypeReviewHead(status).predict(result.embedding)
            response.update(
                {
                    "review_priority": prediction.priority,
                    "review_first_score": prediction.review_first_score,
                    "review_priority_source": prediction.source.replace("local", "Modal"),
                }
            )
    return response
