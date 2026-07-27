"""Vision-caption generation with strict validation and deterministic fallback."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import os
import re
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image
from io import BytesIO


# Keep this prompt byte-for-byte aligned with the prompt approved in the
# product specification.  The retry adds constraints in the user message; it
# never edits this system prompt.
VALIDATED_SYSTEM_PROMPT = r'''# PERSONA
You are a visual pattern-description and workflow-routing assistant embedded in a pathology
image review tool. You are NOT a pathologist, diagnostician, or clinical advisor, and you must
never act like one.

# TASK
Given an image region, its computed priority score, and computed model-agreement/quality
signals, generate three short outputs:
(1) priority_reason — why this region was flagged, using only the given computed score
(2) visual_description — a neutral description of the region's visual/textural properties
(3) workflow_guidance — a process-level suggestion for the reviewer (not about the tissue)

# CONTEXT
- Regions are ranked by a computed feature-variation score from a vision encoder
  (UNI or Hibou-B), not by any clinical or diagnostic model.
- You will be given: the image region, its priority score/percentage, a model-agreement
  score between UNI and Hibou-B, and any available image-quality signal (e.g. contrast/blur).
- The reviewer seeing your output is a human who makes the actual review decision — your
  output is a visual and process aid, never a finding.

# CONSTRAINTS (must follow exactly)
- Never use clinical, diagnostic, or health-judgment language: no "abnormal," "concerning,"
  "healthy," "damaged," "cancer," "malignant," "diagnostic," or similar, under any framing.
- Never reference the tissue/cells in evaluative terms, and never suggest a clinical action
  (biopsy, treatment, diagnosis, follow-up care).
- priority_reason must only restate the numeric/computed reason given — no invented
  interpretation.
- visual_description is limited to strictly visual/textural vocabulary: texture, density,
  color, contrast, boundary shape, pattern regularity. No health/quality judgment.
- workflow_guidance is only about the REVIEW PROCESS — image quality, model agreement,
  routing to a second reviewer, recapture suggestions — never about what the tissue means.
- If you cannot produce visual_description or workflow_guidance without drifting into evaluative/clinical language, set that field to null and set fallback_triggered to true rather than forcing an unsafe output.

# OUTPUT FORMAT
Return only valid JSON, no other text:
{
  "priority_reason": string,
  "visual_description": string | null,
  "workflow_guidance": string | null,
  "fallback_triggered": boolean
}

# EXAMPLES
Input: region contributes 42% of image priority score, model agreement score low (0.3),
image contrast normal

Good priority_reason: "This region contributes the largest share (42%) of this image's
priority score, which is why it's flagged first."
Bad priority_reason: "This region looks the most concerning." (invents judgment — do not do this)

Good visual_description: "Denser, darker texture with irregular boundary lines compared to
the surrounding area."
Bad visual_description: "This area appears abnormal and may need attention." (clinical judgment + implied action — do not do this)

Good workflow_guidance: "UNI and Hibou-B show notably different rankings for this image —
consider routing to a second reviewer."
Bad workflow_guidance: "This may indicate early-stage changes worth investigating."
(clinical implication — do not do this)'''


FORBIDDEN_TERMS = (
    "abnormal", "concerning", "healthy", "damaged", "cancer", "malignant",
    "diagnostic", "diagnosis", "disease", "tumor", "lesion", "benign",
    "malignancy", "biopsy", "treatment", "clinical", "pathologist", "pathology",
    "tissue", "cell", "cells", "medical conclusion", "follow-up care",
    "suspicious", "worrisome", "risk", "investigate", "investigation",
)
VISUAL_EVALUATIVE_TERMS = (
    "attention", "review", "may", "might", "should", "suggest", "indicate",
    "important", "issue", "problem", "needs", "quality",
)
WORKFLOW_TERMS = (
    "review", "route", "routing", "recaptur", "agreement", "contrast", "blur",
    "image-quality", "image quality", "queue",
)


@dataclass(frozen=True)
class CaptionInput:
    region_id: int
    contribution_percentage: float
    priority_score: float
    model_agreement_score: float | None
    quality_signal: dict[str, float]
    location: str

    def as_json(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "region_contribution_percentage": round(self.contribution_percentage, 2),
            "image_priority_score": round(self.priority_score, 4),
            "model_agreement_score": None if self.model_agreement_score is None else round(self.model_agreement_score, 4),
            "image_quality_signal": self.quality_signal,
            "region_location": self.location,
        }


def computed_priority_reason(region: CaptionInput) -> str:
    percentage = _format_percentage(region.contribution_percentage)
    if region.contribution_percentage >= 40:
        return f"This region contributes the largest share ({percentage}) of this image's priority score, which is why it is flagged first."
    return f"This region contributes {percentage} of this image's priority score and is included in the priority queue."


def _format_percentage(value: float) -> str:
    rounded = round(value)
    return f"{rounded}%" if abs(value - rounded) < 0.05 else f"{value:.1f}%"


def _fallback(region: CaptionInput) -> dict[str, Any]:
    return {
        "priority_reason": computed_priority_reason(region),
        "visual_description": None,
        "workflow_guidance": None,
        "fallback_triggered": True,
    }


def _contains_forbidden(value: str) -> bool:
    lowered = value.casefold()
    return any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in FORBIDDEN_TERMS)


def validate_caption(payload: Any, region: CaptionInput) -> dict[str, Any] | None:
    """Validate without sanitizing; invalid output returns None for retry/fallback."""

    if not isinstance(payload, dict):
        return None
    expected_keys = {"priority_reason", "visual_description", "workflow_guidance", "fallback_triggered"}
    if set(payload) != expected_keys:
        return None
    if not isinstance(payload["priority_reason"], str) or not payload["priority_reason"].strip():
        return None
    if not isinstance(payload["fallback_triggered"], bool):
        return None
    for key in ("visual_description", "workflow_guidance"):
        value = payload[key]
        if value is not None and (not isinstance(value, str) or not value.strip() or _contains_forbidden(value)):
            return None
    visual = payload["visual_description"]
    if isinstance(visual, str) and any(re.search(rf"\b{re.escape(term)}\b", visual.casefold()) for term in VISUAL_EVALUATIVE_TERMS):
        return None
    guidance = payload["workflow_guidance"]
    if isinstance(guidance, str) and not any(term in guidance.casefold() for term in WORKFLOW_TERMS):
        return None
    if _contains_forbidden(payload["priority_reason"]):
        return None
    percentage = _format_percentage(region.contribution_percentage)
    if percentage not in payload["priority_reason"]:
        return None
    if payload["fallback_triggered"] and (payload["visual_description"] is not None or payload["workflow_guidance"] is not None):
        return None
    if not payload["fallback_triggered"] and (payload["visual_description"] is None or payload["workflow_guidance"] is None):
        return None
    return {
        "priority_reason": payload["priority_reason"].strip(),
        "visual_description": payload["visual_description"].strip() if isinstance(payload["visual_description"], str) else None,
        "workflow_guidance": payload["workflow_guidance"].strip() if isinstance(payload["workflow_guidance"], str) else None,
        "fallback_triggered": payload["fallback_triggered"],
    }


class VisionCaptionService:
    """Call a configured vision model, retry once, then fall back safely."""

    def __init__(self, request_fn: Callable[[dict[str, Any]], Any] | None = None) -> None:
        self.request_fn = request_fn

    def generate(self, image: Image.Image, region: CaptionInput) -> dict[str, Any]:
        api_key, endpoint, model = self._configuration()
        if not api_key and not self.request_fn:
            return _fallback(region)
        try:
            first = self._request(
                image,
                region,
                retry=False,
                api_key=api_key,
                endpoint=endpoint,
                model=model,
            )
            validated = validate_caption(first, region)
            if validated is not None:
                return validated
            second = self._request(
                image,
                region,
                retry=True,
                api_key=api_key,
                endpoint=endpoint,
                model=model,
            )
            validated = validate_caption(second, region)
            return validated if validated is not None else _fallback(region)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError, json.JSONDecodeError):
            return _fallback(region)

    @staticmethod
    def _configuration() -> tuple[str, str, str]:
        """Resolve GitHub Models first, with the original OpenAI path as a fallback."""

        github_token = os.environ.get("GITHUB_TOKEN", "").strip()
        if github_token:
            endpoint = os.environ.get("PATHOLOGYAI_CAPTION_ENDPOINT", "").strip()
            model = os.environ.get("PATHOLOGYAI_CAPTION_MODEL", "").strip()
            return (
                github_token,
                endpoint or "https://models.github.ai/inference/chat/completions",
                model or "openai/gpt-4.1",
            )

        endpoint = os.environ.get("PATHOLOGYAI_CAPTION_ENDPOINT", "").strip()
        model = os.environ.get("PATHOLOGYAI_CAPTION_MODEL", "").strip()
        return (
            os.environ.get("OPENAI_API_KEY", "").strip(),
            endpoint or "https://api.openai.com/v1/chat/completions",
            model or "gpt-4o",
        )

    def _request(
        self,
        image: Image.Image,
        region: CaptionInput,
        *,
        retry: bool,
        api_key: str,
        endpoint: str,
        model: str,
    ) -> Any:
        output = BytesIO()
        image.convert("RGB").save(output, format="JPEG", quality=90, optimize=True)
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        retry_text = "" if not retry else (
            " The previous response failed the safety validator. Re-answer with the same JSON "
            "schema. Do not use evaluative, clinical, diagnostic, tissue, or cell language; "
            "if either optional field cannot be produced safely, return null and set "
            "fallback_triggered to true."
        )
        request_body = {
            "model": model,
            "temperature": 0.1,
            "max_tokens": 240,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": VALIDATED_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Structured input:\n" + json.dumps(region.as_json(), sort_keys=True) + retry_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                    ],
                },
            ],
        }
        if self.request_fn:
            return self.request_fn(request_body)
        request = Request(
            endpoint,
            data=json.dumps(request_body).encode("utf-8"),
            headers={
                "Accept": "application/vnd.github+json" if "models.github.ai" in endpoint else "application/json",
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                **(
                    {"X-GitHub-Api-Version": os.environ.get("GITHUB_API_VERSION", "2022-11-28")}
                    if "models.github.ai" in endpoint
                    else {}
                ),
            },
            method="POST",
        )
        with urlopen(request, timeout=45) as response:
            result = json.loads(response.read())
        content = result["choices"][0]["message"]["content"]
        return json.loads(content)


__all__ = [
    "CaptionInput",
    "VALIDATED_SYSTEM_PROMPT",
    "VisionCaptionService",
    "computed_priority_reason",
    "validate_caption",
]
