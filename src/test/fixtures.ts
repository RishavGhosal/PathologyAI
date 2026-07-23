import type { ImageRecord, OperationalMetrics, WorkspaceSnapshot } from "../types";

export const disclaimer = "This research and education prototype provides review-priority suggestions only. It does not provide a medical diagnosis and does not replace review by a qualified pathologist.";

export const providers = {
  uni: { ready: true, summary: "Local UNI encoder is ready", detail: "UNI live detail" },
  hibou: { ready: true, summary: "Local Hibou-B encoder is ready", detail: "Hibou live detail" },
  review_model: {
    ready: true,
    summary: "Experimental priority head is ready",
    detail: "Review-head live detail",
    metrics: { balanced_accuracy: 0.81234, sample_count: 100 },
    evaluation_valid: true,
    evaluation_error: null,
  },
};

export const metrics: OperationalMetrics = {
  total_images: 2,
  skipped_count: 1,
  awaiting_count: 2,
  reviewed_count: 0,
  reviewed_percentage: 0,
  effective_priority_counts: { "Review First": 1, "Needs Better Image": 1, "Lower Priority": 0 },
  quality_pass_count: 1,
  quality_issue_counts: { blur: 1 },
  quality_advisory_counts: {},
  experimental_model_prediction_count: 0,
  deterministic_prediction_count: 1,
  quality_gate_count: 1,
  runtime_fallback_count: 0,
  estimated_time_avoided_seconds: 60,
  agreement_by_suggested_priority: [
    { suggested_priority: "Review First", reviewed_count: 0, confirmed_count: 0, overridden_count: 0, agreement_percentage: null },
    { suggested_priority: "Needs Better Image", reviewed_count: 0, confirmed_count: 0, overridden_count: 0, agreement_percentage: null },
    { suggested_priority: "Lower Priority", reviewed_count: 0, confirmed_count: 0, overridden_count: 0, agreement_percentage: null },
  ],
};

function record(id: string, name: string, suggested: "Review First" | "Needs Better Image", adequate: boolean): ImageRecord {
  return {
    id,
    name,
    source_name: id === "one" ? "batch.zip" : "second.png",
    file_type: "PNG",
    size: "12.0 KB",
    dimensions: [256, 256],
    quality: {
      adequate,
      reasons: adequate ? [] : ["Image appears blurred."],
      advisories: [],
      metrics: { brightness: 120 },
      issue_codes: adequate ? [] : ["blur"],
      advisory_codes: [],
    },
    triage: {
      suggested_priority: suggested,
      explanation: "Queue-order explanation",
      priority_source: adequate ? "Deterministic visual-complexity heuristic" : "Quality gate",
      priority_method: adequate ? "deterministic" : "quality_gate",
      review_first_score: null,
      fallback_reason: null,
    },
    attention: {
      provider_name: "Deterministic demonstration attention",
      explanation: "Heatmap explanation",
      overlay_caption: "Feature overlay caption",
      embedding_model: null,
      embedding_available: false,
    },
    metadata_notes: id === "one" ? ["First frame shown."] : [],
    review: {
      priority: suggested,
      notes: "",
      group_id: "",
      reviewed: false,
      reviewed_at_utc: "",
    },
    images: {
      original: `/api/images/${id}/original`,
      overlay: `/api/images/${id}/overlay`,
      heatmap: `/api/images/${id}/heatmap`,
    },
  };
}

export function preUpload(): WorkspaceSnapshot {
  return { disclaimer, providers: structuredClone(providers), batch: null };
}

export function withBatch(): WorkspaceSnapshot {
  return {
    disclaimer,
    providers: structuredClone(providers),
    settings: {
      domain_context: "unknown_or_other",
      screening_seconds: 30,
      provider_kind: "deterministic",
      use_review_model: false,
    },
    batch: {
      uploaded_count: 2,
      records: [
        record("one", "slide-a.png", "Review First", true),
        record("two", "slide-b.png", "Needs Better Image", false),
      ],
      skipped: [{ source_name: "batch.zip", file_name: "notes.txt", reason: "Unsupported file type." }],
      metrics: structuredClone(metrics),
    },
  };
}

export function response(payload: WorkspaceSnapshot, ok = true): Promise<Response> {
  return Promise.resolve({ ok, json: () => Promise.resolve(payload) } as Response);
}
