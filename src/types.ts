export const PRIORITIES = [
  "Review First",
  "Needs Better Image",
  "Lower Priority",
] as const;

export type Priority = (typeof PRIORITIES)[number];
export type ProviderKind = "deterministic" | "uni" | "hibou";
export type DomainContext = "unknown_or_other" | "mhist_like_colorectal_polyp";
export type TabId = "queue" | "dashboard" | "evaluation";
export type ImageView = "original" | "overlay" | "heatmap";
export type ReviewFilter = "All" | "Awaiting" | "Reviewed";

export interface ProviderStatus {
  ready: boolean;
  summary: string;
  detail: string;
}

export interface ReviewModelStatus extends ProviderStatus {
  metrics: Record<string, string | number>;
  evaluation_valid: boolean;
  evaluation_error: string | null;
}

export interface Providers {
  uni: ProviderStatus;
  hibou: ProviderStatus;
  review_model: ReviewModelStatus;
}

export interface WorkspaceSettings {
  domain_context: DomainContext;
  screening_seconds: number;
  provider_kind: ProviderKind;
  use_review_model: boolean;
}

export interface ReviewState {
  priority: Priority;
  notes: string;
  group_id: string;
  reviewed: boolean;
  reviewed_at_utc: string;
  [key: string]: unknown;
}

export interface QualityState {
  adequate: boolean;
  reasons: string[];
  advisories: string[];
  metrics: Record<string, number>;
  issue_codes: string[];
  advisory_codes: string[];
}

export interface TriageState {
  suggested_priority: Priority;
  explanation: string;
  priority_source: string;
  priority_method: string;
  review_first_score: number | null;
  fallback_reason: string | null;
}

export interface AttentionState {
  provider_name: string;
  explanation: string;
  overlay_caption: string;
  embedding_model: string | null;
  embedding_available: boolean;
}

export interface ImageRecord {
  id: string;
  name: string;
  source_name: string;
  file_type: string;
  size: string;
  dimensions: [number, number];
  quality: QualityState;
  triage: TriageState;
  attention: AttentionState;
  metadata_notes: string[];
  review: ReviewState;
  images: Record<ImageView, string>;
}

export interface AgreementRow {
  suggested_priority: Priority;
  reviewed_count: number;
  confirmed_count: number;
  overridden_count: number;
  agreement_percentage: number | null;
}

export interface OperationalMetrics {
  total_images: number;
  skipped_count: number;
  awaiting_count: number;
  reviewed_count: number;
  reviewed_percentage: number;
  effective_priority_counts: Record<Priority, number>;
  quality_pass_count: number;
  quality_issue_counts: Record<string, number>;
  quality_advisory_counts: Record<string, number>;
  experimental_model_prediction_count: number;
  deterministic_prediction_count: number;
  quality_gate_count: number;
  runtime_fallback_count: number;
  estimated_time_avoided_seconds: number;
  agreement_by_suggested_priority: AgreementRow[];
  [key: string]: unknown;
}

export interface SkippedFile {
  source_name: string;
  file_name: string;
  reason: string;
}

export interface BatchState {
  uploaded_count: number;
  records: ImageRecord[];
  skipped: SkippedFile[];
  metrics: OperationalMetrics;
}

export interface WorkspaceSnapshot {
  disclaimer: string;
  providers: Providers;
  settings?: WorkspaceSettings;
  batch: BatchState | null;
}

export interface ReviewPayload {
  priority: Priority;
  notes: string;
  group_id: string;
}
