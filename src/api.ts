import type {
  DomainContext,
  ProviderKind,
  ReviewPayload,
  WorkspaceSnapshot,
} from "./types";

async function workspaceRequest(path: string, init?: RequestInit): Promise<WorkspaceSnapshot> {
  const response = await fetch(path, init);
  const payload = (await response.json()) as WorkspaceSnapshot | { error?: string };
  if (!response.ok) {
    throw new Error("error" in payload && payload.error ? payload.error : "Request failed");
  }
  return payload as WorkspaceSnapshot;
}

function postJson(path: string, body: object): Promise<WorkspaceSnapshot> {
  return workspaceRequest(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export const workspaceApi = {
  status: () => workspaceRequest("/api/status"),
  upload: (
    files: File[],
    settings: {
      providerKind: ProviderKind;
      useReviewModel: boolean;
      domainContext: DomainContext;
      screeningSeconds: number;
    },
  ) => {
    const form = new FormData();
    files.forEach((file) => form.append("files", file));
    form.set("provider_kind", settings.providerKind);
    form.set("use_review_model", settings.useReviewModel ? "true" : "false");
    form.set("domain_context", settings.domainContext);
    form.set("screening_seconds", String(settings.screeningSeconds));
    return workspaceRequest("/api/upload", { method: "POST", body: form });
  },
  settings: (domainContext: DomainContext, screeningSeconds: number) =>
    postJson("/api/settings", {
      domain_context: domainContext,
      screening_seconds: screeningSeconds,
    }),
  saveReview: (id: string, payload: ReviewPayload) =>
    postJson(`/api/reviews/${id}`, payload),
  reopenReview: (id: string) => postJson(`/api/reviews/${id}/reopen`, {}),
  applyGroup: (id: string, groupId: string) =>
    postJson(`/api/groups/${id}`, { group_id: groupId }),
  reset: () => postJson("/api/reset", {}),
};
