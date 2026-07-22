import { useCallback, useEffect, useState } from "react";
import { workspaceApi } from "./api";
import type {
  DomainContext,
  ProviderKind,
  ReviewPayload,
  WorkspaceSnapshot,
} from "./types";

export function useWorkspace() {
  const [workspace, setWorkspace] = useState<WorkspaceSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    workspaceApi
      .status()
      .then((snapshot) => active && setWorkspace(snapshot))
      .catch((reason: unknown) => active && setError(errorMessage(reason)))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, []);

  const mutate = useCallback(async (request: () => Promise<WorkspaceSnapshot>) => {
    const snapshot = await request();
    setWorkspace(snapshot);
    return snapshot;
  }, []);

  const clearError = useCallback(() => setError(null), []);

  return {
    workspace,
    loading,
    error,
    clearError,
    upload: (
      files: File[],
      providerKind: ProviderKind,
      useReviewModel: boolean,
      domainContext: DomainContext,
      screeningSeconds: number,
    ) =>
      mutate(() =>
        workspaceApi.upload(files, {
          providerKind,
          useReviewModel,
          domainContext,
          screeningSeconds,
        }),
      ),
    saveReview: (id: string, payload: ReviewPayload) =>
      mutate(() => workspaceApi.saveReview(id, payload)),
    reopenReview: (id: string) => mutate(() => workspaceApi.reopenReview(id)),
    applyGroup: (id: string, groupId: string) =>
      mutate(() => workspaceApi.applyGroup(id, groupId)),
    reset: () => mutate(workspaceApi.reset),
  };
}

function errorMessage(reason: unknown): string {
  return reason instanceof Error ? reason.message : "Request failed";
}
