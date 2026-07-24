import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import type {
  DomainContext,
  ProviderKind,
  Providers,
  WorkspaceSettings,
} from "../types";

const DEFAULT_SETTINGS: WorkspaceSettings = {
  provider_kind: "deterministic",
  use_review_model: false,
  domain_context: "unknown_or_other",
  screening_seconds: 30,
};

export interface UploadValues {
  files: File[];
  providerKind: ProviderKind;
  useReviewModel: boolean;
  domainContext: DomainContext;
  screeningSeconds: number;
}

export function UploadSettingsForm({
  providers,
  settings,
  busy,
  onSubmit,
}: {
  providers: Providers;
  settings?: WorkspaceSettings;
  busy: boolean;
  onSubmit: (values: UploadValues) => Promise<void>;
}) {
  const active = settings ?? DEFAULT_SETTINGS;
  const [providerKind, setProviderKind] = useState<ProviderKind>(active.provider_kind);
  const [domainContext, setDomainContext] = useState<DomainContext>(active.domain_context);
  const [screeningSeconds, setScreeningSeconds] = useState(active.screening_seconds);
  const [useReviewModel, setUseReviewModel] = useState(active.use_review_model);
  const fileInput = useRef<HTMLInputElement>(null);

  function preferredProviderKind(kind: "uni" | "hibou"): ProviderKind {
    if (kind === "uni") return providers.uni.ready ? "uni" : providers.modal_uni.ready ? "modal_uni" : "uni";
    return providers.hibou.ready ? "hibou" : providers.modal_hibou.ready ? "modal_hibou" : "hibou";
  }

  useEffect(() => {
    const next = settings ?? DEFAULT_SETTINGS;
    setProviderKind(next.provider_kind === "modal_uni" || next.provider_kind === "uni" ? preferredProviderKind("uni") : next.provider_kind === "modal_hibou" || next.provider_kind === "hibou" ? preferredProviderKind("hibou") : next.provider_kind);
    setDomainContext(next.domain_context);
    setScreeningSeconds(next.screening_seconds);
    setUseReviewModel(next.use_review_model);
    if (fileInput.current) fileInput.current.value = "";
  }, [settings, providers]);

  const selectedProvider = providerKind === "modal_uni" ? "uni" : providerKind === "modal_hibou" ? "hibou" : providerKind;

  const providerHelp = useMemo(() => {
    if (providerKind === "uni") return providers.uni;
    if (providerKind === "hibou") return providers.hibou;
    if (providerKind === "modal_uni") return providers.modal_uni;
    if (providerKind === "modal_hibou") return providers.modal_hibou;
    return null;
  }, [providerKind, providers]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    const files = Array.from(fileInput.current?.files ?? []);
    if (!files.length) return;
    await onSubmit({ files, providerKind, useReviewModel, domainContext, screeningSeconds });
  }

  return (
    <form className="upload panel" onSubmit={submit}>
      <div className="section-heading">
        <div>
          <p className="eyebrow">Batch intake</p>
          <h2>Upload pathology images</h2>
        </div>
        <span className="format-note">PNG · JPG · TIFF · ZIP</span>
      </div>
      <p className="muted">
        ZIP contents are checked in memory; unsafe or unreadable entries are reported individually.
      </p>
      <label htmlFor="upload-files">Choose one or more files</label>
      <input
        id="upload-files"
        ref={fileInput}
        required
        multiple
        name="files"
        type="file"
        accept=".png,.jpg,.jpeg,.tif,.tiff,.zip,image/png,image/jpeg,image/tiff,application/zip"
      />
      <div className="settings">
        <div>
          <label htmlFor="provider-kind">Feature provider</label>
          <select
            id="provider-kind"
            name="provider_kind"
            value={selectedProvider}
            onChange={(event) => setProviderKind(event.target.value === "uni" || event.target.value === "hibou" ? preferredProviderKind(event.target.value) : event.target.value as ProviderKind)}
          >
            <option value="deterministic">Deterministic demonstration</option>
            <option value="uni" disabled={!providers.uni.ready && !providers.modal_uni.ready}>
              UNI feature exploration{providers.uni.ready || providers.modal_uni.ready ? "" : " (unavailable)"}
            </option>
            <option value="hibou" disabled={!providers.hibou.ready && !providers.modal_hibou.ready}>
              Hibou-B feature exploration{providers.hibou.ready || providers.modal_hibou.ready ? "" : " (unavailable)"}
            </option>
          </select>
          {providerHelp && (
            <div className="field-help" aria-live="polite">
              <strong>{providerHelp.summary}</strong>
              <span>{providerHelp.detail}</span>
            </div>
          )}
        </div>
        <div>
          <label htmlFor="domain-context">Batch tissue context</label>
          <select
            id="domain-context"
            name="domain_context"
            value={domainContext}
            onChange={(event) => setDomainContext(event.target.value as DomainContext)}
          >
            <option value="unknown_or_other">Unknown or other tissue</option>
            <option value="mhist_like_colorectal_polyp">MHIST-like colorectal-polyp patches</option>
          </select>
          <label htmlFor="screening-seconds">Manual screening seconds/image</label>
          <input
            id="screening-seconds"
            name="screening_seconds"
            type="number"
            min="0"
            max="600"
            step="5"
            value={screeningSeconds}
            onChange={(event) => setScreeningSeconds(Number(event.target.value))}
          />
        </div>
        <div className="model-option">
          <label className="check" htmlFor="use-review-model">
            <input
              id="use-review-model"
              name="use_review_model"
              type="checkbox"
              disabled={!providers.review_model.ready && !providers.modal_uni.ready}
              checked={useReviewModel}
              onChange={(event) => setUseReviewModel(event.target.checked)}
            />
            <span>Use experimental MHIST agreement-proxy head</span>
          </label>
          <div className="field-help">
            <strong>{providers.modal_uni.ready ? "Modal UNI review head is available" : providers.review_model.summary}</strong>
            <span>{providers.modal_uni.ready ? "The UNI embedding and MHIST proxy head will run remotely on Modal." : providers.review_model.detail}</span>
          </div>
        </div>
      </div>
      <div className="actions">
        <button className="primary" disabled={busy} type="submit">
          {busy ? "Processing files…" : "Process files"}
        </button>
      </div>
    </form>
  );
}
