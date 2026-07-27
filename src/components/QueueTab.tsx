import { useEffect, useMemo, useState, type FormEvent } from "react";
import { DataTable, MetricsCards, PriorityChip, TextList } from "./common";
import { workspaceApi } from "../api";
import { PRIORITIES, type BatchState, type CaptionOutput, type ImageRecord, type ImageView, type Priority, type RegionCaption, type RegionCaptionsResponse, type ReviewFilter, type ReviewPayload } from "../types";
import { compareQueueRecords, effectivePriority } from "../queue";

export function QueueTab({
  batch,
  selectedId,
  onSelect,
  statusFilter,
  onStatusFilter,
  priorityFilter,
  onPriorityFilter,
  onSave,
  onReopen,
  onApplyGroup,
  disclaimer,
}: {
  batch: BatchState;
  selectedId: string | null;
  onSelect: (id: string) => void;
  statusFilter: ReviewFilter;
  onStatusFilter: (value: ReviewFilter) => void;
  priorityFilter: "All" | Priority;
  onPriorityFilter: (value: "All" | Priority) => void;
  onSave: (id: string, payload: ReviewPayload) => Promise<void>;
  onReopen: (id: string) => Promise<void>;
  onApplyGroup: (id: string, groupId: string) => Promise<void>;
  disclaimer: string;
}) {
  const visible = useMemo(
    () => batch.records.filter((record) => {
      const statusMatches = statusFilter === "All"
        || (statusFilter === "Reviewed") === record.review.reviewed;
      return statusMatches && (priorityFilter === "All" || effectivePriority(record) === priorityFilter);
    }).sort(compareQueueRecords),
    [batch.records, statusFilter, priorityFilter],
  );
  const detailId = selectedId ?? visible[0]?.id ?? null;
  const selected = batch.records.find((record) => record.id === detailId) ?? null;
  return (
    <section className="panel tab-panel">
      <div className="section-heading">
        <div><p className="eyebrow">Current batch</p><h2>Review queue</h2></div>
        <span className="record-total">{visible.length} shown</span>
      </div>
      <MetricsCards metrics={batch.metrics} />
      <div className="queue">
        <aside>
          <div className="filters">
            <div>
              <label htmlFor="review-status">Review status</label>
              <select id="review-status" value={statusFilter} onChange={(event) => onStatusFilter(event.target.value as ReviewFilter)}>
                <option>All</option><option>Awaiting</option><option>Reviewed</option>
              </select>
            </div>
            <div>
              <label htmlFor="priority-filter">Priority</label>
              <select id="priority-filter" value={priorityFilter} onChange={(event) => onPriorityFilter(event.target.value as "All" | Priority)}>
                <option>All</option>
                {PRIORITIES.map((priority) => <option key={priority}>{priority}</option>)}
              </select>
            </div>
          </div>
          <div className="queue-list" aria-label="Review records">
            {visible.length ? visible.map((record) => (
              <button
                className={`record ${record.id === detailId ? "selected" : ""}`}
                key={record.id}
                type="button"
                aria-pressed={record.id === detailId}
                onClick={() => onSelect(record.id)}
              >
                <span className="record-name">{record.name}</span>
                <small>{record.dimensions.join(" × ")} px · {record.review.reviewed ? "Reviewed" : "Awaiting"}</small>
                <PriorityChip priority={record.review.priority || record.triage.suggested_priority} />
              </button>
            )) : <div className="empty"><strong>No matching records</strong><span>Adjust the review status or priority filter.</span></div>}
          </div>
        </aside>
        <div className="detail">
          {selected ? (
            <RecordDetail
              record={selected}
              records={visible}
              onSelect={onSelect}
              onSave={onSave}
              onReopen={onReopen}
              onApplyGroup={onApplyGroup}
              disclaimer={disclaimer}
            />
          ) : <div className="empty"><strong>No image selected</strong><span>Choose a record to begin human review.</span></div>}
        </div>
      </div>
      <SkippedFiles batch={batch} />
    </section>
  );
}

function RecordDetail({
  record,
  records,
  onSelect,
  onSave,
  onReopen,
  onApplyGroup,
  disclaimer,
}: {
  record: ImageRecord;
  records: ImageRecord[];
  onSelect: (id: string) => void;
  onSave: (id: string, payload: ReviewPayload) => Promise<void>;
  onReopen: (id: string) => Promise<void>;
  onApplyGroup: (id: string, groupId: string) => Promise<void>;
  disclaimer: string;
}) {
  const [view, setView] = useState<ImageView>("original");
  const [captions, setCaptions] = useState<RegionCaptionsResponse | null>(null);
  const [captionLoading, setCaptionLoading] = useState(false);
  const [captionError, setCaptionError] = useState<string | null>(null);
  useEffect(() => setView("original"), [record.id]);
  useEffect(() => {
    setCaptions(null);
    setCaptionError(null);
  }, [record.id]);
  useEffect(() => {
    if (view !== "overlay" || captions) return;
    let active = true;
    setCaptionLoading(true);
    workspaceApi.regionCaptions(record.id)
      .then((value) => active && setCaptions(value))
      .catch((reason: unknown) => active && setCaptionError(reason instanceof Error ? reason.message : "Caption request failed"))
      .finally(() => active && setCaptionLoading(false));
    return () => { active = false; };
  }, [captions, record.id, view]);
  const caption = view === "original"
    ? "Original image for human review."
    : view === "overlay" ? record.attention.overlay_caption : record.attention.explanation;
  const regions = captions?.regions ?? [];
  const guidance = uniqueGuidance(regions.map((region) => region.caption));

  return (
    <article className="record-detail">
      <div className="detail-heading">
        <div><p className="eyebrow">Selected image</p><h2>{record.name}</h2></div>
        <span className={`quality-state ${record.quality.adequate ? "pass" : "fail"}`}>
          {record.quality.adequate ? "Quality passed" : "Quality issue"}
        </span>
      </div>
      <div className="detail-grid">
        <div>
          <div className="viewer">
            <div className="viewer-canvas">
              <img src={record.images[view]} alt={`${view} image preview for ${record.name}`} />
              {view === "overlay" && (captions?.regions ?? record.computed?.regions ?? []).map((region) => (
                <span
                  className="region-marker"
                  key={region.region_id}
                  style={{ left: `${region.x * 100}%`, top: `${region.y * 100}%`, width: `${region.width * 100}%`, height: `${region.height * 100}%` }}
                  title={`Region ${region.region_id}: ${region.location}, ${region.contribution_percentage.toFixed(0)}% of priority score`}
                >
                  <b>{region.region_id}</b>
                </span>
              ))}
            </div>
          </div>
          <div className="viewer-tools" role="group" aria-label="Image view">
            {(["original", "overlay", "heatmap"] as ImageView[]).map((kind) => (
              <button className={view === kind ? "active" : ""} key={kind} type="button" aria-pressed={view === kind} onClick={() => setView(kind)}>
                {kind === "original" ? "Original" : kind === "overlay" ? "Feature overlay" : "Heatmap"}
              </button>
            ))}
          </div>
          <p className="viewer-caption">{caption}</p>
          {view === "overlay" && captionLoading && <p className="muted">Computing region captions…</p>}
          {view === "overlay" && captionError && <p className="inline-error">{captionError}</p>}
          {view === "overlay" && regions.length > 0 && (
            <section className="region-captions" aria-label="Computed region captions">
              <h3>Highlighted regions</h3>
              {regions.map((region) => <RegionCaptionCard key={region.region_id} region={region} disclaimer={disclaimer} />)}
            </section>
          )}
          {view === "overlay" && guidance.length > 0 && (
            <section className="workflow-guidance" aria-label="Workflow guidance">
              <h3>Workflow guidance</h3>
              {guidance.map((value) => <p key={value}>{appendDisclaimer(value, disclaimer)}</p>)}
            </section>
          )}
          <div className="facts">
            <div className="fact"><small>Suggested priority</small><PriorityChip priority={record.triage.suggested_priority} /></div>
            <div className="fact"><small>Review method</small><span>{record.triage.priority_source}</span></div>
            <div className="fact"><small>Image</small><span>{record.dimensions.join(" × ")} px · {record.size}</span></div>
            <div className="fact"><small>Quality</small><span>{record.quality.adequate ? "Passed" : "Needs attention"}</span></div>
            <div className="fact"><small>Feature priority score</small><span>{record.computed ? record.computed.priority_score.toFixed(2) : "Unavailable"}</span></div>
            <div className="fact"><small>UNI/Hibou-B agreement</small><span>{formatAgreement(record.computed?.model_agreement_score ?? null)}</span></div>
          </div>
          <section className="findings"><h3>Quality checks</h3><TextList values={record.quality.reasons} empty="No blocking quality issues." /></section>
          <section className="findings"><h3>Advisories</h3><TextList values={record.quality.advisories} empty="No nonblocking advisories." /></section>
          {record.metadata_notes.length > 0 && <section className="findings"><h3>File notes</h3><TextList values={record.metadata_notes} empty="No file notes." /></section>}
        </div>
        <ReviewForm record={record} records={records} onSelect={onSelect} onSave={onSave} onReopen={onReopen} onApplyGroup={onApplyGroup} />
      </div>
    </article>
  );
}

function RegionCaptionCard({ region, disclaimer }: { region: RegionCaption; disclaimer: string }) {
  const caption = region.caption;
  const text = caption.fallback_triggered
    ? caption.priority_reason
    : [caption.priority_reason, caption.visual_description].filter(Boolean).join(" ");
  return (
    <article className="region-caption-card">
      <div className="region-caption-heading"><strong>Region {region.region_id}</strong><span>{region.location} · {region.contribution_percentage.toFixed(0)}%</span></div>
      <p>{appendDisclaimer(text, disclaimer)}</p>
    </article>
  );
}

function uniqueGuidance(captions: CaptionOutput[]) {
  return Array.from(new Set(captions
    .filter((caption) => !caption.fallback_triggered && caption.workflow_guidance)
    .map((caption) => caption.workflow_guidance as string)));
}

function appendDisclaimer(value: string, disclaimer: string) {
  return `${value} ${disclaimer}`;
}

function formatAgreement(value: number | null) {
  return value == null ? "Unavailable" : value.toFixed(2);
}

function ReviewForm({ record, records, onSelect, onSave, onReopen, onApplyGroup }: {
  record: ImageRecord;
  records: ImageRecord[];
  onSelect: (id: string) => void;
  onSave: (id: string, payload: ReviewPayload) => Promise<void>;
  onReopen: (id: string) => Promise<void>;
  onApplyGroup: (id: string, groupId: string) => Promise<void>;
}) {
  const [priority, setPriority] = useState(record.review.priority);
  const [notes, setNotes] = useState(record.review.notes);
  const [groupId, setGroupId] = useState(record.review.group_id);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    setPriority(record.review.priority);
    setNotes(record.review.notes);
    setGroupId(record.review.group_id);
  }, [record.id, record.review.priority, record.review.notes, record.review.group_id]);

  const index = records.findIndex((candidate) => candidate.id === record.id);
  const nextUnreviewed = index < 0
    ? undefined
    : records.slice(index + 1).find((candidate) => !candidate.review.reviewed);
  const payload = (): ReviewPayload => ({ priority, notes, group_id: groupId });

  async function save(event?: FormEvent) {
    event?.preventDefault();
    setBusy(true);
    try { await onSave(record.id, payload()); } finally { setBusy(false); }
  }

  async function saveNext() {
    await save();
    if (nextUnreviewed) onSelect(nextUnreviewed.id);
  }

  return (
    <form className="review-form" onSubmit={save}>
      <p className="eyebrow">Reviewer decision</p>
      <h3>{record.review.reviewed ? "Update completed review" : "Complete review"}</h3>
      <label htmlFor="review-priority">Confirm or override suggested priority</label>
      <select id="review-priority" name="priority" value={priority} onChange={(event) => setPriority(event.target.value as Priority)}>
        {PRIORITIES.map((value) => <option key={value}>{value}</option>)}
      </select>
      <label htmlFor="review-notes">Reviewer notes {priority !== record.triage.suggested_priority ? "(required for override)" : ""}</label>
      <textarea
        id="review-notes"
        name="notes"
        value={notes}
        onChange={(event) => setNotes(event.target.value)}
        placeholder="Notes are retained only for this local browser session until export."
      />
      <label htmlFor="group-id">De-identified case/slide group ID (optional)</label>
      <input id="group-id" name="group_id" maxLength={64} value={groupId} onChange={(event) => setGroupId(event.target.value)} placeholder="letters, numbers, hyphens, underscores" />
      <div className="actions review-actions">
        <button className="primary" disabled={busy} type="submit">{record.review.reviewed ? "Update review" : "Save review"}</button>
        {nextUnreviewed && <button disabled={busy} type="button" onClick={saveNext}>Save &amp; next unreviewed</button>}
        <button disabled={busy} type="button" onClick={() => onApplyGroup(record.id, groupId)}>Apply group to source</button>
        {record.review.reviewed && <button className="danger" disabled={busy} type="button" onClick={() => onReopen(record.id)}>Reopen review</button>}
      </div>
      <p className="review-status">
        {record.review.reviewed ? `Reviewed at ${record.review.reviewed_at_utc}.` : "A reviewer must save this record before it is included in the export."}
      </p>
    </form>
  );
}

function SkippedFiles({ batch }: { batch: BatchState }) {
  return (
    <details className="skipped-files">
      <summary>Skipped files ({batch.skipped.length})</summary>
      <DataTable
        headers={["Upload", "File", "Reason"]}
        rows={batch.skipped.map((file) => [file.source_name, file.file_name, file.reason])}
        empty="No files were skipped."
      />
    </details>
  );
}
