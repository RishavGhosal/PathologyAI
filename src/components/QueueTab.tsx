import { useEffect, useMemo, useState, type FormEvent } from "react";
import { DataTable, MetricsCards, PriorityChip, TextList } from "./common";
import { PRIORITIES, type BatchState, type ImageRecord, type ImageView, type Priority, type ReviewFilter, type ReviewPayload } from "../types";
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
}: {
  record: ImageRecord;
  records: ImageRecord[];
  onSelect: (id: string) => void;
  onSave: (id: string, payload: ReviewPayload) => Promise<void>;
  onReopen: (id: string) => Promise<void>;
  onApplyGroup: (id: string, groupId: string) => Promise<void>;
}) {
  const [view, setView] = useState<ImageView>("original");
  useEffect(() => setView("original"), [record.id]);
  const caption = view === "original"
    ? "Original image for human review."
    : view === "overlay" ? record.attention.overlay_caption : record.attention.explanation;

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
          <div className="viewer"><img src={record.images[view]} alt={`${view} image preview for ${record.name}`} /></div>
          <div className="viewer-tools" role="group" aria-label="Image view">
            {(["original", "overlay", "heatmap"] as ImageView[]).map((kind) => (
              <button className={view === kind ? "active" : ""} key={kind} type="button" aria-pressed={view === kind} onClick={() => setView(kind)}>
                {kind === "original" ? "Original" : kind === "overlay" ? "Feature overlay" : "Heatmap"}
              </button>
            ))}
          </div>
          <p className="viewer-caption">{caption}</p>
          <div className="facts">
            <div className="fact"><small>Suggested priority</small><PriorityChip priority={record.triage.suggested_priority} /></div>
            <div className="fact"><small>Review method</small><span>{record.triage.priority_source}</span></div>
            <div className="fact"><small>Image</small><span>{record.dimensions.join(" × ")} px · {record.size}</span></div>
            <div className="fact"><small>Quality</small><span>{record.quality.adequate ? "Passed" : "Needs attention"}</span></div>
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
