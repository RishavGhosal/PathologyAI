import { useCallback, useEffect, useRef, useState } from "react";
import { Header, Toast } from "./components/common";
import { DashboardTab } from "./components/DashboardTab";
import { EvaluationTab } from "./components/EvaluationTab";
import { Overview } from "./components/Overview";
import { QueueTab } from "./components/QueueTab";
import { UploadSettingsForm, type UploadValues } from "./components/UploadSettingsForm";
import { useWorkspace } from "./useWorkspace";
import { compareQueueRecords } from "./queue";
import type { Priority, ReviewFilter, ReviewPayload, TabId } from "./types";

interface ToastState { kind: "success" | "error" | "info"; message: string }

export default function App() {
  const store = useWorkspace();
  const [activeTab, setActiveTab] = useState<TabId>("queue");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<ReviewFilter>("All");
  const [priorityFilter, setPriorityFilter] = useState<"All" | Priority>("All");
  const [uploading, setUploading] = useState(false);
  const [toast, setToast] = useState<ToastState | null>(null);
  const toastTimer = useRef<number | null>(null);

  const showToast = useCallback((message: string, kind: ToastState["kind"] = "success") => {
    if (toastTimer.current) window.clearTimeout(toastTimer.current);
    setToast({ message, kind });
    toastTimer.current = window.setTimeout(() => setToast(null), 5000);
  }, []);

  useEffect(() => () => {
    if (toastTimer.current) window.clearTimeout(toastTimer.current);
  }, []);

  useEffect(() => {
    if (store.error) {
      showToast(store.error, "error");
      store.clearError();
    }
  }, [store.error, store.clearError, showToast]);

  useEffect(() => {
    const records = store.workspace?.batch?.records ?? [];
    if (!records.length) {
      setSelectedId(null);
    } else if (!selectedId || !records.some((record) => record.id === selectedId)) {
      setSelectedId([...records].sort(compareQueueRecords)[0].id);
    }
  }, [store.workspace, selectedId]);

  async function handleUpload(values: UploadValues) {
    setUploading(true);
    showToast("Processing files. Large images or local models can take a moment.", "info");
    try {
      const snapshot = await store.upload(values.files, values.providerKind, values.useReviewModel, values.domainContext, values.screeningSeconds);
      setActiveTab("queue");
      setSelectedId(snapshot.batch ? [...snapshot.batch.records].sort(compareQueueRecords)[0]?.id ?? null : null);
      showToast("Batch processed.");
    } catch (reason) {
      showToast(message(reason), "error");
    } finally {
      setUploading(false);
    }
  }

  async function handleReview(id: string, payload: ReviewPayload) {
    try { await store.saveReview(id, payload); showToast("Review saved."); }
    catch (reason) { showToast(message(reason), "error"); throw reason; }
  }

  async function handleReopen(id: string) {
    try { await store.reopenReview(id); showToast("Review reopened."); }
    catch (reason) { showToast(message(reason), "error"); }
  }

  async function handleApplyGroup(id: string, groupId: string) {
    try { await store.applyGroup(id, groupId); showToast("Group ID applied to images from this upload source."); }
    catch (reason) { showToast(message(reason), "error"); }
  }

  async function handleReset() {
    if (!window.confirm("Discard this local batch and its unsaved session state?")) return;
    try {
      await store.reset();
      setSelectedId(null);
      setActiveTab("queue");
      setStatusFilter("All");
      setPriorityFilter("All");
      showToast("Local batch cleared.");
    } catch (reason) { showToast(message(reason), "error"); }
  }

  if (store.loading) {
    return <main className="loading-screen"><div className="loading-mark" /><p>Loading local review workspace…</p></main>;
  }

  if (!store.workspace) {
    return (
      <>
        {toast && <div className="toast-region"><Toast {...toast} /></div>}
        <main className="loading-screen">
          <p>Unable to load the local review workspace.</p>
        </main>
      </>
    );
  }

  const workspace = store.workspace;
  return (
    <>
      <Header onReset={handleReset} />
      {toast && <div className="toast-region"><Toast {...toast} /></div>}
      <main className="app-main">
        {!workspace.batch && <Overview disclaimer={workspace.disclaimer} providers={workspace.providers} />}
        <UploadSettingsForm providers={workspace.providers} settings={workspace.settings} busy={uploading} onSubmit={handleUpload} />
        {workspace.batch && (
          <>
            <nav className="tabs" role="tablist" aria-label="Workspace views">
              {(["queue", "dashboard", "evaluation"] as TabId[]).map((tab) => (
                <button
                  id={`tab-${tab}`}
                  key={tab}
                  type="button"
                  role="tab"
                  aria-selected={activeTab === tab}
                  aria-controls={`panel-${tab}`}
                  className={activeTab === tab ? "active" : ""}
                  onClick={() => setActiveTab(tab)}
                >
                  {tab === "queue" ? "Review Queue" : tab === "dashboard" ? "Operational Dashboard" : "Model Evaluation & Limits"}
                </button>
              ))}
            </nav>
            <div className="tab-content" id={`panel-${activeTab}`} role="tabpanel" aria-labelledby={`tab-${activeTab}`}>
              {activeTab === "queue" && (
                <QueueTab
                  batch={workspace.batch}
                  selectedId={selectedId}
                  onSelect={setSelectedId}
                  statusFilter={statusFilter}
                  onStatusFilter={setStatusFilter}
                  priorityFilter={priorityFilter}
                  onPriorityFilter={setPriorityFilter}
                  onSave={handleReview}
                  onReopen={handleReopen}
                  onApplyGroup={handleApplyGroup}
                  disclaimer={workspace.disclaimer}
                />
              )}
              {activeTab === "dashboard" && <DashboardTab batch={workspace.batch} />}
              {activeTab === "evaluation" && <EvaluationTab workspace={workspace} />}
            </div>
          </>
        )}
      </main>
    </>
  );
}

function message(reason: unknown) {
  return reason instanceof Error ? reason.message : "Request failed";
}
