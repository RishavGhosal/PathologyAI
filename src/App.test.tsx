import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { disclaimer, preUpload, response, withBatch } from "./test/fixtures";
import type { WorkspaceSnapshot } from "./types";

const fetchMock = vi.fn<typeof fetch>();

beforeEach(() => {
  vi.stubGlobal("fetch", fetchMock);
  fetchMock.mockReset();
});

afterEach(() => {
  cleanup();
  window.localStorage.removeItem("pathologyai-theme");
  window.localStorage.removeItem("pathologyai-accent");
  delete document.documentElement.dataset.theme;
  delete document.documentElement.dataset.accent;
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe("PathologyAI React frontend", () => {
  it("switches and persists the selected color theme", async () => {
    const user = userEvent.setup();
    fetchMock.mockImplementation(() => response(preUpload()));
    render(<App />);

    await screen.findByRole("heading", { name: "Human review, organized" });
    const theme = screen.getByLabelText("Color theme");
    expect(theme).toHaveValue("dark");
    await user.selectOptions(theme, "sage");
    expect(document.documentElement.dataset.theme).toBe("sage");
    expect(window.localStorage.getItem("pathologyai-theme")).toBe("sage");
    await user.selectOptions(screen.getByLabelText("Accent color"), "violet");
    expect(document.documentElement.dataset.accent).toBe("violet");
    expect(window.localStorage.getItem("pathologyai-accent")).toBe("violet");
  });

  it("renders the pre-upload contract, provider state, and exact multipart upload", async () => {
    const user = userEvent.setup();
    const initial = preUpload();
    const uploaded = withBatch();
    initial.providers.hibou.ready = false;
    initial.providers.modal_hibou.ready = false;
    fetchMock
      .mockImplementationOnce(() => response(initial))
      .mockImplementationOnce(() => response(uploaded));

    render(<App />);
    expect(await screen.findByRole("heading", { name: "Human review, organized" })).toBeVisible();
    expect(screen.getByText(disclaimer)).toBeVisible();
    expect(screen.getByText("Suggestions order a review queue only. They do not identify tissue, disease, cancer, or clinical urgency.")).toBeVisible();
    expect(screen.getByText("Optional local encoders never download weights at runtime and fall back to the deterministic method when unavailable.")).toBeVisible();
    expect(screen.getByText("PathologyAI")).toBeVisible();
    expect(document.body).not.toHaveTextContent("🔬");
    expect(screen.getByRole("option", { name: "Hibou-B feature exploration (unavailable)" })).toBeDisabled();

    await user.selectOptions(screen.getByLabelText("Feature provider"), "uni");
    expect(screen.getByText("Local UNI encoder is ready")).toBeVisible();
    expect(screen.getAllByText("UNI live detail").length).toBeGreaterThanOrEqual(2);

    const file = new File(["png"], "sample.png", { type: "image/png" });
    const uploadInput = screen.getByLabelText("Choose one or more files") as HTMLInputElement;
    await user.upload(uploadInput, file);
    expect(uploadInput.files).toHaveLength(1);
    await user.click(screen.getByLabelText("Use experimental MHIST agreement-proxy head"));
    fireEvent.submit(screen.getByRole("button", { name: "Process files" }).closest("form")!);

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    await screen.findByRole("tab", { name: "Review Queue" });
    const [, uploadInit] = fetchMock.mock.calls[1];
    expect(fetchMock.mock.calls[1][0]).toBe("/api/upload");
    expect(uploadInit?.method).toBe("POST");
    const form = uploadInit?.body as FormData;
    expect((form.get("files") as File).name).toBe("sample.png");
    expect(form.get("provider_kind")).toBe("uni");
    expect(form.get("use_review_model")).toBe("true");
    expect(form.get("domain_context")).toBe("unknown_or_other");
    expect(form.get("screening_seconds")).toBe("30");
    expect(screen.getByRole("status")).toHaveTextContent("Batch processed.");
  });

  it("supports tabs, live filters, record selection, image modes, and live override guidance", async () => {
    const user = userEvent.setup();
    fetchMock.mockImplementation(() => response(withBatch()));
    render(<App />);

    expect(await screen.findByRole("heading", { name: "Review queue" })).toBeVisible();
    expect(screen.getByRole("button", { name: /slide-a\.png/ })).toHaveAttribute("aria-pressed", "true");
    await user.click(screen.getByRole("button", { name: "Feature overlay" }));
    expect(screen.getByText("Feature overlay caption")).toBeVisible();
    expect(screen.getByRole("img")).toHaveAttribute("src", "/api/images/one/overlay");
    await user.click(screen.getByRole("button", { name: "Heatmap" }));
    expect(screen.getByText("Heatmap explanation")).toBeVisible();
    expect(screen.getByRole("img")).toHaveAttribute("src", "/api/images/one/heatmap");

    await user.click(screen.getByRole("button", { name: /slide-b\.png/ }));
    expect(screen.getByText("Original image for human review.")).toBeVisible();
    expect(screen.getByRole("img")).toHaveAttribute("src", "/api/images/two/original");
    await user.selectOptions(screen.getByLabelText("Priority"), "Review First");
    expect(screen.queryByRole("button", { name: /slide-b\.png/ })).not.toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText("Priority"), "All");
    await user.selectOptions(screen.getByLabelText("Review status"), "Reviewed");
    expect(screen.getByText("No matching records")).toBeVisible();
    expect(screen.getByRole("heading", { name: "slide-b.png" })).toBeVisible();
    await user.selectOptions(screen.getByLabelText("Review status"), "All");
    await user.click(screen.getByRole("button", { name: /slide-a\.png/ }));
    await user.selectOptions(screen.getByLabelText("Confirm or override suggested priority"), "Lower Priority");
    expect(screen.getByText("Reviewer notes (required for override)")).toBeVisible();

    await user.click(screen.getByRole("tab", { name: "Operational Dashboard" }));
    expect(screen.getByText("Estimated time avoided covers only unusable or skipped files, not time saved by ranking. 0s means this batch had no unusable or skipped files.")).toBeVisible();
    expect(screen.getByLabelText("Batch metrics")).toHaveTextContent("2valid images");
    expect(screen.getByLabelText("Batch metrics")).toHaveTextContent("0/2reviewed");
    expect(screen.getByLabelText("Batch metrics")).toHaveTextContent("60sestimated time avoided");
    expect(screen.getAllByText("Not measured")).toHaveLength(3);
    await user.click(screen.getByRole("tab", { name: "Model Evaluation & Limits" }));
    expect(screen.getByText("Experimental MHIST agreement-proxy head")).toBeVisible();
    expect(screen.getByText("These local methods are exploratory. The displayed evaluation, when present, is dataset-specific and is not clinical performance.")).toBeVisible();
    expect(screen.getByText("0.812")).toBeVisible();
  });

  it("saves and advances, applies groups, reopens reviews, and keeps API payloads unchanged", async () => {
    const user = userEvent.setup();
    const initial = withBatch();
    const reviewed = withBatch();
    reviewed.batch!.records[0].review = {
      ...reviewed.batch!.records[0].review,
      notes: "Reason for override",
      group_id: "case_01",
      priority: "Lower Priority",
      reviewed: true,
      reviewed_at_utc: "2026-07-22T20:00:00+00:00",
    };
    reviewed.batch!.metrics.reviewed_count = 1;
    reviewed.batch!.metrics.awaiting_count = 1;
    const grouped = structuredClone(reviewed);
    grouped.batch!.records[0].review.group_id = "case_02";
    grouped.batch!.records[1].review.group_id = "case_02";
    const reopened = structuredClone(grouped);
    reopened.batch!.records[0].review.reviewed = false;
    reopened.batch!.records[0].review.reviewed_at_utc = "";

    fetchMock
      .mockImplementationOnce(() => response(initial))
      .mockImplementationOnce(() => response(reviewed))
      .mockImplementationOnce(() => response(grouped))
      .mockImplementationOnce(() => response(reopened));
    render(<App />);
    await screen.findByRole("heading", { name: "Review queue" });

    await user.selectOptions(screen.getByLabelText("Confirm or override suggested priority"), "Lower Priority");
    await user.type(screen.getByLabelText(/Reviewer notes/), "Reason for override");
    await user.type(screen.getByLabelText("De-identified case/slide group ID (optional)"), "case_01");
    await user.click(screen.getByRole("button", { name: "Save & next unreviewed" }));
    await screen.findByRole("heading", { name: "slide-b.png" });
    expect(fetchMock.mock.calls[1][0]).toBe("/api/reviews/one");
    expect(JSON.parse(fetchMock.mock.calls[1][1]?.body as string)).toEqual({
      priority: "Lower Priority", notes: "Reason for override", group_id: "case_01",
    });

    await user.click(screen.getByRole("button", { name: /slide-a\.png/ }));
    const groupInput = screen.getByLabelText("De-identified case/slide group ID (optional)");
    await user.clear(groupInput);
    await user.type(groupInput, "case_02");
    await user.click(screen.getByRole("button", { name: "Apply group to source" }));
    await waitFor(() => expect(fetchMock.mock.calls[2][0]).toBe("/api/groups/one"));
    expect(JSON.parse(fetchMock.mock.calls[2][1]?.body as string)).toEqual({ group_id: "case_02" });
    await user.click(screen.getByRole("button", { name: /slide-b\.png/ }));
    expect(screen.getByLabelText("De-identified case/slide group ID (optional)")).toHaveValue("case_02");

    await user.click(screen.getByRole("button", { name: /slide-a\.png/ }));
    await user.click(screen.getByRole("button", { name: "Reopen review" }));
    await waitFor(() => expect(fetchMock.mock.calls[3][0]).toBe("/api/reviews/one/reopen"));
    expect(screen.getByRole("button", { name: "Save review" })).toBeVisible();
  });

  it("supports ordinary review saves and updates completed reviews", async () => {
    const user = userEvent.setup();
    const initial = withBatch();
    const saved = structuredClone(initial);
    saved.batch!.records[0].review.reviewed = true;
    saved.batch!.records[0].review.reviewed_at_utc = "2026-07-22T20:00:00+00:00";
    saved.batch!.metrics.reviewed_count = 1;
    saved.batch!.metrics.awaiting_count = 1;
    const updated = structuredClone(saved);
    updated.batch!.records[0].review.notes = "Updated note";

    fetchMock
      .mockImplementationOnce(() => response(initial))
      .mockImplementationOnce(() => response(saved))
      .mockImplementationOnce(() => response(updated));
    render(<App />);
    await screen.findByRole("heading", { name: "Review queue" });

    await user.click(screen.getByRole("button", { name: "Save review" }));
    expect(await screen.findByRole("button", { name: "Update review" })).toBeVisible();
    expect(JSON.parse(fetchMock.mock.calls[1][1]?.body as string)).toEqual({
      priority: "Review First", notes: "", group_id: "",
    });

    await user.type(screen.getByLabelText(/^Reviewer notes/), "Updated note");
    await user.click(screen.getByRole("button", { name: "Update review" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(3));
    expect(JSON.parse(fetchMock.mock.calls[2][1]?.body as string)).toEqual({
      priority: "Review First", notes: "Updated note", group_id: "",
    });
  });

  it("keeps export direct and confirms before resetting the current batch", async () => {
    const user = userEvent.setup();
    const initial = withBatch();
    fetchMock.mockImplementationOnce(() => response(initial)).mockImplementationOnce(() => response(preUpload()));
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
    render(<App />);
    await screen.findByRole("heading", { name: "Review queue" });
    expect(screen.getByRole("link", { name: "Export reviewed CSV" })).toHaveAttribute("href", "/api/export");
    await user.selectOptions(screen.getByLabelText("Feature provider"), "uni");
    await user.selectOptions(screen.getByLabelText("Batch tissue context"), "mhist_like_colorectal_polyp");
    const pendingFile = screen.getByLabelText("Choose one or more files") as HTMLInputElement;
    await user.upload(pendingFile, new File(["png"], "pending.png", { type: "image/png" }));
    await user.click(screen.getByRole("button", { name: "New batch" }));
    expect(confirm).toHaveBeenCalledWith("Discard this local batch and its unsaved session state?");
    await screen.findByRole("heading", { name: "Human review, organized" });
    expect(fetchMock.mock.calls[1][0]).toBe("/api/reset");
    expect(fetchMock.mock.calls[1][1]?.body).toBe("{}");
    expect(screen.getByLabelText("Feature provider")).toHaveValue("uni");
    expect(screen.getByLabelText("Batch tissue context")).toHaveValue("unknown_or_other");
    expect(pendingFile.files).toHaveLength(0);
  });

  it("shows an initial status error instead of leaving the loading view in place", async () => {
    fetchMock.mockImplementationOnce(() => Promise.resolve({
      ok: false,
      json: () => Promise.resolve({ error: "Status unavailable" }),
    } as Response));
    render(<App />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Status unavailable");
    expect(screen.getByText("Unable to load the local review workspace.")).toBeVisible();
  });

  it("shows API errors and auto-dismisses toast messages after five seconds", async () => {
    vi.useFakeTimers();
    fetchMock
      .mockImplementationOnce(() => response(preUpload()))
      .mockImplementationOnce(() => Promise.resolve({ ok: false, json: () => Promise.resolve({ error: "Upload rejected" }) } as Response));
    render(<App />);
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(screen.getByRole("heading", { name: "Human review, organized" })).toBeVisible();
    const uploadInput = screen.getByLabelText("Choose one or more files");
    fireEvent.change(uploadInput, {
      target: { files: [new File(["bad"], "bad.png", { type: "image/png" })] },
    });
    await act(async () => {
      fireEvent.submit(screen.getByRole("button", { name: "Process files" }).closest("form")!);
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(screen.getByRole("alert")).toHaveTextContent("Upload rejected");
    act(() => vi.advanceTimersByTime(5000));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
