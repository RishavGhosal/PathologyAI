import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { densityAwareDownsample, EmbeddingMap } from "./EmbeddingMap";
import type { EmbeddingProjectionPoint } from "../types";

function point(id: string, x: number, y: number, proxy_label: "HP" | "SSA" | null): EmbeddingProjectionPoint {
  return {
    id,
    name: id,
    x,
    y,
    proxy_label,
    suggested_priority: "Review First",
    embedding_model: "MahmoodLab/UNI",
    reviewed: false,
  };
}

const projection = {
  available: true,
  points: [
    point("hp-1", 0.1, 0.1, "HP"),
    point("hp-2", 0.11, 0.1, "HP"),
    point("ssa-1", 0.1, 0.1, "SSA"),
    point("unknown-1", 0.9, 0.9, null),
  ],
  method: "L2-normalized, PCA-initialized t-SNE",
  sample_count: 4,
  full_count: 4,
  error: null,
};

describe("densityAwareDownsample", () => {
  it("caps each cell, preserves class presence, and keeps full input untouched", () => {
    const points = [
      ...Array.from({ length: 6 }, (_, index) => point(`hp-${index}`, 0.1, 0.1, "HP")),
      ...Array.from({ length: 2 }, (_, index) => point(`ssa-${index}`, 0.1, 0.1, "SSA")),
      point("other", 0.9, 0.9, null),
    ];
    const rendered = densityAwareDownsample(points, { columns: 2, rows: 2, perCellCap: 4, totalCap: 5 });
    expect(rendered).toHaveLength(5);
    expect(new Set(rendered.filter((item) => item.x === 0.1).map((item) => item.proxy_label))).toEqual(new Set(["HP", "SSA"]));
    expect(points).toHaveLength(9);
  });

  it("is deterministic and allocates more capacity to denser cells", () => {
    const points = [
      ...Array.from({ length: 8 }, (_, index) => point(`dense-${index}`, 0.1, 0.1, null)),
      point("sparse", 0.9, 0.9, null),
    ];
    const options = { columns: 2, rows: 2, perCellCap: 8, totalCap: 5 };
    const first = densityAwareDownsample(points, options);
    const second = densityAwareDownsample(points, options);
    expect(first.map((item) => item.id)).toEqual(second.map((item) => item.id));
    expect(first.filter((item) => item.x === 0.1)).toHaveLength(5);
  });
});

describe("EmbeddingMap", () => {
  it("renders progress, palette controls, tooltips, and selection", () => {
    render(<EmbeddingMap projection={projection} />);

    expect(screen.getByRole("status", { name: /Embedding map progress/ })).toHaveTextContent("25%");
    expect(screen.getByLabelText("HP color")).toBeInTheDocument();
    expect(screen.getByText("Showing 4 of 4 points")).toBeInTheDocument();

    const firstPoint = screen.getByLabelText("hp-1, HP proxy, awaiting review");
    fireEvent.mouseEnter(firstPoint);
    expect(screen.getByText("hp-1")).toBeInTheDocument();
    fireEvent.click(firstPoint);
    expect(screen.getByText(/Selected:/)).toHaveTextContent("hp-1");

    fireEvent.change(screen.getByLabelText("Palette"), { target: { value: "ocean" } });
    expect(screen.getByLabelText("HP color")).toHaveValue("#0f766e");
    fireEvent.change(screen.getByLabelText("SSA color"), { target: { value: "#123456" } });
    expect(screen.getByLabelText("SSA color")).toHaveValue("#123456");
  });

  it("shows an explicit unavailable state", () => {
    render(<EmbeddingMap projection={{ available: false, points: [], method: null, sample_count: 0, full_count: 0, error: "At least eight UNI embeddings are required." }} />);
    expect(screen.getByText("At least eight UNI embeddings are required.")).toBeInTheDocument();
  });
});
