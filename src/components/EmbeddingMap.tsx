import { useEffect, useMemo, useState } from "react";
import type { EmbeddingProjection, EmbeddingProjectionPoint, EmbeddingProxyLabel } from "../types";

export const EMBEDDING_RENDER_POINT_CAP = 2500;
export const EMBEDDING_GRID_COLUMNS = 48;
export const EMBEDDING_GRID_ROWS = 36;
export const EMBEDDING_POINTS_PER_CELL = 8;

type Palette = Record<"HP" | "SSA" | "unlabeled", string>;
type PaletteName = "colorblind" | "ocean" | "warm" | "violet";
type ProgressStage = "loading" | "layout" | "rendering" | "ready";

export const EMBEDDING_PALETTES: Record<PaletteName, Palette> = {
  colorblind: { HP: "#0072B2", SSA: "#D55E00", unlabeled: "#94A3B8" },
  ocean: { HP: "#0F766E", SSA: "#38BDF8", unlabeled: "#94A3B8" },
  warm: { HP: "#B45309", SSA: "#BE185D", unlabeled: "#94A3B8" },
  violet: { HP: "#7C3AED", SSA: "#DB2777", unlabeled: "#94A3B8" },
};

const PROGRESS: Record<ProgressStage, { percent: number; label: string }> = {
  loading: { percent: 25, label: "Loading embedding data" },
  layout: { percent: 55, label: "Preparing layout" },
  rendering: { percent: 85, label: "Rendering map" },
  ready: { percent: 100, label: "Ready" },
};

function pointLabel(point: EmbeddingProjectionPoint): "HP" | "SSA" | "unlabeled" {
  return point.proxy_label ?? "unlabeled";
}

function stratifyCell(points: EmbeddingProjectionPoint[], cap: number): EmbeddingProjectionPoint[] {
  const groups = new Map<string, EmbeddingProjectionPoint[]>();
  for (const point of points) {
    const label = pointLabel(point);
    const group = groups.get(label) ?? [];
    group.push(point);
    groups.set(label, group);
  }
  const orderedGroups = [...groups.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([, group]) => group.sort((left, right) => left.id.localeCompare(right.id)));
  const selected: EmbeddingProjectionPoint[] = [];
  let index = 0;
  while (selected.length < cap && orderedGroups.some((group) => index < group.length)) {
    for (const group of orderedGroups) {
      if (index < group.length && selected.length < cap) selected.push(group[index]);
    }
    index += 1;
  }
  return selected;
}

export function densityAwareDownsample(
  points: EmbeddingProjectionPoint[],
  options: { columns?: number; rows?: number; perCellCap?: number; totalCap?: number } = {},
): EmbeddingProjectionPoint[] {
  const columns = options.columns ?? EMBEDDING_GRID_COLUMNS;
  const rows = options.rows ?? EMBEDDING_GRID_ROWS;
  const perCellCap = options.perCellCap ?? EMBEDDING_POINTS_PER_CELL;
  const totalCap = options.totalCap ?? EMBEDDING_RENDER_POINT_CAP;
  const cells = new Map<string, EmbeddingProjectionPoint[]>();

  for (const point of points) {
    const column = Math.max(0, Math.min(columns - 1, Math.floor(point.x * columns)));
    const row = Math.max(0, Math.min(rows - 1, Math.floor(point.y * rows)));
    const key = `${row}:${column}`;
    const cell = cells.get(key) ?? [];
    cell.push(point);
    cells.set(key, cell);
  }

  const candidates = [...cells.entries()]
    .map(([key, cell]) => ({ key, points: stratifyCell(cell, perCellCap), cursor: 0 }))
    .sort((left, right) => right.points.length - left.points.length || left.key.localeCompare(right.key));
  const selected: EmbeddingProjectionPoint[] = [];
  while (selected.length < totalCap) {
    let best: typeof candidates[number] | undefined;
    for (const candidate of candidates) {
      if (candidate.cursor >= candidate.points.length) continue;
      if (!best || candidate.points.length - candidate.cursor > best.points.length - best.cursor || (candidate.points.length - candidate.cursor === best.points.length - best.cursor && candidate.key < best.key)) {
        best = candidate;
      }
    }
    if (!best) break;
    selected.push(best.points[best.cursor]);
    best.cursor += 1;
  }
  return selected;
}

function paletteKey(label: EmbeddingProxyLabel): keyof Palette {
  return label ?? "unlabeled";
}

export function EmbeddingMap({ projection }: { projection?: EmbeddingProjection }) {
  const [paletteName, setPaletteName] = useState<PaletteName>("colorblind");
  const [palette, setPalette] = useState<Palette>(EMBEDDING_PALETTES.colorblind);
  const [stage, setStage] = useState<ProgressStage>("loading");
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const renderedPoints = useMemo(
    () => densityAwareDownsample(projection?.points ?? []),
    [projection?.points],
  );
  const hovered = projection?.points.find((point) => point.id === hoveredId) ?? null;
  const selected = projection?.points.find((point) => point.id === selectedId) ?? null;
  const progress = PROGRESS[stage];

  useEffect(() => {
    setStage("loading");
    setHoveredId(null);
    setSelectedId(null);
    const first = window.requestAnimationFrame(() => {
      setStage("layout");
      const second = window.requestAnimationFrame(() => {
        setStage("rendering");
        const third = window.requestAnimationFrame(() => setStage("ready"));
        return () => window.cancelAnimationFrame(third);
      });
      return () => window.cancelAnimationFrame(second);
    });
    return () => window.cancelAnimationFrame(first);
  }, [projection]);

  function updateCustomColor(label: keyof Palette, color: string) {
    setPaletteName("colorblind");
    setPalette((current) => ({ ...current, [label]: color }));
  }

  function selectPalette(name: PaletteName) {
    setPaletteName(name);
    setPalette(EMBEDDING_PALETTES[name]);
  }

  return (
    <section className="embedding-map-panel" aria-labelledby="embedding-map-title">
      <div className="embedding-map-heading">
        <div>
          <p className="eyebrow">UNI representation space</p>
          <h3 id="embedding-map-title">2D t-SNE embedding map</h3>
          <p className="embedding-map-note">The map is exploratory. Similar visual representations appear nearer to one another.</p>
        </div>
        <div className="embedding-map-controls">
          <label htmlFor="embedding-palette">Palette</label>
          <select id="embedding-palette" value={paletteName} onChange={(event) => selectPalette(event.target.value as PaletteName)}>
            <option value="colorblind">Colorblind safe</option>
            <option value="ocean">Ocean</option>
            <option value="warm">Warm</option>
            <option value="violet">Violet</option>
          </select>
          <div className="embedding-swatches" aria-label="Custom class colors">
            {(["HP", "SSA", "unlabeled"] as const).map((label) => (
              <label key={label} className="embedding-swatch">
                <span>{label === "unlabeled" ? "Unlabeled" : `${label} proxy`}</span>
                <input type="color" aria-label={`${label} color`} value={palette[label]} onChange={(event) => updateCustomColor(label, event.target.value)} />
              </label>
            ))}
          </div>
        </div>
      </div>

      <div className="embedding-progress" role="status" aria-label={`Embedding map progress: ${progress.percent}%`}>
        <div className="embedding-progress-label"><span>{progress.label}</span><strong>{progress.percent}%</strong></div>
        <div className="embedding-progress-track"><span style={{ width: `${progress.percent}%` }} /></div>
      </div>

      {!projection?.available ? (
        <p className="empty compact embedding-map-empty">{projection?.error ?? "UNI embedding data is not available for this batch."}</p>
      ) : (
        <>
          <div className="embedding-map-meta">
            <span>Showing {renderedPoints.length.toLocaleString()} of {projection.full_count.toLocaleString()} points</span>
            <span>Grid {EMBEDDING_GRID_COLUMNS}×{EMBEDDING_GRID_ROWS} · {EMBEDDING_POINTS_PER_CELL} per cell max</span>
          </div>
          <div className="embedding-map-canvas">
            <svg viewBox="0 0 760 440" role="img" aria-label="UNI t-SNE embedding map. Select a point to inspect its record.">
              <rect className="embedding-map-background" x="0" y="0" width="760" height="440" rx="10" />
              <line className="embedding-map-axis" x1="48" y1="400" x2="730" y2="400" />
              <line className="embedding-map-axis" x1="48" y1="28" x2="48" y2="400" />
              {renderedPoints.map((point) => {
                const x = 56 + point.x * 666;
                const y = 34 + (1 - point.y) * 350;
                const isSelected = point.id === selectedId;
                return (
                  <circle
                    key={point.id}
                    className="embedding-map-point"
                    cx={x}
                    cy={y}
                    r={isSelected ? 6.5 : 4.2}
                    fill={palette[paletteKey(point.proxy_label)]}
                    opacity={point.id === hoveredId || isSelected ? 1 : 0.78}
                    stroke={isSelected ? "var(--text)" : "var(--surface)"}
                    strokeWidth={isSelected ? 2.2 : 1}
                    tabIndex={0}
                    onMouseEnter={() => setHoveredId(point.id)}
                    onMouseLeave={() => setHoveredId(null)}
                    onFocus={() => setHoveredId(point.id)}
                    onBlur={() => setHoveredId(null)}
                    onClick={() => setSelectedId(point.id)}
                    onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); setSelectedId(point.id); } }}
                    aria-label={`${point.name}, ${point.proxy_label ?? "unlabeled"} proxy, ${point.reviewed ? "reviewed" : "awaiting review"}`}
                  />
                );
              })}
              <text className="embedding-map-axis-label" x="48" y="425">representation 1</text>
              <text className="embedding-map-axis-label" x="10" y="40" transform="rotate(-90 10 40)">representation 2</text>
              {hovered && <MapTooltip point={hovered} x={56 + hovered.x * 666} y={34 + (1 - hovered.y) * 350} />}
            </svg>
          </div>
          <div className="embedding-map-legend" aria-label="Embedding map legend">
            {(["HP", "SSA", "unlabeled"] as const).map((label) => <span key={label}><i style={{ background: palette[label] }} />{label === "unlabeled" ? "Unlabeled" : `${label} proxy`}</span>)}
          </div>
          {selected && <p className="embedding-selection" role="status">Selected: <strong>{selected.name}</strong> · {selected.proxy_label ?? "Unlabeled"} proxy · {selected.reviewed ? "Reviewed" : "Awaiting review"}</p>}
        </>
      )}
    </section>
  );
}

function MapTooltip({ point, x, y }: { point: EmbeddingProjectionPoint; x: number; y: number }) {
  const left = x > 570 ? x - 190 : x + 12;
  const top = Math.max(8, Math.min(350, y - 50));
  return <g className="embedding-map-tooltip" pointerEvents="none"><rect x={left} y={top} width="178" height="62" rx="6" /><text x={left + 10} y={top + 20}>{point.name.slice(0, 24)}</text><text x={left + 10} y={top + 38}>{point.proxy_label ?? "Unlabeled"} proxy</text><text x={left + 10} y={top + 54}>{point.reviewed ? "Reviewed" : "Awaiting review"}</text></g>;
}
