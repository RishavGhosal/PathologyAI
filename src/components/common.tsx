import type { ReactNode } from "react";
import type { OperationalMetrics, Priority } from "../types";

export type ThemeName = "dark" | "light" | "sage";
export type AccentName = "cyan" | "sage" | "violet";

export function Header({ onReset, theme, onThemeChange, accent, onAccentChange }: {
  onReset: () => void;
  theme: ThemeName;
  onThemeChange: (theme: ThemeName) => void;
  accent: AccentName;
  onAccentChange: (accent: AccentName) => void;
}) {
  return (
    <header className="site-header">
      <div className="brand">
        <strong>PathologyAI</strong>
        <small>Research/Education Prototype · Human Review Required</small>
      </div>
      <div className="header-actions">
        <label className="theme-control">
          <span>Theme</span>
          <select aria-label="Color theme" value={theme} onChange={(event) => onThemeChange(event.target.value as ThemeName)}>
            <option value="dark">Dark</option>
            <option value="light">Light</option>
            <option value="sage">Sage</option>
          </select>
        </label>
        <label className="theme-control">
          <span>Accent</span>
          <select aria-label="Accent color" value={accent} onChange={(event) => onAccentChange(event.target.value as AccentName)}>
            <option value="cyan">Cyan</option>
            <option value="sage">Sage</option>
            <option value="violet">Violet</option>
          </select>
        </label>
        <button type="button" onClick={onReset}>New batch</button>
        <a className="button" href="/api/export">Export reviewed CSV</a>
      </div>
    </header>
  );
}

export function PriorityChip({ priority }: { priority: Priority }) {
  return <span className={`chip ${slug(priority)}`}>{priority}</span>;
}

export function MetricsCards({ metrics }: { metrics: OperationalMetrics }) {
  const cards: Array<[string | number, string]> = [
    [metrics.total_images, "valid images"],
    [`${metrics.reviewed_count}/${metrics.total_images}`, "reviewed"],
    [metrics.awaiting_count, "awaiting review"],
    [metrics.quality_pass_count, "passed quality"],
    [metrics.skipped_count, "skipped inputs"],
    [`${Math.round(metrics.estimated_time_avoided_seconds)}s`, "estimated time avoided"],
  ];
  return (
    <div className="metrics" aria-label="Batch metrics">
      {cards.map(([value, label]) => (
        <div className="metric" key={label}>
          <b>{value}</b>
          <span>{label}</span>
        </div>
      ))}
    </div>
  );
}

export function DataTable({
  headers,
  rows,
  empty = "None recorded.",
}: {
  headers: string[];
  rows: ReactNode[][];
  empty?: string;
}) {
  if (!rows.length) return <p className="empty compact">{empty}</p>;
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>{headers.map((header) => <th key={header}>{header}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {row.map((cell, cellIndex) => <td key={cellIndex}>{cell}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function TextList({ values, empty }: { values: string[]; empty: string }) {
  return values.length ? (
    <ul className="list">{values.map((value) => <li key={value}>{value}</li>)}</ul>
  ) : (
    <p className="empty compact">{empty}</p>
  );
}

export function Toast({ kind, message }: { kind: "success" | "error" | "info"; message: string }) {
  return <div className={`toast ${kind}`} role={kind === "error" ? "alert" : "status"}>{message}</div>;
}

function slug(value: string) {
  return value.toLowerCase().replaceAll(" ", "-");
}
