import type { LandingProjectionPoint } from "./embeddingProjection";

/** Keep both categorical tones visible while reducing the landing render load. */
export function sampleLandingPoints(points: LandingProjectionPoint[], cap: number): LandingProjectionPoint[] {
  if (points.length <= cap) return points;
  const groups = [
    points.filter((point) => point.tone === "teal"),
    points.filter((point) => point.tone === "sage"),
  ];
  const perGroup = Math.max(1, Math.floor(cap / groups.length));
  const sampled = groups.flatMap((group) => {
    const count = Math.min(perGroup, group.length);
    return Array.from({ length: count }, (_, index) => group[Math.round(index * (group.length - 1) / Math.max(1, count - 1))]);
  });
  return sampled.slice(0, cap);
}
