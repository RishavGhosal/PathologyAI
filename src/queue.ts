import { PRIORITIES, type ImageRecord, type Priority } from "./types";

export function effectivePriority(record: ImageRecord): Priority {
  return record.review.priority || record.triage.suggested_priority;
}

export function compareQueueRecords(left: ImageRecord, right: ImageRecord): number {
  const priorityDifference = PRIORITIES.indexOf(effectivePriority(left))
    - PRIORITIES.indexOf(effectivePriority(right));
  if (priorityDifference) return priorityDifference;

  const leftScore = left.triage.review_first_score ?? -1;
  const rightScore = right.triage.review_first_score ?? -1;
  if (leftScore !== rightScore) return rightScore - leftScore;

  return left.name.localeCompare(right.name, undefined, { sensitivity: "base" });
}
