import type { DenseHit, StoredRecord } from "../types";
import type { RecordStore } from "../store/records";

export function denseSearch(
  records: RecordStore,
  query: {
    collection: string;
    vector: Float32Array;
    whereSql: string;
    params: Array<string | number>;
    limit: number;
    candidateLimit: number;
  },
): DenseHit[] {
  const candidates = records.listVectors(
    query.collection,
    query.whereSql,
    query.params,
    query.candidateLimit,
  );
  if (candidates.length === 0) return [];

  const dim = query.vector.length;
  const scored: Array<{ id: number; docId: string; score: number }> = [];
  for (const candidate of candidates) {
    if (candidate.vector.length !== dim) continue;
    let sum = 0;
    const vector = candidate.vector;
    for (let i = 0; i < dim; i++) sum += vector[i] * query.vector[i];
    scored.push({ id: candidate.id, docId: candidate.docId, score: sum });
  }
  if (scored.length === 0) return [];

  // 同分时按 docId 定序，保证结果可复现
  scored.sort((a, b) => b.score - a.score || a.docId.localeCompare(b.docId));
  const top = scored.slice(0, Math.max(1, query.limit));
  const recordMap = records.getRecordsByIds(top.map((item) => item.id));

  const hits: DenseHit[] = [];
  for (let rank = 0; rank < top.length; rank++) {
    const record: StoredRecord | undefined = recordMap.get(top[rank].id);
    if (!record) continue;
    hits.push({ record, score: top[rank].score, rank: rank + 1 });
  }
  return hits;
}
