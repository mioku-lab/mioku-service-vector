import type { DenseHit, SearchHit, SparseHit } from "../types";

const RRF_K = 60;

export interface FuseOptions {
  denseWeight: number;
  fetchK: number;
  recencyHalfLifeMs: number;
  minScore: number;
}

export interface FusedEntry {
  hit: SearchHit;
  rrf: number;
  denseRank: number;
  sparseRank: number;
}

/**
 * 归一化加权融合 + RRF 兜底排序。
 *
 * BM25 分数只在同一索引内可比，所以稀疏分数单独归一化；
 * 稠密分数在本次候选内做 min-max。RRF 不参与主排序，
 * 只在融合分相同时提供确定性顺序。
 */
export function fuseHits(
  dense: DenseHit[],
  sparse: SparseHit[],
  options: FuseOptions,
): SearchHit[] {
  const denseNormalized = minMax(dense.map((hit) => hit.score));
  const sparseNormalized = minMax(sparse.map((hit) => hit.score));

  const entries = new Map<number, FusedEntry>();

  for (let i = 0; i < dense.length; i++) {
    const hit = dense[i];
    entries.set(hit.record.id, {
      hit: {
        docId: hit.record.docId,
        content: hit.record.content,
        metadata: hit.record.metadata,
        score: 0,
        denseScore: hit.score,
        createdAt: hit.record.createdAt,
        updatedAt: hit.record.updatedAt,
      },
      rrf: 1 / (RRF_K + hit.rank),
      denseRank: hit.rank,
      sparseRank: Number.POSITIVE_INFINITY,
    });
    const entry = entries.get(hit.record.id);
    if (entry) entry.hit.score = options.denseWeight * denseNormalized[i];
  }

  for (let i = 0; i < sparse.length; i++) {
    const hit = sparse[i];
    const existing = entries.get(hit.record.id);
    if (existing) {
      existing.hit.sparseScore = hit.score;
      existing.hit.score += (1 - options.denseWeight) * sparseNormalized[i];
      existing.sparseRank = hit.rank;
      existing.rrf += 1 / (RRF_K + hit.rank);
      continue;
    }
    entries.set(hit.record.id, {
      hit: {
        docId: hit.record.docId,
        content: hit.record.content,
        metadata: hit.record.metadata,
        score: (1 - options.denseWeight) * sparseNormalized[i],
        sparseScore: hit.score,
        createdAt: hit.record.createdAt,
        updatedAt: hit.record.updatedAt,
      },
      rrf: 1 / (RRF_K + hit.rank),
      denseRank: Number.POSITIVE_INFINITY,
      sparseRank: hit.rank,
    });
  }

  const now = Date.now();
  const fused = [...entries.values()]
    .map((entry) => ({
      ...entry,
      hit: {
        ...entry.hit,
        score:
          entry.hit.score *
          recencyFactor(now - entry.hit.updatedAt, options.recencyHalfLifeMs),
      },
    }))
    .filter((entry) => entry.hit.score >= options.minScore)
    .sort((a, b) => {
      if (b.hit.score !== a.hit.score) return b.hit.score - a.hit.score;
      if (b.rrf !== a.rrf) return b.rrf - a.rrf;
      if (a.denseRank !== b.denseRank) return a.denseRank - b.denseRank;
      if (a.sparseRank !== b.sparseRank) return a.sparseRank - b.sparseRank;
      return a.hit.docId.localeCompare(b.hit.docId);
    });

  return fused.slice(0, Math.max(1, options.fetchK)).map((entry) => entry.hit);
}

function recencyFactor(ageMs: number, halfLifeMs: number): number {
  if (halfLifeMs <= 0 || ageMs <= halfLifeMs) return 1;
  return halfLifeMs / (halfLifeMs + (ageMs - halfLifeMs));
}

function minMax(scores: number[]): number[] {
  if (scores.length === 0) return [];
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (const score of scores) {
    if (score < min) min = score;
    if (score > max) max = score;
  }
  const range = max - min;
  if (range === 0) return scores.map(() => 1);
  return scores.map((score) => (score - min) / range);
}
