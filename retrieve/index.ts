import type {
  CollectionInfo,
  DenseHit,
  SearchHit,
  SearchQuery,
  SparseHit,
} from "../types";
import type { Embedder } from "../embed";
import { compileFilter } from "../store/filter";
import type { RecordStore } from "../store/records";
import type { VectorStore } from "../store/schema";
import { denseSearch } from "./dense";
import { fuseHits } from "./fusion";
import { sparseSearch } from "./sparse";

export interface RetrieveDeps {
  store: VectorStore;
  records: RecordStore;
  embedder: Embedder;
  defaults: {
    denseK: number;
    sparseK: number;
    denseWeight: number;
    candidateLimit: number;
    recencyHalfLifeMs: number;
  };
  rerank?: Reranker;
  logger: {
    debug: (message: string) => void;
    warn: (message: string) => void;
  };
}

export interface Reranker {
  rerank(query: string, documents: string[]): Promise<number[] | undefined>;
}

/** 检索漏斗：稠密 + 稀疏 → 归一化融合 → 可选重排 */
export async function runSearch(
  deps: RetrieveDeps,
  query: SearchQuery,
  collection: CollectionInfo,
): Promise<SearchHit[]> {
  const mode = query.mode ?? "hybrid";
  const topK = clampInt(query.topK, 5, 1, 200);
  const fetchK = clampInt(query.fetchK, Math.max(topK * 4, 20), topK, 500);
  const denseK = clampInt(query.denseK, deps.defaults.denseK, 1, 2000);
  const sparseK = clampInt(query.sparseK, deps.defaults.sparseK, 1, 2000);
  const denseWeight = clampFloat(
    query.denseWeight,
    deps.defaults.denseWeight,
    0,
    1,
  );
  const recencyHalfLifeMs = Math.max(
    0,
    query.recencyHalfLifeMs ?? deps.defaults.recencyHalfLifeMs,
  );

  const compiled = compileFilter(query.filter);
  const whereSql = compiled.sql;
  const params = compiled.params;

  const hasVector = collection.dim > 0 && deps.embedder.available;
  const wantDense = mode !== "sparse" && hasVector;
  const wantSparse = mode !== "dense" && deps.store.ftsAvailable;

  if (mode !== "sparse" && !hasVector) {
    deps.logger.debug(
      `[vector] ${collection.name} 无可用向量化，降级为关键词检索`,
    );
  }

  let denseHits: DenseHit[] = [];
  let sparseHits: SparseHit[] = [];

  const tasks: Array<Promise<void>> = [];
  if (wantDense) {
    tasks.push(
      (async () => {
        const [vector] = await deps.embedder.embed([query.query]);
        denseHits = denseSearch(deps.records, {
          collection: collection.name,
          vector,
          whereSql,
          params,
          limit: denseK,
          candidateLimit: deps.defaults.candidateLimit,
        });
      })(),
    );
  }
  if (wantSparse) {
    tasks.push(
      (async () => {
        sparseHits = sparseSearch(deps.store, deps.records, {
          collection: collection.name,
          text: query.query,
          whereSql,
          params,
          limit: sparseK,
        });
      })(),
    );
  }
  await Promise.all(tasks);

  const weight =
    denseHits.length === 0 ? 0 : sparseHits.length === 0 ? 1 : denseWeight;

  const fused = fuseHits(denseHits, sparseHits, {
    denseWeight: weight,
    fetchK,
    recencyHalfLifeMs,
    minScore: Number.isFinite(query.minScore) ? Number(query.minScore) : 0,
  });

  const reranked = await applyRerank(deps, query, fused);
  return reranked.slice(0, topK);
}

async function applyRerank(
  deps: RetrieveDeps,
  query: SearchQuery,
  hits: SearchHit[],
): Promise<SearchHit[]> {
  if (!deps.rerank || hits.length === 0) return hits;
  try {
    const scores = await deps.rerank.rerank(
      query.query,
      hits.map((hit) => hit.content),
    );
    if (!scores || scores.length !== hits.length) return hits;
    return hits
      .map((hit, index) => ({ ...hit, score: scores[index] }))
      .sort((a, b) => b.score - a.score);
  } catch (error) {
    deps.logger.warn(`[vector] 重排失败，退回融合结果: ${error}`);
    return hits;
  }
}

function clampInt(
  value: number | undefined,
  fallback: number,
  min: number,
  max: number,
): number {
  const num = Number(value);
  if (!Number.isFinite(num)) return fallback;
  return Math.min(max, Math.max(min, Math.trunc(num)));
}

function clampFloat(
  value: number | undefined,
  fallback: number,
  min: number,
  max: number,
): number {
  const num = Number(value);
  if (!Number.isFinite(num)) return fallback;
  return Math.min(max, Math.max(min, num));
}
