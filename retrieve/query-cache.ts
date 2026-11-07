import type { SearchHit, SearchQuery } from "../types";

interface Entry {
  collection: string;
  hits: SearchHit[];
  expiresAt: number;
}

export interface QueryCacheStats {
  hits: number;
  misses: number;
  entries: number;
  evictions: number;
  invalidations: number;
}

const SEPARATOR = "\u0000";

/** 稳定序列化：对象键排序，保证同一过滤条件得到同一指纹 */
function canonical(value: unknown): string {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value) ?? "null";
  }
  if (Array.isArray(value)) {
    return `[${value.map((item) => canonical(item)).join(",")}]`;
  }
  const entries = Object.entries(value as Record<string, unknown>).sort(
    ([a], [b]) => (a < b ? -1 : a > b ? 1 : 0),
  );
  return `{${entries
    .map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`)
    .join(",")}}`;
}

/** 查询指纹：只包含真正影响结果的参数 */
export function queryCacheKey(query: SearchQuery, mode: string): string {
  return [
    query.collection,
    mode,
    query.query,
    query.topK ?? "",
    query.fetchK ?? "",
    query.denseK ?? "",
    query.sparseK ?? "",
    query.denseWeight ?? "",
    query.recencyHalfLifeMs ?? "",
    query.minScore ?? "",
    canonical(query.filter ?? null),
  ].join(SEPARATOR);
}

function cloneHits(hits: SearchHit[]): SearchHit[] {
  return hits.map((hit) => ({ ...hit }));
}

export interface QueryCacheOptions {
  enabled: boolean;
  ttlMs: number;
  maxEntries: number;
}

/**
 * 检索结果短缓存
 */
export class QueryCache {
  private readonly entries = new Map<string, Entry>();
  private readonly pending = new Map<string, Promise<SearchHit[]>>();
  /** 失效代数：写入期间算出来的结果不回填，避免缓存旧值 */
  private generation = 0;
  private hits = 0;
  private misses = 0;
  private evictions = 0;
  private invalidations = 0;

  constructor(private readonly options: QueryCacheOptions) {}

  get enabled(): boolean {
    return (
      this.options.enabled &&
      this.options.ttlMs > 0 &&
      this.options.maxEntries > 0
    );
  }

  stats(): QueryCacheStats {
    return {
      hits: this.hits,
      misses: this.misses,
      entries: this.entries.size,
      evictions: this.evictions,
      invalidations: this.invalidations,
    };
  }

  /** 不传 collection 时清空全部 */
  invalidate(collection?: string): void {
    this.generation += 1;
    this.invalidations += 1;
    if (collection === undefined) {
      this.entries.clear();
      return;
    }
    for (const [key, entry] of this.entries) {
      if (entry.collection === collection) this.entries.delete(key);
    }
  }

  clear(): void {
    this.entries.clear();
    this.pending.clear();
    this.generation += 1;
  }

  async memo(
    collection: string,
    key: string,
    compute: () => Promise<SearchHit[]>,
  ): Promise<SearchHit[]> {
    if (!this.enabled) return compute();

    const entry = this.entries.get(key);
    if (entry && entry.expiresAt > Date.now()) {
      this.hits += 1;
      // 命中即刷新 LRU 顺序
      this.entries.delete(key);
      this.entries.set(key, entry);
      return cloneHits(entry.hits);
    }
    if (entry) this.entries.delete(key);

    const inflight = this.pending.get(key);
    if (inflight) {
      this.hits += 1;
      return inflight.then(cloneHits);
    }

    this.misses += 1;
    const generation = this.generation;
    const promise = compute()
      .then((hits) => {
        if (generation === this.generation) this.store(key, collection, hits);
        return hits;
      })
      .finally(() => {
        this.pending.delete(key);
      });
    this.pending.set(key, promise);
    return promise.then(cloneHits);
  }

  private store(key: string, collection: string, hits: SearchHit[]): void {
    this.entries.set(key, {
      collection,
      hits: cloneHits(hits),
      expiresAt: Date.now() + this.options.ttlMs,
    });
    while (this.entries.size > this.options.maxEntries) {
      const oldest = this.entries.keys().next().value;
      if (oldest === undefined) break;
      this.entries.delete(oldest);
      this.evictions += 1;
    }
  }
}
