import { Embedder, type TextEmbeddingModel } from "./embed";
import { RecordStore } from "./store/records";
import { VectorStore } from "./store/schema";
import { compileFilter } from "./store/filter";
import { runSearch, type Reranker } from "./retrieve";
import { QueryCache, queryCacheKey } from "./retrieve/query-cache";
import { BootstrapStateStore, type BootstrapState } from "./bootstrap/state";
import { ensureModelFiles, modelDirectory } from "./bootstrap/model";
import { toIndexText } from "./utils/tokenize";
import { embedHash } from "./utils/hash";
import { findEmbeddingModel, RETRIEVAL_DEFAULTS, STORAGE_DEFAULTS } from "./config";
import type { VectorConfig } from "./config";
import type {
  BatchOptions,
  BatchUpsertResult,
  CollectionInfo,
  CollectionOptions,
  CollectionStats,
  MetadataFilter,
  ReindexOptions,
  ReindexResult,
  SearchHit,
  SearchQuery,
  StoredRecord,
  UpsertInput,
  UpsertResult,
  VectorCacheStats,
  VectorServiceApi,
  VectorMetadata,
} from "./types";

export interface VectorServiceDeps {
  dbPath: string;
  modelDir: string;
  config: VectorConfig;
  logger: {
    info: (message: string) => void;
    debug: (message: string) => void;
    warn: (message: string) => void;
    error: (message: string) => void;
  };
  rerank?: Reranker;
  /** 注入推理后端，测试用；缺省时用本地 ONNX 模型 */
  model?: TextEmbeddingModel;
}

export class VectorServiceImpl implements VectorServiceApi {
  private readonly store: VectorStore;
  private readonly records: RecordStore;
  private readonly embedder: Embedder;
  private readonly deps: VectorServiceDeps;
  private readonly statusStore: BootstrapStateStore;
  private readonly cache: QueryCache;

  constructor(deps: VectorServiceDeps) {
    this.deps = deps;
    this.store = new VectorStore(deps.dbPath);
    this.records = new RecordStore(this.store);

    const model = findEmbeddingModel(deps.config.model);
    const repo = model?.repo ?? "";
    this.embedder = new Embedder({
      modelRoot: deps.modelDir,
      repo,
      modelKey: deps.config.model,
      // 本地推理没有网络限流，批给大使每次调用次数降到最低
      batchSize: 64,
      cacheSize: 4096,
      ...(deps.model && { model: deps.model }),
      logger: deps.logger,
    });

    this.cache = new QueryCache({
      enabled: deps.config.cacheEnabled,
      ttlMs: deps.config.cacheTtlMs,
      maxEntries: deps.config.cacheMaxEntries,
    });

    this.statusStore = new BootstrapStateStore(
      deps.config.enabled
        ? { phase: "pending", message: "等待加载本地向量模型", ready: false }
        : {
            phase: "disabled",
            message: "已在配置中关闭向量化，仅关键词检索",
            ready: false,
          },
    );
  }

  start(): void {
    if (!this.deps.config.enabled) return;
    void this.prepare().catch((error) => {
      this.statusStore.set({
        phase: "failed",
        message: "本地向量模型准备失败，降级为关键词检索",
        ready: false,
        reason: String(error instanceof Error ? error.message : error),
      });
    });
  }

  private async prepare(): Promise<void> {
    const model = findEmbeddingModel(this.deps.config.model);
    if (!model) {
      this.statusStore.set({
        phase: "failed",
        message: `未知的向量模型 ${this.deps.config.model}`,
        ready: false,
        reason: "配置中的模型不在可选列表中",
      });
      return;
    }

    // 注入推理后端（测试替身）不需要模型文件
    if (this.deps.model) {
      const ready = await this.embedder.load();
      this.statusStore.set(
        ready
          ? {
              phase: "ready",
              message: `本地向量模型 ${model.label} 已就绪`,
              ready: true,
            }
          : {
              phase: "failed",
              message: "本地向量模型加载失败，降级为关键词检索",
              ready: false,
              reason: this.embedder.error,
            },
      );
      return;
    }

    if (!this.deps.config.autoDownload) {
      const dir = modelDirectory(this.deps.modelDir, model.repo);
      const ready = await this.embedder.load();
      this.statusStore.set(
        ready
          ? {
              phase: "ready",
              message: `本地向量模型 ${model.label} 已就绪`,
              ready: true,
            }
          : {
              phase: "failed",
              message: `已关闭自动下载，且 ${dir} 下没有可用模型`,
              ready: false,
              reason: "关闭了自动下载，请手动放置模型文件或打开该开关",
            },
      );
      return;
    }

    this.statusStore.set({
      phase: "downloading",
      message: `正在准备本地向量模型 ${model.label}`,
      ready: false,
    });
    await ensureModelFiles({
      model,
      modelRoot: this.deps.modelDir,
      mirror: this.deps.config.hfMirror,
      logger: this.deps.logger,
      onProgress: (message) => {
        this.statusStore.set({
          phase: "downloading",
          message,
          ready: false,
        });
      },
    });

    const ok = await this.embedder.load();
    if (ok) {
      this.markReady(model.label);
      return;
    }
    this.statusStore.set({
      phase: "failed",
      message: "本地向量模型加载失败，降级为关键词检索",
      ready: false,
      reason: this.embedder.error,
    });
  }

  private markReady(label: string): void {
    const dim = this.embedder.knownDim;
    this.statusStore.set({
      phase: "ready",
      message: `本地向量模型 ${label} 已就绪${dim ? `（${dim} 维）` : ""}`,
      ready: true,
    });
  }

  status(): BootstrapState {
    return this.statusStore.current;
  }

  onStatus(listener: (state: BootstrapState) => void): () => void {
    return this.statusStore.subscribe(listener);
  }

  describe(): string {
    const state = this.statusStore.current;
    const sparse = this.store.ftsAvailable ? "FTS5" : "不可用";
    return `向量化=${state.phase}${state.ready ? "" : `(${state.message})`}, 稀疏=${sparse}`;
  }

  async ensureCollection(
    name: string,
    options: CollectionOptions = {},
  ): Promise<CollectionInfo> {
    const key = normalizeName(name);
    const existingRow = this.records.getCollectionRow(key);
    const requestedDim = Math.max(0, Math.trunc(options.dim ?? 0));

    if (existingRow) {
      const info = this.records.toInfo(existingRow);
      this.assertCompatible(info, options, requestedDim);
      const patch: Parameters<RecordStore["updateCollection"]>[1] = {};
      let changed = false;
      if (options.metric !== undefined && options.metric !== info.metric) {
        patch.metric = options.metric;
        changed = true;
      }
      if (
        options.maxContentChars !== undefined &&
        options.maxContentChars !== info.maxContentChars
      ) {
        patch.maxContentChars = options.maxContentChars;
        changed = true;
      }
      if (
        options.maxRecords !== undefined &&
        options.maxRecords !== info.maxRecords
      ) {
        patch.maxRecords = options.maxRecords;
        changed = true;
      }
      if (options.metadata !== undefined) {
        patch.metadata = { ...info.metadata, ...options.metadata };
        changed = true;
      }
      const model = resolveEmbedModel(options, this.embedder.modelKey);
      if (model && model !== info.embedModel) {
        this.deps.logger.warn(
          `[vector] ${key} 的向量模型由 ${info.embedModel || "(空)"} 变为 ${model}，旧向量已不可比，需要 reindex`,
        );
        patch.embedModel = model;
        changed = true;
      }
      if (!changed) return info;
      const updated = this.records.updateCollection(key, patch);
      this.cache.invalidate(key);
      return updated;
    }

    // 模型可能还在加载，维度先留 0，拿到向量后回填
    const dim = requestedDim > 0 ? requestedDim : (this.embedder.knownDim ?? 0);
    const created = this.store.transaction(() =>
      this.records.insertCollection(key, {
        ...options,
        dim,
        embedModel: resolveEmbedModel(options, this.embedder.modelKey),
        metric: options.metric ?? this.deps.config.metric,
        maxContentChars: options.maxContentChars ?? STORAGE_DEFAULTS.maxContentChars,
        maxRecords: options.maxRecords ?? STORAGE_DEFAULTS.maxRecords,
      }),
    );
    this.store.ftsTableName(key);
    this.deps.logger.debug(
      `[vector] 新建 collection ${key} (dim=${created.dim}, metric=${created.metric})`,
    );
    return created;
  }

  getCollection(name: string): CollectionInfo | undefined {
    const row = this.records.getCollectionRow(normalizeName(name));
    return row ? this.records.toInfo(row) : undefined;
  }

  listCollections(): CollectionInfo[] {
    return this.records.listCollectionRows().map((row) => this.records.toInfo(row));
  }

  stats(name?: string): CollectionStats[] {
    const rows = name
      ? [this.records.getCollectionRow(normalizeName(name))].filter(
          (row): row is Record<string, unknown> => row !== null,
        )
      : this.records.listCollectionRows();
    return rows.map((row) => {
      const info = this.records.toInfo(row);
      return {
        ...info,
        embedded: this.records.countEmbedded(info.name),
        approxBytes: this.records.approxBytes(info.name),
      };
    });
  }

  async dropCollection(name: string): Promise<void> {
    const key = normalizeName(name);
    if (!this.records.getCollectionRow(key)) return;
    this.records.dropCollection(key);
    this.cache.invalidate(key);
    this.deps.logger.debug(`[vector] 已删除 collection ${key}`);
  }

  async upsert(input: UpsertInput): Promise<UpsertResult> {
    const result = await this.upsertMany([input]);
    const first = result.results[0];
    if (!first) throw new Error(`写入 ${input.collection} 失败`);
    return first;
  }

  async upsertMany(
    inputs: UpsertInput[],
    options: BatchOptions = {},
  ): Promise<BatchUpsertResult> {
    if (inputs.length === 0) return { results: [], failed: 0 };

    const collectionName = normalizeName(inputs[0].collection);
    for (const input of inputs) {
      if (normalizeName(input.collection) !== collectionName) {
        throw new Error("upsertMany 不支持跨 collection 批量写入");
      }
    }
    const info = await this.ensureCollection(collectionName);

    const results: UpsertResult[] = [];
    let failed = 0;
    const pending: Array<{
      input: UpsertInput;
      content: string;
      metadata: VectorMetadata;
      hash: string;
      existingHash: string;
      existingId?: number;
    }> = [];

    for (const input of inputs) {
      const docId = String(input.docId ?? "").trim();
      if (!docId) {
        failed += 1;
        continue;
      }
      const content = truncate(String(input.content ?? ""), info.maxContentChars);
      const metadata = input.metadata ?? {};
      const hash = embedHash(content, this.embedder.modelKey);
      const existing = this.records.findExisting(collectionName, docId);
      pending.push({
        input: { ...input, docId, content },
        content,
        metadata,
        hash,
        existingHash: existing?.embedHash ?? "",
        ...(existing && { existingId: existing.id }),
      });
    }

    // 内容未变、或调用方直接给了向量：不需要推理
    const needsEmbed = pending.filter(
      (item) => !hasUsableVector(item.input, info) && item.hash !== item.existingHash,
    );

    const vectors = new Map<number, Float32Array>();
    const embeddedIndexes = new Set<number>();
    if (needsEmbed.length > 0) {
      if (!this.embedder.available) {
        // 模型还在加载或不可用：先落文本，下次写入时补向量
        this.deps.logger.debug(
          `[vector] ${collectionName} 有 ${needsEmbed.length} 条待向量化，但模型未就绪，先只写文本`,
        );
      } else {
        try {
          const embedded = await this.embedder.embed(
            needsEmbed.map((item) => item.content),
          );
          needsEmbed.forEach((item, index) => {
            const target = pending.indexOf(item);
            vectors.set(target, embedded[index]);
            embeddedIndexes.add(target);
          });
        } catch (error) {
          this.deps.logger.warn(
            `[vector] ${collectionName} 向量化失败，仅写入文本: ${error}`,
          );
          failed += needsEmbed.length;
        }
      }
    }

    for (let i = 0; i < pending.length; i++) {
      const item = pending[i];
      const direct = toDirectVector(item.input, info, this.records);
      const vector = direct ?? vectors.get(i);
      const dim = vector?.length ?? info.dim;
      this.store.transaction(() => {
        const written = this.records.write({
          collection: collectionName,
          docId: item.input.docId,
          content: item.content,
          metadata: item.metadata,
          embedHash: vector ? item.hash : item.existingHash,
          ...(vector && { vector }),
          ...(item.input.createdAt != null && { createdAt: item.input.createdAt }),
        });
        this.store.reindexRow(
          collectionName,
          written.id,
          toIndexText(item.content),
        );
      });
      results.push({
        docId: item.input.docId,
        action: item.existingId === undefined ? "inserted" : "updated",
        // 只有真正跑过推理才算
        embedded: embeddedIndexes.has(i),
        dim,
      });
    }

    if (info.maxRecords > 0) {
      const evicted = this.records.enforceLimit(collectionName, info.maxRecords);
      if (evicted.length > 0) {
        this.store.removeIndexRows(collectionName, evicted);
        this.deps.logger.debug(
          `[vector] ${collectionName} 超出上限 ${info.maxRecords}，淘汰 ${evicted.length} 条`,
        );
      }
    }

    // 首次拿到向量后回填维度，让后续检索走稠密
    if (info.dim === 0) {
      const probed = this.embedder.knownDim;
      if (probed && probed > 0) {
        this.records.updateCollection(collectionName, { dim: probed });
      }
    }

    this.cache.invalidate(collectionName);
    return { results, failed };
  }

  async search(query: SearchQuery): Promise<SearchHit[]> {
    return this.runQuery(query, query.mode ?? "hybrid");
  }

  async searchDense(query: SearchQuery): Promise<SearchHit[]> {
    return this.runQuery(query, "dense");
  }

  async searchSparse(query: SearchQuery): Promise<SearchHit[]> {
    return this.runQuery(query, "sparse");
  }

  cacheStats(): VectorCacheStats {
    return this.cache.stats();
  }

  clearCache(): void {
    this.cache.clear();
  }

  private async runQuery(
    query: SearchQuery,
    mode: SearchQuery["mode"],
  ): Promise<SearchHit[]> {
    const key = normalizeName(query.collection);
    const info = this.getCollection(key);
    if (!info) return [];
    const execute = (): Promise<SearchHit[]> =>
      runSearch(
        {
          store: this.store,
          records: this.records,
          embedder: this.embedder,
          defaults: {
            denseK: RETRIEVAL_DEFAULTS.denseK,
            sparseK: RETRIEVAL_DEFAULTS.sparseK,
            denseWeight: RETRIEVAL_DEFAULTS.denseWeight,
            candidateLimit: RETRIEVAL_DEFAULTS.candidateLimit,
            recencyHalfLifeMs: 0,
          },
          ...(this.deps.rerank && { rerank: this.deps.rerank }),
          logger: this.deps.logger,
        },
        { ...query, mode },
        info,
      );

    if (query.cache === false) return execute();
    return this.cache.memo(key, queryCacheKey(query, mode ?? "hybrid"), execute);
  }

  get(collection: string, docId: string): StoredRecord | undefined {
    return this.records.getRecord(normalizeName(collection), docId);
  }

  count(collection: string, filter?: MetadataFilter): number {
    const key = normalizeName(collection);
    if (!filter) return this.records.countRecords(key);
    const compiled = compileFilter(filter, this.store.filterColumns);
    return this.records.countByFilter(key, compiled.sql, compiled.params);
  }

  remove(collection: string, docIds: string | string[]): number {
    const key = normalizeName(collection);
    const list = (Array.isArray(docIds) ? docIds : [docIds]).map((id) =>
      String(id),
    );
    const ids = this.records.deleteByDocIds(key, list);
    this.store.removeIndexRows(key, ids);
    this.cache.invalidate(key);
    return ids.length;
  }

  removeWhere(collection: string, filter: MetadataFilter): number {
    const key = normalizeName(collection);
    const compiled = compileFilter(filter, this.store.filterColumns);
    const ids = this.records.deleteByFilter(key, compiled.sql, compiled.params);
    this.store.removeIndexRows(key, ids);
    this.cache.invalidate(key);
    return ids.length;
  }

  clear(collection: string): number {
    const key = normalizeName(collection);
    const ids = this.records.clearCollection(key);
    this.store.clearFts(key);
    this.cache.invalidate(key);
    return ids.length;
  }

  async reindex(
    collection: string,
    options: ReindexOptions = {},
  ): Promise<ReindexResult> {
    const key = normalizeName(collection);
    const info = await this.ensureCollection(key, {
      ...(options.dim != null && { dim: options.dim }),
      ...(options.embedModel !== undefined && {
        embedModel: options.embedModel,
      }),
    });

    if (!(await this.embedder.load())) {
      throw new Error(
        `reindex ${key} 失败：本地向量模型不可用${this.embedder.error ? `（${this.embedder.error}）` : ""}`,
      );
    }

    const records = this.records.listCandidates(key, "1", [], 1_000_000);
    if (options.dim != null && options.dim !== info.dim) {
      this.records.updateCollection(key, { dim: options.dim });
      this.store.transaction(() => {
        for (const record of records) {
          this.store.db.run(
            "UPDATE records SET vector = NULL, embed_hash = '', dim = 0 WHERE id = ?",
            [record.id],
          );
        }
      });
    }

    const result: ReindexResult = {
      total: records.length,
      embedded: 0,
      failed: 0,
      dim: options.dim ?? info.dim,
    };

    const batchSize = Math.max(1, options.batchSize ?? 64);
    for (let i = 0; i < records.length; i += batchSize) {
      const batch = records.slice(i, i + batchSize);
      try {
        const vectors = await this.embedder.embed(
          batch.map((record) => record.content),
        );
        for (let j = 0; j < batch.length; j++) {
          this.records.updateVector(
            batch[j].id,
            embedHash(batch[j].content, this.embedder.modelKey),
            vectors[j],
          );
          result.dim = vectors[j].length;
          result.embedded += 1;
        }
      } catch (error) {
        result.failed += batch.length;
        this.deps.logger.warn(`[vector] reindex ${key} 批次失败: ${error}`);
      }
      options.onProgress?.(
        Math.min(i + batchSize, records.length),
        records.length,
      );
    }

    if (result.dim > 0 && result.dim !== info.dim) {
      this.records.updateCollection(key, { dim: result.dim });
    }
    this.cache.invalidate(key);
    return result;
  }

  dispose(): void {
    this.cache.clear();
    this.embedder.dispose();
    this.store.close();
  }

  private assertCompatible(
    info: CollectionInfo,
    options: CollectionOptions,
    requestedDim: number,
  ): void {
    if (requestedDim > 0 && info.dim > 0 && requestedDim !== info.dim) {
      throw new Error(
        `collection ${info.name} 的维度是 ${info.dim}，无法改成 ${requestedDim}；需要换维度请用 reindex`,
      );
    }
    const model = resolveEmbedModel(options, "");
    if (model && info.embedModel && model !== info.embedModel) {
      this.deps.logger.warn(
        `[vector] collection ${info.name} 已在 ${info.embedModel} 下建库，新模型 ${model} 的向量不可混用`,
      );
    }
  }
}

export function createVectorService(deps: VectorServiceDeps): VectorServiceImpl {
  return new VectorServiceImpl(deps);
}

function resolveEmbedModel(
  options: CollectionOptions,
  fallback: string,
): string {
  return String(options.embedModel ?? "").trim() || fallback;
}

function hasUsableVector(input: UpsertInput, info: CollectionInfo): boolean {
  if (!input.vector) return false;
  if (info.dim > 0 && input.vector.length !== info.dim) {
    throw new Error(
      `collection ${info.name} 需要 ${info.dim} 维向量，收到 ${input.vector.length} 维`,
    );
  }
  return true;
}

function toDirectVector(
  input: UpsertInput,
  info: CollectionInfo,
  records: RecordStore,
): Float32Array | undefined {
  if (!input.vector) return undefined;
  return records.prepare(input.vector, info.metric);
}

function truncate(text: string, max: number): string {
  return text.length > max ? text.slice(0, max) : text;
}

function normalizeName(name: string): string {
  const key = String(name ?? "").trim();
  if (!key) throw new Error("collection 名不能为空");
  return key;
}
