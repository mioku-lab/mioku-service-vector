import { defineService } from "mioku";
import type { BootstrapState } from "./bootstrap/state";

export type Metric = "cosine" | "dot";

export type TokenizerMode = "bigram";

export type VectorMetadata = Record<string, unknown>;

/** collection 声明，维度与 embedding 模型一经确定不再变更 */
export interface CollectionOptions {
  /** 向量维度；0 表示只做关键词检索 */
  dim?: number;
  /** 记录本 collection 使用的 embedding 模型，便于换模型时排查 */
  embedModel?: string;
  metric?: Metric;
  /** 单条内容超过该长度时截断后向量化，默认 4000 */
  maxContentChars?: number;
  /** 该 collection 的记录上限，0 表示不限；超出按 updated_at 淘汰 */
  maxRecords?: number;
  metadata?: VectorMetadata;
}

export interface CollectionInfo {
  name: string;
  dim: number;
  embedModel: string;
  metric: Metric;
  maxContentChars: number;
  maxRecords: number;
  metadata: VectorMetadata;
  records: number;
  createdAt: number;
  updatedAt: number;
}

export interface CollectionStats extends CollectionInfo {
  /** 带向量的记录数 */
  embedded: number;
  /** 磁盘占用字节数，按内容与向量估算 */
  approxBytes: number;
}

export interface UpsertInput {
  collection: string;
  /** 业务主键；同一 collection 内重复写入覆盖旧值 */
  docId: string;
  content: string;
  metadata?: VectorMetadata;
  /** 直接给定向量，跳过 embedding；维度必须与 collection 一致 */
  vector?: readonly number[];
  createdAt?: number;
}

export interface UpsertResult {
  docId: string;
  action: "inserted" | "updated";
  /** 本次是否真的调用了 embedding（内容未变时为 false） */
  embedded: boolean;
  dim: number;
}

export interface BatchOptions {
  /** 单批向量化的文本条数，默认取服务配置 */
  batchSize?: number;
  /** 并发批次上限，默认取服务配置 */
  concurrency?: number;
  abortSignal?: AbortSignal;
}

export interface BatchUpsertResult {
  results: UpsertResult[];
  failed: number;
}

export interface SearchHit {
  docId: string;
  content: string;
  metadata: VectorMetadata;
  score: number;
  denseScore?: number;
  sparseScore?: number;
  createdAt: number;
  updatedAt: number;
}

export type SearchMode = "hybrid" | "dense" | "sparse";

export interface SearchQuery {
  collection: string;
  query: string;
  topK?: number;
  /** 融合阶段候选数，默认 max(topK * 4, 20) */
  fetchK?: number;
  denseK?: number;
  sparseK?: number;
  denseWeight?: number;
  mode?: SearchMode;
  filter?: MetadataFilter;
  /** 时间衰减半衰期（毫秒）；0 表示不衰减 */
  recencyHalfLifeMs?: number;
  minScore?: number;
}

/** 元数据过滤器，只支持必要的几种形状 */
export type MetadataFilter =
  | { eq: Record<string, string | number | boolean> }
  | { in: Record<string, Array<string | number>> }
  | { range: Record<string, { gte?: number; lte?: number }> }
  | { and: MetadataFilter[] }
  | { or: MetadataFilter[] };

export interface ReindexOptions {
  embedModel?: string;
  /** 新维度；与旧值不同时重建向量列 */
  dim?: number;
  batchSize?: number;
  concurrency?: number;
  onProgress?: (done: number, total: number) => void;
}

export interface ReindexResult {
  total: number;
  embedded: number;
  failed: number;
  dim: number;
}

/** 向量存储服务暴露给插件的接口 */
export interface VectorServiceApi {
  /** 幂等创建 collection；维度与既有值冲突时抛错 */
  ensureCollection(
    name: string,
    options?: CollectionOptions,
  ): Promise<CollectionInfo>;
  getCollection(name: string): CollectionInfo | undefined;
  listCollections(): CollectionInfo[];
  stats(name?: string): CollectionStats[];
  dropCollection(name: string): Promise<void>;

  /** 写入一条，按 docId upsert；内容未变则跳过向量化 */
  upsert(input: UpsertInput): Promise<UpsertResult>;
  upsertMany(
    inputs: UpsertInput[],
    options?: BatchOptions,
  ): Promise<BatchUpsertResult>;

  /** 混合检索：稠密 + 稀疏融合 */
  search(query: SearchQuery): Promise<SearchHit[]>;
  searchDense(query: SearchQuery): Promise<SearchHit[]>;
  searchSparse(query: SearchQuery): Promise<SearchHit[]>;

  get(collection: string, docId: string): StoredRecord | undefined;
  count(collection: string, filter?: MetadataFilter): number;
  remove(collection: string, docIds: string | string[]): number;
  removeWhere(collection: string, filter: MetadataFilter): number;
  clear(collection: string): number;
  /** 换模型/换维度后重建向量 */
  reindex(collection: string, options?: ReindexOptions): Promise<ReindexResult>;

  /** 本地向量模型的加载状态 */
  status(): BootstrapState;
  /** 订阅状态变化，注册时会立刻回调一次当前状态 */
  onStatus(listener: (state: BootstrapState) => void): () => void;

  dispose(): void;
}

/** 服务内部与检索层共用的完整记录 */
export interface StoredRecord {
  id: number;
  collection: string;
  docId: string;
  content: string;
  metadata: VectorMetadata;
  embedHash: string;
  dim: number;
  vector?: Float32Array;
  createdAt: number;
  updatedAt: number;
}

export interface SparseHit {
  record: StoredRecord;
  score: number;
  rank: number;
}

export interface DenseHit {
  record: StoredRecord;
  score: number;
  rank: number;
}

export const VectorService = defineService<VectorServiceApi>("vector");

/** collection 引用的品牌标记，避免把裸字符串当引用传 */
declare const collectionBrand: unique symbol;

export type VectorCollection<TMeta extends VectorMetadata = VectorMetadata> =
  string & { readonly [collectionBrand]?: TMeta };

/** 声明一个 collection 的类型化引用 */
export function defineVectorCollection<
  TMeta extends VectorMetadata = VectorMetadata,
>(name: string): VectorCollection<TMeta> {
  return name as VectorCollection<TMeta>;
}
