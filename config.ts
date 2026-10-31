import type { Metric } from "./types";

/** 内部检索参数：用户不调，集中在这里便于统一调整 */
export const RETRIEVAL_DEFAULTS = {
  denseK: 50,
  sparseK: 50,
  denseWeight: 0.9,
  /** 稠密检索单次扫描的记录上限，防止大库拖慢消息主路径 */
  candidateLimit: 20000,
} as const;

export const STORAGE_DEFAULTS = {
  /** 单条内容超过该长度时截断后再向量化 */
  maxContentChars: 4000,
  /** 0 表示不限；collection 可按需覆盖 */
  maxRecords: 0,
} as const;

export interface LocalEmbeddingModel {
  id: string;
  label: string;
  repo: string;
  sizeMb: number;
  note: string;
}

export const LOCAL_EMBEDDING_MODELS: LocalEmbeddingModel[] = [
  {
    id: "bge-small-zh",
    label: "bge-small-zh-v1.5（24MB，推荐）",
    repo: "Xenova/bge-small-zh-v1.5",
    sizeMb: 24,
    note: "中文短文本检索，速度快、体积小，适合群聊记忆",
  },
  {
    id: "bge-base-zh",
    label: "bge-base-zh-v1.5（103MB，更准）",
    repo: "Xenova/bge-base-zh-v1.5",
    sizeMb: 103,
    note: "中文检索精度更高，代价是首次下载更久、单条耗时略高",
  },
  {
    id: "multilingual-mini",
    label: "paraphrase-multilingual-MiniLM（118MB，多语言）",
    repo: "Xenova/paraphrase-multilingual-MiniLM-L12-v2",
    sizeMb: 118,
    note: "中英混排或含较多外语内容时更稳，纯中文场景不如上面两个",
  },
];

export const DEFAULT_EMBEDDING_MODEL = LOCAL_EMBEDDING_MODELS[0].id;

/** 国内访问 HuggingFace 直连不稳定，默认走镜像；置空可改回官方地址 */
export const DEFAULT_HF_MIRROR = "https://hf-mirror.com";

/** 服务配置：键名与 config.md 中的 `base.` 字段一一对应 */
export interface VectorConfig {
  enabled: boolean;
  /** LOCAL_EMBEDDING_MODELS 中的 id */
  model: string;
  hfMirror: string;
  autoDownload: boolean;
  metric: Metric;
}

export const DEFAULT_VECTOR_CONFIG: VectorConfig = {
  enabled: true,
  model: DEFAULT_EMBEDDING_MODEL,
  hfMirror: DEFAULT_HF_MIRROR,
  autoDownload: true,
  metric: "cosine",
};

export function findEmbeddingModel(id: string): LocalEmbeddingModel | undefined {
  const key = String(id ?? "").trim();
  return LOCAL_EMBEDDING_MODELS.find((item) => item.id === key);
}

function bool(value: unknown, fallback: boolean): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function str(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value.trim() : fallback;
}

/** 合并磁盘配置与默认值；未知模型回落到默认模型 */
export function normalizeVectorConfig(
  value: Record<string, unknown>,
): VectorConfig {
  const model = str(value.model, DEFAULT_EMBEDDING_MODEL);
  return {
    enabled: bool(value.enabled, DEFAULT_VECTOR_CONFIG.enabled),
    model: findEmbeddingModel(model) ? model : DEFAULT_EMBEDDING_MODEL,
    hfMirror: value.hfMirror === undefined ? DEFAULT_VECTOR_CONFIG.hfMirror : str(value.hfMirror),
    autoDownload: bool(value.autoDownload, DEFAULT_VECTOR_CONFIG.autoDownload),
    metric: str(value.metric) === "dot" ? "dot" : "cosine",
  };
}
