import { VectorCache } from "./cache";
import { LocalEmbeddingModel } from "./local";
import { embedHash } from "../utils/hash";

export interface TextEmbeddingModel {
  readonly available: boolean;
  readonly knownDim: number | undefined;
  readonly error: string;
  load(): Promise<boolean>;
  embed(texts: string[]): Promise<Float32Array[]>;
}

export interface EmbedderOptions {
  /** 模型文件根目录（已下载完成） */
  modelRoot: string;
  repo: string;
  /** 模型 id，用于内容指纹与 collection 记录 */
  modelKey: string;
  batchSize: number;
  cacheSize: number;
  /** 注入推理后端；缺省时用本地 ONNX 模型 */
  model?: TextEmbeddingModel;
  logger: {
    debug: (message: string) => void;
    warn: (message: string) => void;
  };
}

/**
 * 向量化门面：驱动本地模型、按内容哈希缓存、按批推理。
 * 模型未就绪时 available 为 false，检索与写入自动退化。
 */
export class Embedder {
  private readonly cache: VectorCache;
  private readonly model: TextEmbeddingModel;
  /** 加载完成前的写入请求会被拒绝，避免阻塞消息主路径 */
  private loading: Promise<boolean> | null = null;

  constructor(private readonly options: EmbedderOptions) {
    this.cache = new VectorCache(options.cacheSize);
    this.model =
      options.model ??
      new LocalEmbeddingModel({
        modelRoot: options.modelRoot,
        repo: options.repo,
        logger: options.logger,
      });
  }

  get available(): boolean {
    return this.model.available;
  }

  get modelKey(): string {
    return this.options.modelKey;
  }

  get knownDim(): number | undefined {
    return this.model.knownDim;
  }

  get error(): string {
    return this.model.error;
  }

  load(): Promise<boolean> {
    if (this.model.available) return Promise.resolve(true);
    this.loading ??= this.model.load();
    return this.loading;
  }

  async embed(texts: string[]): Promise<Float32Array[]> {
    if (texts.length === 0) return [];
    const ready = await this.load();
    if (!ready) {
      throw new Error(this.error || "本地向量模型不可用");
    }

    const results = new Array<Float32Array | undefined>(texts.length);
    const pending: Array<{ index: number; text: string; key: string }> = [];

    for (let i = 0; i < texts.length; i++) {
      const key = embedHash(texts[i], this.options.modelKey);
      const cached = this.cache.get(key);
      if (cached) results[i] = cached;
      else pending.push({ index: i, text: texts[i], key });
    }

    const size = Math.max(1, this.options.batchSize);
    for (let i = 0; i < pending.length; i += size) {
      const batch = pending.slice(i, i + size);
      const vectors = await this.model.embed(batch.map((item) => item.text));
      for (let j = 0; j < batch.length; j++) {
        const vector = vectors[j];
        if (!vector) continue;
        results[batch[j].index] = vector;
        this.cache.set(batch[j].key, vector);
      }
    }

    return results.map((vector, index) => {
      if (vector) return vector;
      throw new Error(`向量化失败：第 ${index} 条未返回向量`);
    });
  }

  clearCache(): void {
    this.cache.clear();
  }

  dispose(): void {
    this.cache.clear();
  }
}
