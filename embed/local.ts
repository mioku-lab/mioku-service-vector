export interface LocalPipeline {
  (
    texts: string[],
    options: { pooling: "mean"; normalize: boolean },
  ): Promise<{ dims: number[]; tolist(): number[][] }>;
}

interface TransformersModule {
  pipeline: (
    task: string,
    model: string,
    options?: Record<string, unknown>,
  ) => Promise<unknown>;
  env: {
    allowRemoteModels: boolean;
    allowLocalModels: boolean;
    localModelPath: string;
    cacheDir: string | null;
    backends?: { onnx?: { logLevel?: string } };
  };
}

export interface LocalModelOptions {
  /**
   * 模型根目录；库会把仓库名拼在其后，即 <modelRoot>/<repo>。
   * 该目录下需已含 config.json / tokenizer.json / onnx/model_quantized.onnx。
   */
  modelRoot: string;
  /** 仓库名，同时作为传给库的模型标识 */
  repo: string;
  logger: { debug: (message: string) => void };
}

/**
 * 本地向量模型：只从本地目录加载量化 ONNX 并推理。
 * 依赖缺失或初始化失败都只影响 available，不影响服务启动。
 */
export class LocalEmbeddingModel {
  private pipelineInstance: LocalPipeline | null = null;
  private initPromise: Promise<boolean> | null = null;
  private dim: number | undefined;
  private lastError = "";

  constructor(private readonly options: LocalModelOptions) {}

  get available(): boolean {
    return this.pipelineInstance !== null;
  }

  get error(): string {
    return this.lastError;
  }

  get knownDim(): number | undefined {
    return this.dim;
  }

  load(): Promise<boolean> {
    if (this.pipelineInstance) return Promise.resolve(true);
    this.initPromise ??= this.doLoad();
    return this.initPromise;
  }

  private async doLoad(): Promise<boolean> {
    try {
      const module = await this.importTransformers();
      const { env } = module;

      // 模型文件由 bootstrap 自己下好，这里只认本地目录。
      // 模型名必须是合法 HF id，库才会把 localModelPath 拼进路径。
      env.allowRemoteModels = false;
      env.allowLocalModels = true;
      env.localModelPath = this.options.modelRoot;
      env.cacheDir = null;
      if (env.backends?.onnx) env.backends.onnx.logLevel = "error";

      const pipeline = (await module.pipeline(
        "feature-extraction",
        this.options.repo,
        { dtype: "q8" },
      )) as LocalPipeline;
      this.pipelineInstance = pipeline;

      const [warm] = await this.embed(["warmup"]);
      if (!warm) throw new Error("预热推理未返回向量");
      return true;
    } catch (error) {
      this.lastError = String(
        error instanceof Error ? error.message : error,
      ).slice(0, 300);
      this.options.logger.debug(
        `[vector] 本地向量模型加载失败: ${this.lastError}`,
      );
      return false;
    }
  }

  private async importTransformers(): Promise<TransformersModule> {
    // 动态导入：依赖缺失时给出可读提示，而不是崩在模块解析阶段
    const loaded = (await import("@huggingface/transformers")) as unknown as
      | TransformersModule
      | { default: TransformersModule };
    const module =
      "pipeline" in loaded
        ? loaded
        : (loaded as { default: TransformersModule }).default;
    if (!module?.pipeline) {
      throw new Error("@huggingface/transformers 导出结构异常");
    }
    return module;
  }

  async embed(texts: string[]): Promise<Float32Array[]> {
    if (texts.length === 0) return [];
    const pipeline = this.pipelineInstance;
    if (!pipeline) throw new Error("本地向量模型尚未就绪");

    const output = await pipeline(texts, { pooling: "mean", normalize: true });
    const rows = output.tolist();
    if (rows.length !== texts.length) {
      throw new Error(
        `向量化返回数量不符：期望 ${texts.length}，实际 ${rows.length}`,
      );
    }
    const dim = rows[0]?.length ?? 0;
    if (dim > 0) this.dim = dim;
    return rows.map((row) => Float32Array.from(row));
  }
}
