import * as path from "node:path";
import { downloadModelFiles, type DownloadFileSpec } from "./download";
import type { LocalEmbeddingModel } from "../config";

/**
 * 加载 feature-extraction 量化模型需要的文件。
 * 文件名与 transformers.js 的约定一致：dtype=q8 对应 onnx/model_quantized.onnx。
 */
const MODEL_FILES: DownloadFileSpec[] = [
  { file: "config.json", minBytes: 100 },
  { file: "tokenizer.json", minBytes: 1_000 },
  { file: "tokenizer_config.json", minBytes: 100 },
  { file: "onnx/model_quantized.onnx", minBytes: 5_000_000 },
];

export interface EnsureModelOptions {
  model: LocalEmbeddingModel;
  /** 模型根目录，实际落盘到 <modelRoot>/<repo> */
  modelRoot: string;
  mirror: string;
  logger: { debug: (message: string) => void; info: (message: string) => void };
  onProgress?: (text: string) => void;
}

export function modelDirectory(modelRoot: string, repo: string): string {
  return path.join(modelRoot, repo);
}

export async function ensureModelFiles(options: EnsureModelOptions): Promise<string> {
  const { model, modelRoot, mirror } = options;
  const targetDir = modelDirectory(modelRoot, model.repo);
  const remoteHost = mirror.trim() || "https://huggingface.co";

  let lastReported = -1;
  await downloadModelFiles({
    repo: model.repo,
    targetDir,
    remoteHost,
    files: MODEL_FILES,
    logger: options.logger,
    onProgress: (progress) => {
      if (!options.onProgress) return;
      // 按 10% 一档汇报，避免刷屏
      const percent =
        progress.total > 0
          ? Math.floor((progress.loaded / progress.total) * 100)
          : 0;
      const bucket = Math.floor(percent / 10);
      if (bucket === lastReported) return;
      lastReported = bucket;
      options.onProgress(
        `正在下载模型 ${progress.file}（${progress.index}/${progress.count}，${percent}%）`,
      );
    },
  });

  return targetDir;
}
