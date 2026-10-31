import { mkdir, rename, rm, stat, writeFile } from "node:fs/promises";
import * as path from "node:path";

export interface DownloadProgress {
  file: string;
  loaded: number;
  total: number;
  index: number;
  count: number;
}

export interface DownloadFileSpec {
  file: string;
  /** 体积下限，用于判断已有文件是否完整 */
  minBytes: number;
}

export interface DownloadOptions {
  repo: string;
  /** 落地目录，文件按仓库内相对路径写入 */
  targetDir: string;
  remoteHost: string;
  files: DownloadFileSpec[];
  onProgress?: (progress: DownloadProgress) => void;
  logger: { debug: (message: string) => void };
}

const MAX_ATTEMPTS = 3;

export async function isFileComplete(
  filePath: string,
  minBytes: number,
): Promise<boolean> {
  try {
    const info = await stat(filePath);
    return info.isFile() && info.size >= minBytes;
  } catch {
    return false;
  }
}

export async function downloadModelFiles(
  options: DownloadOptions,
): Promise<void> {
  const { repo, targetDir, remoteHost, files } = options;
  const base = remoteHost.replace(/\/$/, "");

  for (let i = 0; i < files.length; i++) {
    const spec = files[i];
    const target = path.join(targetDir, spec.file);
    if (await isFileComplete(target, spec.minBytes)) {
      options.logger.debug(`[vector] ${spec.file} 已存在，跳过下载`);
      continue;
    }

    const url = `${base}/${repo}/resolve/main/${spec.file}`;
    let lastError = "";
    let done = false;
    for (let attempt = 1; attempt <= MAX_ATTEMPTS && !done; attempt++) {
      try {
        await downloadOne(url, target, (loaded, total) => {
          options.onProgress?.({
            file: spec.file,
            loaded,
            total,
            index: i + 1,
            count: files.length,
          });
        });
        done = true;
      } catch (error) {
        lastError = String(error instanceof Error ? error.message : error);
        options.logger.debug(
          `[vector] 下载 ${spec.file} 第 ${attempt} 次失败: ${lastError}`,
        );
        if (attempt < MAX_ATTEMPTS) {
          await sleep(1_000 * 2 ** (attempt - 1));
        }
      }
    }
    if (!done) {
      throw new Error(
        `下载 ${spec.file} 失败（已重试 ${MAX_ATTEMPTS} 次）：${lastError}`,
      );
    }
  }
}

async function downloadOne(
  url: string,
  target: string,
  onProgress: (loaded: number, total: number) => void,
): Promise<void> {
  await mkdir(path.dirname(target), { recursive: true });
  const temp = `${target}.part`;
  try {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const total = Number(res.headers.get("content-length") ?? 0);
    const buffer = new Uint8Array(await res.arrayBuffer());
    // 一次性写入：模型文件最大百兆级，分片写入带来的复杂度不值得
    await writeFile(temp, buffer);
    onProgress(buffer.byteLength, total || buffer.byteLength);
    await rename(temp, target);
  } catch (error) {
    await rm(temp, { force: true });
    throw error;
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
