import {
  ensureDataDir,
  getServiceConfig,
  logger,
  registerServiceConfig,
} from "mioku";
import type { MiokuService } from "mioku";
import * as path from "node:path";
import { createVectorService, type VectorServiceImpl } from "./service";
import { DEFAULT_VECTOR_CONFIG, normalizeVectorConfig } from "./config";
import type { VectorServiceApi } from "./types";

const vectorService: MiokuService = {
  name: "vector",
  api: {} as VectorServiceApi,

  async init() {
    await registerServiceConfig("vector", "base", {
      ...DEFAULT_VECTOR_CONFIG,
    });
    const config = normalizeVectorConfig(
      await getServiceConfig("vector", "base"),
    );

    const dataDir = ensureDataDir("vector");
    const impl = createVectorService({
      dbPath: path.join(dataDir, "vector.db"),
      modelDir: path.join(dataDir, "models"),
      config,
      logger: {
        info: (message) => logger.info(message),
        debug: (message) => logger.debug(message),
        warn: (message) => logger.warn(message),
        error: (message) => logger.error(message),
      },
    });

    // 模型加载放后台：服务先就绪，检索在模型可用前自动退化为关键词
    impl.onStatus((state) => {
      const line = `[vector] ${state.message}${state.reason ? `：${state.reason}` : ""}`;
      if (state.phase === "failed") logger.warn(line);
      else logger.info(line);
    });
    impl.start();

    this.api = impl;
    logger.info(`vector 服务已就绪（${impl.describe()}）`);
  },

  async dispose() {
    (this.api as VectorServiceImpl | undefined)?.dispose?.();
    logger.info("vector 服务已卸载");
  },
};

export default vectorService;

export { VectorService, defineVectorCollection } from "./types";
export type { VectorCollection, VectorServiceApi } from "./types";
export type {
  BatchOptions,
  BatchUpsertResult,
  CollectionInfo,
  CollectionOptions,
  CollectionStats,
  MetadataFilter,
  Metric,
  ReindexOptions,
  ReindexResult,
  SearchHit,
  SearchMode,
  SearchQuery,
  StoredRecord,
  UpsertInput,
  UpsertResult,
  VectorMetadata,
} from "./types";
export type { BootstrapPhase, BootstrapState } from "./bootstrap/state";
export type { VectorConfig } from "./config";
