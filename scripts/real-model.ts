/**
 * 真实模型端到端验证：用已下载的本地模型跑语义检索。
 * 跑法（需先启动过一次 example 以完成模型下载）：
 *   bun run scripts/real-model.ts
 */
import { mkdtempSync, rmSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import * as path from "node:path";
import { createVectorService } from "../service";
import { DEFAULT_VECTOR_CONFIG, LOCAL_EMBEDDING_MODELS } from "../config";

const modelRoot =
  process.env.VECTOR_MODEL_ROOT ??
  path.resolve(import.meta.dir, "../../../example/data/vector/models");

let failures = 0;
let checks = 0;
function check(label: string, ok: boolean, detail = ""): void {
  checks += 1;
  if (ok) console.log(`  ✓ ${label}`);
  else {
    failures += 1;
    console.log(`  ✗ ${label}${detail ? ` — ${detail}` : ""}`);
  }
}

const modelId = process.argv[2] ?? "bge-small-zh";
const model = LOCAL_EMBEDDING_MODELS.find((m) => m.id === modelId);
if (!model) {
  console.error(`未知模型 ${modelId}，可选：${LOCAL_EMBEDDING_MODELS.map((m) => m.id).join(", ")}`);
  process.exit(1);
}
const dir = path.join(modelRoot, model.repo);
if (!existsSync(dir)) {
  console.error(`模型未下载：${dir}\n请先启动一次 example，或手动放置模型文件。`);
  process.exit(1);
}

const dataDir = mkdtempSync(path.join(tmpdir(), "mioku-vector-real-"));
const service = createVectorService({
  dbPath: path.join(dataDir, "vector.db"),
  modelDir: modelRoot,
  config: { ...DEFAULT_VECTOR_CONFIG, model: model.id, autoDownload: false },
  logger: {
    info: (m) => console.log(`    ${m}`),
    debug: () => {},
    warn: (m) => console.log(`    WARN ${m}`),
    error: () => {},
  },
});

console.log(`模型：${model.label}\n`);
const t0 = Date.now();
service.start();
await new Promise<void>((resolve) => {
  const off = service.onStatus((s) => {
    if (s.phase === "ready" || s.phase === "failed") {
      off();
      resolve();
    }
  });
});
const state = service.status();
check(`模型加载就绪（${((Date.now() - t0) / 1000).toFixed(1)}s）`, state.ready, JSON.stringify(state));
if (!state.ready) {
  service.dispose();
  process.exit(1);
}

const CORPUS = [
  { docId: "a1", content: "今天加班到十点，累死了", tag: "tired" },
  { docId: "a2", content: "刚吃完火锅，好撑", tag: "food" },
  { docId: "a3", content: "晚上一起开黑上分吗", tag: "game" },
  { docId: "a4", content: "这首歌好好听，单曲循环了", tag: "music" },
  { docId: "a5", content: "小明昨天退群了，说是要备考", tag: "leave" },
  { docId: "a6", content: "群号 123456 的公告已经更新", tag: "notice" },
];
const written = await service.upsertMany(
  CORPUS.map((item) => ({
    collection: "real",
    docId: item.docId,
    content: item.content,
    metadata: { tag: item.tag },
  })),
);
check("全部写入并向量化", written.results.length === 6 && written.results.every((r) => r.embedded));
const info = service.getCollection("real");
check(`维度自动探测为 ${model.id === "multilingual-mini" ? 384 : model.id === "bge-base-zh" ? 768 : 512}`, info?.dim === (model.id === "multilingual-mini" ? 384 : model.id === "bge-base-zh" ? 768 : 512), JSON.stringify(info?.dim));

console.log("\n语义召回（查询与原文无字面重合）：");
const cases: Array<[query: string, expected: string, label: string]> = [
  ["最近好疲惫啊", "a1", "「疲惫」→「加班累死了」"],
  ["想吃点热乎的", "a2", "「热乎的」→「火锅」"],
  ["有人一起打排位吗", "a3", "「打排位」→「开黑上分」"],
  ["想听点音乐放松", "a4", "「音乐」→「单曲循环」"],
];
for (const [query, expected, label] of cases) {
  const hits = await service.searchDense({ collection: "real", query, topK: 1 });
  check(label, hits[0]?.docId === expected, `实际 ${hits[0]?.docId}（${hits[0]?.denseScore?.toFixed(3)}）`);
}

console.log("\n关键词召回（混合检索）：");
const kw = await service.search({ collection: "real", query: "退群", topK: 1 });
check("「退群」→ 退群那条", kw[0]?.docId === "a5", JSON.stringify(kw.map((h) => h.docId)));
const kw2 = await service.search({ collection: "real", query: "123456", topK: 1 });
check("「123456」→ 公告那条", kw2[0]?.docId === "a6", JSON.stringify(kw2.map((h) => h.docId)));

console.log("\n元数据过滤：");
const filtered = await service.search({ collection: "real", query: "累", topK: 3, filter: { eq: { tag: "tired" } } });
check("按 tag 过滤生效", filtered.every((h) => h.metadata.tag === "tired"), JSON.stringify(filtered.map((h) => h.metadata.tag)));

console.log("\nreindex：");
const re = await service.reindex("real");
check("reindex 覆盖全部", re.total === 6 && re.embedded === 6 && re.failed === 0, JSON.stringify(re));

service.dispose();
rmSync(dataDir, { recursive: true, force: true });
console.log(`\n${failures === 0 ? "全部通过" : "存在失败"}：${checks - failures}/${checks}`);
if (failures > 0) process.exit(1);
