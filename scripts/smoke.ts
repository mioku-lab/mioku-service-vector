/**
 * 冒烟验证：直接实例化服务实现，用确定性假向量替代真实 embedding。
 * 覆盖写入/检索/融合/降级/持久化/删除/重建，跑法：bun run smoke
 */
import { rmSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import * as path from "node:path";
import { createVectorService, type VectorServiceImpl } from "../service";
import { DEFAULT_VECTOR_CONFIG, type VectorConfig } from "../config";
import type { TextEmbeddingModel } from "../embed";

let failures = 0;
let checks = 0;

function check(label: string, condition: boolean, detail = ""): void {
  checks += 1;
  if (condition) {
    console.log(`  ✓ ${label}`);
  } else {
    failures += 1;
    console.log(`  ✗ ${label}${detail ? ` — ${detail}` : ""}`);
  }
}

function section(title: string): void {
  console.log(`\n=== ${title} ===`);
}

/**
 * 假模型：不加载真实 ONNX，用关键词主题轴生成确定性向量，
 * 让"语义相近"可被验证，同时避免测试依赖模型下载。
 */
const TOPICS: Record<string, number[]> = {
  tired: [1, 0, 0, 0],
  food: [0, 1, 0, 0],
  game: [0, 0, 1, 0],
  music: [0, 0, 0, 1],
};
const KEYWORDS: Record<string, string[]> = {
  tired: ["累", "疲惫", "加班", "辛苦"],
  food: ["吃", "火锅", "奶茶", "外卖", "撑"],
  game: ["开黑", "上分", "原神", "游戏"],
  music: ["歌", "音乐", "单曲循环", "唱"],
};

function fakeVector(text: string): number[] {
  const vector: number[] = [0, 0, 0, 0];
  let matched = false;
  for (const [topic, words] of Object.entries(KEYWORDS)) {
    if (words.some((word) => text.includes(word))) {
      matched = true;
      const axis = TOPICS[topic];
      for (let i = 0; i < 4; i++) vector[i] += axis[i];
    }
  }
  // 不用 every() 判断：TS 会把它当类型守卫，把 number[] 收窄成元组
  if (!matched) {
    let seed = 0;
    for (const ch of text) seed = (seed * 31 + ch.codePointAt(0)!) % 9973;
    for (let i = 0; i < 4; i++) vector[i] = ((seed >> i) & 7) / 8 + 0.05;
  }
  return vector;
}

interface HarnessOptions {
  /** false 时模拟模型不可用，验证降级路径 */
  embed?: boolean;
  model?: TextEmbeddingModel;
}

/** 假推理后端：同步可用，输出确定性 4 维向量 */
function fakeModel(options: { fail?: boolean } = {}): TextEmbeddingModel {
  let ready = !options.fail;
  let dim: number | undefined = options.fail ? undefined : 4;
  return {
    get available() {
      return ready;
    },
    get knownDim() {
      return dim;
    },
    get error() {
      return options.fail ? "测试用假模型故意失败" : "";
    },
    async load() {
      if (options.fail) return false;
      ready = true;
      dim = 4;
      return true;
    },
    async embed(texts: string[]) {
      if (!ready) throw new Error("模型未就绪");
      return texts.map((text) => Float32Array.from(fakeVector(text)));
    },
  };
}

function harness(dir: string, options: HarnessOptions = {}): VectorServiceImpl {
  const config: VectorConfig = {
    ...DEFAULT_VECTOR_CONFIG,
    ...(options.embed === false && { enabled: false }),
  };
  const service = createVectorService({
    dbPath: path.join(dir, "vector.db"),
    modelDir: path.join(dir, "models"),
    config,
    ...(options.embed !== false && {
      model: options.model ?? fakeModel(),
    }),
    logger: {
      info: () => {},
      debug: () => {},
      warn: () => {},
      error: () => {},
    },
  });
  service.start();
  return service;
}

const CORPUS = [
  { docId: "m1", content: "今天加班到十点，累死了", groupId: "g1", userId: "u1" },
  { docId: "m2", content: "刚吃完火锅，好撑", groupId: "g1", userId: "u2" },
  { docId: "m3", content: "晚上一起开黑上分吗", groupId: "g1", userId: "u1" },
  { docId: "m4", content: "这首歌好好听，单曲循环了", groupId: "g2", userId: "u3" },
  { docId: "m5", content: "小明昨天退群了，说是要备考", groupId: "g1", userId: "u4" },
  { docId: "m6", content: "群号 123456 的公告已经更新", groupId: "g2", userId: "u5" },
];

async function main(): Promise<void> {
  const dir = mkdtempSync(path.join(tmpdir(), "mioku-vector-"));
  console.log(`临时目录: ${dir}`);

  section("1. collection 生命周期");
  let service = harness(dir);
  const created = await service.ensureCollection("chat.message", { dim: 4 });
  check("创建 collection", created.name === "chat.message" && created.dim === 4);
  const again = await service.ensureCollection("chat.message", { dim: 4 });
  check("重复创建幂等", again.createdAt === created.createdAt);
  let conflict = "";
  try {
    await service.ensureCollection("chat.message", { dim: 8 });
  } catch (error) {
    conflict = String(error);
  }
  check("维度冲突被拒绝", conflict.includes("维度"), conflict);

  section("2. 批量写入");
  const written = await service.upsertMany(
    CORPUS.map((item) => ({
      collection: "chat.message",
      docId: item.docId,
      content: item.content,
      metadata: { groupId: item.groupId, userId: item.userId },
    })),
  );
  check("全部写入成功", written.results.length === 6 && written.failed === 0, JSON.stringify(written.failed));
  check("首次写入均为 inserted", written.results.every((r) => r.action === "inserted"));
  check("全部完成向量化", written.results.every((r) => r.embedded));

  section("3. 稀疏检索（中文两字词，trigram 会漏，bigram 必须命中）");
  const sparse = await service.searchSparse({
    collection: "chat.message",
    query: "退群",
    topK: 3,
  });
  check("两字中文查询命中", sparse.length === 1 && sparse[0].docId === "m5", JSON.stringify(sparse.map((h) => h.docId)));
  const sparseFood = await service.searchSparse({ collection: "chat.message", query: "火锅", topK: 3 });
  check("另一个两字查询命中", sparseFood[0]?.docId === "m2", JSON.stringify(sparseFood.map((h) => h.docId)));
  const sparseNumber = await service.searchSparse({ collection: "chat.message", query: "123456", topK: 3 });
  check("数字查询命中", sparseNumber[0]?.docId === "m6", JSON.stringify(sparseNumber.map((h) => h.docId)));
  const sparseNone = await service.searchSparse({ collection: "chat.message", query: "完全不相关的内容", topK: 3 });
  check("无关查询返回空", sparseNone.length === 0, JSON.stringify(sparseNone.map((h) => h.docId)));

  section("4. 稠密检索（字面不重合，靠语义命中）");
  const dense = await service.searchDense({ collection: "chat.message", query: "好疲惫啊", topK: 3 });
  check("语义命中「加班累死了」", dense[0]?.docId === "m1", JSON.stringify(dense.map((h) => h.docId)));
  const denseMusic = await service.searchDense({ collection: "chat.message", query: "想听点音乐", topK: 3 });
  check("语义命中音乐类", denseMusic[0]?.docId === "m4", JSON.stringify(denseMusic.map((h) => h.docId)));

  section("5. 混合检索与元数据过滤");
  const hybrid = await service.search({ collection: "chat.message", query: "退群", topK: 3 });
  check("混合检索命中", hybrid[0]?.docId === "m5", JSON.stringify(hybrid.map((h) => h.docId)));
  check("混合结果带稠密与稀疏分", hybrid[0]?.sparseScore !== undefined);
  const filtered = await service.search({
    collection: "chat.message",
    query: "累",
    topK: 5,
    filter: { eq: { groupId: "g2" } },
  });
  check(
    "metadata 过滤生效",
    filtered.every((hit) => hit.metadata.groupId === "g2"),
    JSON.stringify(filtered.map((h) => h.metadata.groupId)),
  );
  const filteredIn = await service.count("chat.message", { in: { userId: ["u1"] } });
  check("in 过滤计数正确", filteredIn === 2, String(filteredIn));
  const filteredRange = await service.count("chat.message", {
    range: { createdAt: { gte: 0 } },
  });
  check("range 过滤计数正确", filteredRange === 6, String(filteredRange));

  section("6. 去重：内容未变不重复向量化");
  const noop = await service.upsertMany([
    {
      collection: "chat.message",
      docId: "m1",
      content: "今天加班到十点，累死了",
      metadata: { groupId: "g1", userId: "u1" },
    },
  ]);
  check("内容未变时 embedded=false", noop.results[0]?.embedded === false);
  check("内容未变时 action=updated", noop.results[0]?.action === "updated");
  check("记录数未增加", service.count("chat.message") === 6, String(service.count("chat.message")));

  const changed = await service.upsertMany([
    {
      collection: "chat.message",
      docId: "m1",
      content: "今天加班到十二点，累死了",
      metadata: { groupId: "g1", userId: "u1" },
    },
  ]);
  check("内容变化时重新向量化", changed.results[0]?.embedded === true);
  const updated = await service.searchSparse({ collection: "chat.message", query: "十二点", topK: 3 });
  check("索引随内容更新", updated[0]?.docId === "m1", JSON.stringify(updated.map((h) => h.docId)));

  section("7. 直接给向量（跳过 embedding）");
  // 单独一个 collection：其他记录的假向量会干扰，这里只验证直接向量能落库并参与检索
  await service.ensureCollection("direct", { dim: 4 });
  const direct = await service.upsertMany([
    {
      collection: "direct",
      docId: "d1",
      content: "外部已经算好向量的内容",
      metadata: { groupId: "g3", userId: "u9" },
      vector: [1, 0, 0, 0],
    },
  ]);
  check("直接向量写入成功", direct.results[0]?.dim === 4, JSON.stringify(direct.results[0]));
  check("直接向量不算 embedding 调用", direct.results[0]?.embedded === false);
  const directHit = await service.search({
    collection: "direct",
    query: "外部算好的",
    topK: 1,
    mode: "dense",
  });
  check("直接向量参与检索", directHit[0]?.docId === "d1", JSON.stringify(directHit.map((h) => h.docId)));
  let dimMismatch = "";
  try {
    await service.upsert({
      collection: "direct",
      docId: "bad",
      content: "维度不对",
      vector: [1, 0, 0],
    });
  } catch (error) {
    dimMismatch = String(error);
  }
  check("直接向量维度不符时抛错", dimMismatch.includes("维向量"), dimMismatch);
  check("维度不符的写入未落库", service.get("direct", "bad") === undefined);

  section("8. 删除与清空");
  const removed = service.remove("chat.message", ["m6"]);
  check("按 docId 删除", removed === 1, String(removed));
  // 断言看稀疏侧：稠密侧一定会返回"最不相似"的几条，不能用来判断删除
  const removedSparse = await service.searchSparse({
    collection: "chat.message",
    query: "123456",
    topK: 3,
  });
  check(
    "删除后稀疏索引同步清掉",
    removedSparse.length === 0,
    JSON.stringify(removedSparse.map((h) => h.docId)),
  );
  check("被删记录不再命中", service.get("chat.message", "m6") === undefined);
  const removedById = service.removeWhere("chat.message", { eq: { groupId: "g2" } });
  check("按元数据批量删除", removedById === 1, String(removedById));
  check("删除后计数正确", service.count("chat.message") === 4, String(service.count("chat.message")));

  section("9. 持久化（重新打开同一个库）");
  service.dispose();
  service = harness(dir);
  const reopened = service.getCollection("chat.message");
  check("collection 元信息保留", reopened?.dim === 4, JSON.stringify(reopened));
  check("记录保留", service.count("chat.message") === 4, String(service.count("chat.message")));
  const reopenedSparse = await service.searchSparse({ collection: "chat.message", query: "退群", topK: 3 });
  check("FTS 索引跨重启可用", reopenedSparse[0]?.docId === "m5", JSON.stringify(reopenedSparse.map((h) => h.docId)));
  const reopenedDense = await service.searchDense({ collection: "chat.message", query: "好疲惫", topK: 2 });
  check("向量跨重启可用", reopenedDense[0]?.docId === "m1", JSON.stringify(reopenedDense.map((h) => h.docId)));

  section("10. stats / 记录上限 / reindex");
  const stats = service.stats("chat.message");
  // m6 已删，余下 4 条都带向量
  check("stats 有 embedded 计数", stats[0]?.embedded === 4, JSON.stringify(stats[0]));
  check("stats 有体积估算", (stats[0]?.approxBytes ?? 0) > 0);

  await service.ensureCollection("limited", { dim: 4, maxRecords: 3 });
  for (let i = 0; i < 6; i++) {
    await service.upsert({
      collection: "limited",
      docId: `k${i}`,
      content: `第 ${i} 条测试内容`,
      metadata: { index: i },
    });
  }
  check("超出上限自动淘汰", service.count("limited") === 3, String(service.count("limited")));
  const keptIds = [0, 1, 2, 3, 4, 5]
    .map((i) => `k${i}`)
    .filter((id) => service.get("limited", id) !== undefined);
  check("淘汰的是最旧记录", keptIds.join(",") === "k3,k4,k5", keptIds.join(","));

  const reindexed = await service.reindex("limited");
  check("reindex 覆盖全部记录", reindexed.total === 3 && reindexed.embedded === 3, JSON.stringify(reindexed));
  check("reindex 无失败", reindexed.failed === 0);

  section("11. 降级：模型不可用时仅关键词");
  // 配置关闭：完全不加载模型
  const offDir = mkdtempSync(path.join(tmpdir(), "mioku-vector-off-"));
  const off = harness(offDir, { embed: false });
  check("配置关闭时状态为 disabled", off.status().phase === "disabled", JSON.stringify(off.status()));

  // 模型加载失败：降级但状态可观测
  const failDir = mkdtempSync(path.join(tmpdir(), "mioku-vector-fail-"));
  const broken = harness(failDir, { model: fakeModel({ fail: true }) });
  await new Promise((resolve) => setTimeout(resolve, 10));
  const failedState = broken.status();
  check("模型失败时状态为 failed", failedState.phase === "failed", JSON.stringify(failedState));
  check("失败原因可读", failedState.reason === "测试用假模型故意失败", String(failedState.reason));
  await broken.ensureCollection("notes", { dim: 0 });
  const brokenWrite = await broken.upsertMany(
    CORPUS.map((item) => ({
      collection: "notes",
      docId: item.docId,
      content: item.content,
      metadata: { groupId: item.groupId },
    })),
  );
  check("模型失败时仍写入文本", brokenWrite.results.length === 6, JSON.stringify(brokenWrite.failed));
  const degraded = await broken.search({ collection: "notes", query: "退群", topK: 3 });
  check(
    "模型失败时关键词检索仍可用",
    degraded[0]?.docId === "m5",
    JSON.stringify(degraded.map((h) => h.docId)),
  );
  check("模型失败时没有稠密分", degraded[0]?.denseScore === undefined);
  const degradedDense = await broken.search({
    collection: "notes",
    query: "退群",
    mode: "dense",
    topK: 3,
  });
  check("模型失败时 dense 模式返回空", degradedDense.length === 0);
  let reindexError = "";
  try {
    await broken.reindex("notes");
  } catch (error) {
    reindexError = String(error);
  }
  check("模型失败时 reindex 明确报错", reindexError.includes("不可用"), reindexError);
  broken.dispose();
  off.dispose();

  section("12. 模型未就绪时先写文本，就绪后补向量");
  const lateDir = mkdtempSync(path.join(tmpdir(), "mioku-vector-late-"));
  const late = harness(lateDir, { model: fakeModel({ fail: true }) });
  await late.ensureCollection("late", { dim: 0 });
  const before = await late.upsertMany([
    { collection: "late", docId: "a", content: "今天加班好累", metadata: { groupId: "g" } },
  ]);
  check("模型未就绪时写入不报错", before.results.length === 1);
  check("模型未就绪时该条未向量化", before.results[0]?.embedded === false);
  late.dispose();

  section("13. 状态订阅");
  const subDir = mkdtempSync(path.join(tmpdir(), "mioku-vector-sub-"));
  const sub = harness(subDir);
  await new Promise((resolve) => setTimeout(resolve, 10));
  const seen: string[] = [];
  const unsubscribe = sub.onStatus((state) => seen.push(state.phase));
  check("订阅时立刻回调当前状态", seen.length === 1, JSON.stringify(seen));
  check("模型就绪后状态为 ready", sub.status().ready, JSON.stringify(sub.status()));
  unsubscribe();
  const afterUnsub = seen.length;
  sub.onStatus(() => {});
  check("取消订阅后不再收到回调", seen.length === afterUnsub, JSON.stringify(seen));
  sub.dispose();

  section("14. 空集合与未知 collection");
  const empty = await service.search({ collection: "does-not-exist", query: "任意", topK: 3 });
  check("未知 collection 返回空", empty.length === 0);
  const emptyQuery = await service.search({ collection: "chat.message", query: "", topK: 3 });
  check("空查询不报错", Array.isArray(emptyQuery));

  service.dispose();

  console.log(`\n${failures === 0 ? "全部通过" : "存在失败"}：${checks - failures}/${checks}`);
  rmSync(dir, { recursive: true, force: true });
  rmSync(path.join(tmpdir(), "mioku-vector-sparse-"), { recursive: true, force: true });
  if (failures > 0) process.exit(1);
}

await main();
