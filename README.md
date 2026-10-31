# mioku-service-vector

为插件提供向量存储与混合检索：把文本变成向量存下来，再用「语义 + 关键词」两路检索找回来。
面向长期记忆、群员轮廓、群事件回溯这类场景。

向量化完全在本机跑（ONNX），**不需要任何 API Key**。模型文件按需下载到 `data/vector/models`，
数据落在 `data/vector/vector.db`。

## 插件中声明依赖

```json
{
  "mioku": {
    "services": ["vector"]
  }
}
```

## 取服务

```ts
import { VectorService } from "mioku-service-vector";
import { getService } from "mioku";

const vectors = getService(ctx, VectorService); // 可选依赖，缺失时自己降级
if (!vectors) return;
```

## 快速开始

```ts
await vectors.ensureCollection("chat.memory");

await vectors.upsertMany([
  {
    collection: "chat.memory",
    docId: "msg-1001",
    content: "小明上周说他妈住院了，最近压力很大",
    metadata: { groupId: "123", userId: "456", ts: Date.now() },
  },
]);

const hits = await vectors.search({
  collection: "chat.memory",
  query: "小明最近怎么了",
  topK: 5,
  filter: { eq: { groupId: "123" } },
});
// hits[0].content -> "小明上周说他妈住院了，最近压力很大"
```

## 模型状态

模型在服务启动后**后台加载**，不阻塞服务就绪。加载完成前写入只落文本、检索只用关键词；
就绪后自动恢复向量能力，此前写入的文本可以用 `reindex` 补向量。

```ts
vectors.status(); // { phase, message, ready, reason? }
vectors.onStatus((state) => {}); // 订阅变化，注册时立刻回调一次
```

`phase` 取值：`pending` 加载中 · `ready` 就绪 · `failed` 加载失败（已降级）· `disabled` 配置关闭。

## 导出接口

| 方法                                                                                     | 作用                      |
|----------------------------------------------------------------------------------------|-------------------------|
| `ensureCollection(name, options?)`                                                     | 幂等建库；维度冲突时抛错            |
| `listCollections()` / `getCollection(name)` / `stats(name?)`                           | 元信息与用量                  |
| `upsert(input)` / `upsertMany(inputs, options?)`                                       | 按 `docId` 覆盖写入，内容未变跳过推理 |
| `search(query)`                                                                        | 混合检索（稠密 + BM25 融合）      |
| `searchDense(query)` / `searchSparse(query)`                                           | 单路检索，便于调试与对照            |
| `get(collection, docId)` / `count(collection, filter?)`                                | 单条读取与计数                 |
| `remove(collection, docIds)` / `removeWhere(collection, filter)` / `clear(collection)` | 删除                      |
| `reindex(collection, options?)`                                                        | 换模型或换维度后重建向量            |
| `dropCollection(name)`                                                                 | 删除整个 collection         |
| `status()` / `onStatus(fn)`                                                            | 模型加载状态                  |

`defineVectorCollection<TMeta>(name)` 生成带类型的 collection 引用，业务 metadata 形状由调用方声明。

## 配置

见 `config.md`，只有四项：启用开关、模型选择、自动下载开关、下载源。

下载源默认 `https://hf-mirror.com`（国内直连 HuggingFace 不稳定）。首次下载失败会按
指数退避自动重试，仍失败则降级为关键词检索，并在日志里给出原因。

## 依赖

- `@huggingface/transformers`（含 ONNX Runtime）—— **安装体积约 500MB**，其中绝大部分是
  `onnxruntime-node` / `onnxruntime-web` 的多平台原生二进制。这是换取「不依赖外部服务、
  不需要 API Key」的代价。
- `bun:sqlite` —— 框架自带。

向量检索是 TypeScript 暴力点积：10000 条 × 768 维实测约 10ms
