import type { Metric } from "../types";

interface SqlValue {
  sql: string;
  params: Array<string | number>;
}

const TRUE: SqlValue = { sql: "1", params: [] };

/** 可以直接映射到列、无需走 json_extract 的顶层字段 */
const TOP_LEVEL_COLUMNS: Record<string, string> = {
  docId: "doc_id",
  createdAt: "created_at",
  updatedAt: "updated_at",
};

/** metadata 里建有生成列 + 索引的键；由 store 探测后传入，缺列时自动退回 json_extract */
export type FilterColumns = ReadonlySet<string>;

/** 由 store 维护的生成列，命中时走索引 */
const GENERATED_COLUMNS: Record<string, string> = {
  groupId: "group_id",
  userId: "user_id",
};

/** 元数据键写入 JSON path，单引号与引号需剔除 */
function escapeKey(key: string): string {
  return key.replace(/['"\\]/g, "");
}

function fieldExpr(key: string, columns?: FilterColumns): string {
  const column = TOP_LEVEL_COLUMNS[key];
  if (column) return column;
  const generated = columns?.has(key) ? GENERATED_COLUMNS[key] : undefined;
  if (generated) return generated;
  return `json_extract(metadata, '$."${escapeKey(key)}"')`;
}

function toParam(value: string | number | boolean): string | number {
  if (typeof value === "boolean") return value ? 1 : 0;
  return value;
}

function combine(parts: SqlValue[], joiner: "AND" | "OR"): SqlValue {
  if (parts.length === 0) return TRUE;
  return {
    sql: parts.map((part) => `(${part.sql})`).join(` ${joiner} `),
    params: parts.flatMap((part) => part.params),
  };
}

function compileNode(filter: unknown, columns?: FilterColumns): SqlValue {
  if (!filter || typeof filter !== "object") return TRUE;
  const node = filter as Record<string, unknown>;

  if (Array.isArray(node.and)) {
    return combine(node.and.map((item) => compileNode(item, columns)), "AND");
  }
  if (Array.isArray(node.or)) {
    return combine(node.or.map((item) => compileNode(item, columns)), "OR");
  }

  if (node.eq && typeof node.eq === "object") {
    const parts: SqlValue[] = [];
    for (const [key, value] of Object.entries(node.eq as Record<string, unknown>)) {
      if (value === undefined || value === null) continue;
      parts.push({
        sql: `${fieldExpr(key, columns)} = ?`,
        params: [toParam(value as string | number | boolean)],
      });
    }
    return combine(parts, "AND");
  }

  if (node.in && typeof node.in === "object") {
    const parts: SqlValue[] = [];
    for (const [key, values] of Object.entries(node.in as Record<string, unknown>)) {
      if (!Array.isArray(values) || values.length === 0) continue;
      parts.push({
        sql: `${fieldExpr(key, columns)} IN (${values.map(() => "?").join(", ")})`,
        params: values.map((value) => toParam(value as string | number | boolean)),
      });
    }
    return combine(parts, "AND");
  }

  if (node.range && typeof node.range === "object") {
    const parts: SqlValue[] = [];
    for (const [key, bounds] of Object.entries(node.range as Record<string, unknown>)) {
      if (!bounds || typeof bounds !== "object") continue;
      const { gte, lte } = bounds as { gte?: number; lte?: number };
      const expr = fieldExpr(key, columns);
      if (typeof gte === "number") {
        parts.push({ sql: `${expr} >= ?`, params: [gte] });
      }
      if (typeof lte === "number") {
        parts.push({ sql: `${expr} <= ?`, params: [lte] });
      }
    }
    return combine(parts, "AND");
  }

  return TRUE;
}

export function compileFilter(filter: unknown, columns?: FilterColumns): SqlValue {
  const compiled = compileNode(filter, columns);
  return { sql: compiled.sql, params: compiled.params };
}

export function normalizeForMetric(
  vector: Float32Array,
  metric: Metric,
): Float32Array {
  if (metric !== "cosine") return vector;
  let sum = 0;
  for (let i = 0; i < vector.length; i++) sum += vector[i] * vector[i];
  const norm = Math.sqrt(sum);
  if (norm === 0 || !Number.isFinite(norm)) return vector;
  const out = new Float32Array(vector.length);
  for (let i = 0; i < vector.length; i++) out[i] = vector[i] / norm;
  return out;
}
