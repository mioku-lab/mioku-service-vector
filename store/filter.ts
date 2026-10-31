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

/** 元数据键写入 JSON path，单引号与引号需剔除 */
function escapeKey(key: string): string {
  return key.replace(/['"\\]/g, "");
}

function fieldExpr(key: string): string {
  const column = TOP_LEVEL_COLUMNS[key];
  if (column) return column;
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

function compileNode(filter: unknown): SqlValue {
  if (!filter || typeof filter !== "object") return TRUE;
  const node = filter as Record<string, unknown>;

  if (Array.isArray(node.and)) {
    return combine(node.and.map(compileNode), "AND");
  }
  if (Array.isArray(node.or)) {
    return combine(node.or.map(compileNode), "OR");
  }

  if (node.eq && typeof node.eq === "object") {
    const parts: SqlValue[] = [];
    for (const [key, value] of Object.entries(node.eq as Record<string, unknown>)) {
      if (value === undefined || value === null) continue;
      parts.push({
        sql: `${fieldExpr(key)} = ?`,
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
        sql: `${fieldExpr(key)} IN (${values.map(() => "?").join(", ")})`,
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
      const expr = fieldExpr(key);
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

export function compileFilter(filter: unknown): SqlValue {
  const compiled = compileNode(filter);
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
