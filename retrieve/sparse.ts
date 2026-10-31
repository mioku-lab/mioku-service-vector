import type { SparseHit, StoredRecord } from "../types";
import { toMatchQuery } from "../utils/tokenize";
import { FTS_TOKENS_COLUMN } from "../store/schema";
import type { RecordStore } from "../store/records";
import type { VectorStore } from "../store/schema";

type SqlRow = Record<string, unknown>;

function rowNumber(row: SqlRow, key: string, fallback = 0): number {
  const value = Number(row?.[key]);
  return Number.isFinite(value) ? value : fallback;
}

/**
 * BM25 稀疏检索。
 *
 * bm25() 分数只在本 collection 内可比，融合阶段按 collection 各自归一化。
 */
export function sparseSearch(
  store: VectorStore,
  records: RecordStore,
  query: {
    collection: string;
    text: string;
    whereSql: string;
    params: Array<string | number>;
    limit: number;
  },
): SparseHit[] {
  if (!store.ftsAvailable) return [];
  const match = toMatchQuery(query.text);
  if (!match) return [];

  const table = store.ftsTableName(query.collection);
  let rows: SqlRow[];
  try {
    rows = store.db
      .query(
        `SELECT r.id AS id, bm25(${table}) AS score
           FROM ${table}
           JOIN records r ON r.id = ${table}.rowid
          WHERE ${table}.${FTS_TOKENS_COLUMN} MATCH ?
            AND r.collection = ?
            AND (${query.whereSql})
          ORDER BY score ASC, r.id ASC
          LIMIT ?`,
      )
      .all(match, query.collection, ...query.params, query.limit) as SqlRow[];
  } catch {
    // 索引与记录不同步时 MATCH 可能失败，退化为无稀疏结果
    return [];
  }

  if (rows.length === 0) return [];
  const ids = rows.map((row) => rowNumber(row, "id"));
  const recordMap = records.getRecordsByIds(ids);
  const hits: SparseHit[] = [];
  for (let rank = 0; rank < rows.length; rank++) {
    const record: StoredRecord | undefined = recordMap.get(ids[rank]);
    if (!record) continue;
    // bm25() 返回负分，越小越相关；翻正后越大越相关
    hits.push({
      record,
      score: -rowNumber(rows[rank], "score"),
      rank: rank + 1,
    });
  }
  return hits;
}
