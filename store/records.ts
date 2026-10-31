import type {
  CollectionInfo,
  CollectionOptions,
  Metric,
  StoredRecord,
  VectorMetadata,
} from "../types";
import { normalizeForMetric } from "./filter";
import { VectorStore } from "./schema";

type SqlRow = Record<string, unknown>;

const DEFAULT_MAX_CONTENT_CHARS = 4000;

function rowNumber(row: SqlRow | null | undefined, key: string, fallback = 0): number {
  const value = Number(row?.[key]);
  return Number.isFinite(value) ? value : fallback;
}

function rowString(row: SqlRow | null | undefined, key: string, fallback = ""): string {
  const value = row?.[key];
  return typeof value === "string" ? value : fallback;
}

function rowId(row: SqlRow | null | undefined, key: string, fallback = ""): string {
  const value = row?.[key];
  if (value == null) return fallback;
  const text = String(value);
  return text.length > 0 ? text : fallback;
}

function parseJson(text: string, fallback: VectorMetadata): VectorMetadata {
  try {
    const parsed = JSON.parse(text);
    return parsed && typeof parsed === "object" ? (parsed as VectorMetadata) : fallback;
  } catch {
    return fallback;
  }
}

export function vectorToBlob(vector: Float32Array): Uint8Array {
  return new Uint8Array(vector.buffer, vector.byteOffset, vector.byteLength);
}

export function blobToVector(blob: unknown, dim: number): Float32Array | undefined {
  if (!blob || dim <= 0) return undefined;
  const bytes =
    blob instanceof Uint8Array
      ? blob
      : blob instanceof ArrayBuffer
        ? new Uint8Array(blob)
        : undefined;
  if (!bytes || bytes.byteLength !== dim * 4) return undefined;
  // 复制一份，避免与 SQLite 的复用缓冲共享内存
  const copy = new Uint8Array(bytes);
  return new Float32Array(copy.buffer, 0, dim);
}

function toCollectionInfo(row: SqlRow, records: number): CollectionInfo {
  return {
    name: rowString(row, "name"),
    dim: rowNumber(row, "dim"),
    embedModel: rowString(row, "embed_model"),
    metric: rowString(row, "metric", "cosine") === "dot" ? "dot" : "cosine",
    maxContentChars: rowNumber(row, "max_content_chars", DEFAULT_MAX_CONTENT_CHARS),
    maxRecords: rowNumber(row, "max_records"),
    metadata: parseJson(rowString(row, "metadata", "{}"), {}),
    records,
    createdAt: rowNumber(row, "created_at"),
    updatedAt: rowNumber(row, "updated_at"),
  };
}

function toRecord(row: SqlRow, withVector: boolean): StoredRecord {
  const dim = rowNumber(row, "dim");
  const record: StoredRecord = {
    id: rowNumber(row, "id"),
    collection: rowString(row, "collection"),
    docId: rowId(row, "doc_id"),
    content: rowString(row, "content"),
    metadata: parseJson(rowString(row, "metadata", "{}"), {}),
    embedHash: rowString(row, "embed_hash"),
    dim,
    createdAt: rowNumber(row, "created_at"),
    updatedAt: rowNumber(row, "updated_at"),
  };
  if (withVector) {
    const vector = blobToVector(row.vector, dim);
    if (vector) record.vector = vector;
  }
  return record;
}

export interface ExistingRecord {
  id: number;
  embedHash: string;
  metadata: string;
  dim: number;
}

/** records 与 collections 两张表的读写门面 */
export class RecordStore {
  constructor(private readonly store: VectorStore) {}

    get vectorStore(): VectorStore {
    return this.store;
  }

  private get db() {
    return this.store.db;
  }

  getCollectionRow(name: string): SqlRow | null {
    return this.db
      .query("SELECT * FROM collections WHERE name = ?")
      .get(name) as SqlRow | null;
  }

  listCollectionRows(): SqlRow[] {
    return this.db
      .query("SELECT * FROM collections ORDER BY name ASC")
      .all() as SqlRow[];
  }

  countRecords(name: string): number {
    const row = this.db
      .query("SELECT COUNT(*) AS count FROM records WHERE collection = ?")
      .get(name) as SqlRow | null;
    return rowNumber(row, "count");
  }

  countEmbedded(name: string): number {
    const row = this.db
      .query(
        "SELECT COUNT(*) AS count FROM records WHERE collection = ? AND vector IS NOT NULL",
      )
      .get(name) as SqlRow | null;
    return rowNumber(row, "count");
  }

  approxBytes(name: string): number {
    const row = this.db
      .query(
        `SELECT COALESCE(SUM(
           LENGTH(CAST(content AS BLOB)) + LENGTH(CAST(metadata AS BLOB))
           + COALESCE(LENGTH(vector), 0)
         ), 0) AS bytes
         FROM records WHERE collection = ?`,
      )
      .get(name) as SqlRow | null;
    return rowNumber(row, "bytes");
  }

  toInfo(row: SqlRow): CollectionInfo {
    return toCollectionInfo(row, this.countRecords(rowString(row, "name")));
  }

  insertCollection(name: string, options: CollectionOptions): CollectionInfo {
    const now = Date.now();
    this.db.run(
      `INSERT INTO collections
         (name, dim, embed_model, metric, max_content_chars, max_records, metadata, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        name,
        Math.max(0, Math.trunc(options.dim ?? 0)),
        String(options.embedModel ?? ""),
        options.metric ?? "cosine",
        Math.max(1, Math.trunc(options.maxContentChars ?? DEFAULT_MAX_CONTENT_CHARS)),
        Math.max(0, Math.trunc(options.maxRecords ?? 0)),
        JSON.stringify(options.metadata ?? {}),
        now,
        now,
      ],
    );
    return this.toInfo(this.getCollectionRow(name) as SqlRow);
  }

  updateCollection(
    name: string,
    patch: {
      dim?: number;
      embedModel?: string;
      metric?: Metric;
      maxContentChars?: number;
      maxRecords?: number;
      metadata?: VectorMetadata;
    },
  ): CollectionInfo {
    const sets: string[] = [];
    const values: Array<string | number> = [];
    if (patch.dim !== undefined) {
      sets.push("dim = ?");
      values.push(Math.max(0, Math.trunc(patch.dim)));
    }
    if (patch.embedModel !== undefined) {
      sets.push("embed_model = ?");
      values.push(patch.embedModel);
    }
    if (patch.metric !== undefined) {
      sets.push("metric = ?");
      values.push(patch.metric);
    }
    if (patch.maxContentChars !== undefined) {
      sets.push("max_content_chars = ?");
      values.push(Math.max(1, Math.trunc(patch.maxContentChars)));
    }
    if (patch.maxRecords !== undefined) {
      sets.push("max_records = ?");
      values.push(Math.max(0, Math.trunc(patch.maxRecords)));
    }
    if (patch.metadata !== undefined) {
      sets.push("metadata = ?");
      values.push(JSON.stringify(patch.metadata));
    }
    if (sets.length > 0) {
      sets.push("updated_at = ?");
      values.push(Date.now(), name);
      this.db.run(
        `UPDATE collections SET ${sets.join(", ")} WHERE name = ?`,
        values,
      );
    }
    return this.toInfo(this.getCollectionRow(name) as SqlRow);
  }

  dropCollection(name: string): void {
    this.store.transaction(() => {
      this.db.run("DELETE FROM records WHERE collection = ?", [name]);
      this.db.run("DELETE FROM collections WHERE name = ?", [name]);
    });
    this.store.dropFts(name);
  }

  findExisting(collection: string, docId: string): ExistingRecord | null {
    const row = this.db
      .query(
        "SELECT id, embed_hash, metadata, dim FROM records WHERE collection = ? AND doc_id = ?",
      )
      .get(collection, docId) as SqlRow | null;
    if (!row) return null;
    return {
      id: rowNumber(row, "id"),
      embedHash: rowString(row, "embed_hash"),
      metadata: rowString(row, "metadata", "{}"),
      dim: rowNumber(row, "dim"),
    };
  }

  getRecord(collection: string, docId: string): StoredRecord | undefined {
    const row = this.db
      .query("SELECT * FROM records WHERE collection = ? AND doc_id = ?")
      .get(collection, docId) as SqlRow | null;
    return row ? toRecord(row, true) : undefined;
  }

  /** 不负责 FTS 维护，调用方需自行同步索引 */
  write(input: {
    collection: string;
    docId: string;
    content: string;
    metadata: VectorMetadata;
    embedHash: string;
    vector?: Float32Array;
    createdAt?: number;
  }): { id: number; action: "inserted" | "updated" } {
    const now = Date.now();
    const dim = input.vector?.length ?? 0;
    const blob = input.vector ? vectorToBlob(input.vector) : null;
    const metadataJson = JSON.stringify(input.metadata);

    const existing = this.findExisting(input.collection, input.docId);
    if (existing) {
      this.db.run(
        `UPDATE records
           SET content = ?, metadata = ?, embed_hash = ?, dim = ?,
               vector = COALESCE(?, vector), updated_at = ?
         WHERE id = ?`,
        [
          input.content,
          metadataJson,
          input.embedHash,
          dim > 0 ? dim : existing.dim,
          blob,
          now,
          existing.id,
        ],
      );
      return { id: existing.id, action: "updated" };
    }

    const result = this.db
      .query(
        `INSERT INTO records
           (collection, doc_id, content, metadata, embed_hash, dim, vector, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        input.collection,
        input.docId,
        input.content,
        metadataJson,
        input.embedHash,
        dim,
        blob,
        input.createdAt ?? now,
        now,
      );
    return { id: Number(result.lastInsertRowid), action: "inserted" };
  }

  updateVector(id: number, embedHash: string, vector: Float32Array): void {
    this.db.run(
      "UPDATE records SET embed_hash = ?, dim = ?, vector = ?, updated_at = ? WHERE id = ?",
      [
        embedHash,
        vector.length,
        vectorToBlob(vector),
        Date.now(),
        id,
      ],
    );
  }

  deleteByDocIds(collection: string, docIds: string[]): number[] {
    if (docIds.length === 0) return [];
    const ids: number[] = [];
    for (let i = 0; i < docIds.length; i += 500) {
      const chunk = docIds.slice(i, i + 500);
      const marks = chunk.map(() => "?").join(", ");
      const rows = this.db
        .query(
          `SELECT id FROM records WHERE collection = ? AND doc_id IN (${marks})`,
        )
        .all(collection, ...chunk) as SqlRow[];
      for (const row of rows) ids.push(rowNumber(row, "id"));
      this.db.run(
        `DELETE FROM records WHERE collection = ? AND doc_id IN (${marks})`,
        [collection, ...chunk],
      );
    }
    return ids;
  }

  deleteByFilter(collection: string, whereSql: string, params: Array<string | number>): number[] {
    const rows = this.db
      .query(`SELECT id FROM records WHERE collection = ? AND ${whereSql}`)
      .all(collection, ...params) as SqlRow[];
    const ids = rows.map((row) => rowNumber(row, "id"));
    if (ids.length > 0) {
      for (let i = 0; i < ids.length; i += 500) {
        const chunk = ids.slice(i, i + 500);
        const marks = chunk.map(() => "?").join(", ");
        this.db.run(`DELETE FROM records WHERE id IN (${marks})`, chunk);
      }
    }
    return ids;
  }

  clearCollection(collection: string): number[] {
    const rows = this.db
      .query("SELECT id FROM records WHERE collection = ?")
      .all(collection) as SqlRow[];
    const ids = rows.map((row) => rowNumber(row, "id"));
    this.db.run("DELETE FROM records WHERE collection = ?", [collection]);
    return ids;
  }

  countByFilter(
    collection: string,
    whereSql: string,
    params: Array<string | number>,
  ): number {
    const row = this.db
      .query(
        `SELECT COUNT(*) AS count FROM records WHERE collection = ? AND ${whereSql}`,
      )
      .get(collection, ...params) as SqlRow | null;
    return rowNumber(row, "count");
  }

  listCandidates(
    collection: string,
    whereSql: string,
    params: Array<string | number>,
    limit: number,
  ): StoredRecord[] {
    const rows = this.db
      .query(
        `SELECT * FROM records WHERE collection = ? AND ${whereSql}
         ORDER BY updated_at DESC LIMIT ?`,
      )
      .all(collection, ...params, limit) as SqlRow[];
    return rows.map((row) => toRecord(row, true));
  }

  /** 仅取 id、docId 与向量，用于暴力检索；避免构建完整记录 */
  listVectors(
    collection: string,
    whereSql: string,
    params: Array<string | number>,
    limit: number,
  ): Array<{ id: number; docId: string; vector: Float32Array }> {
    const rows = this.db
      .query(
        `SELECT id, doc_id, dim, vector FROM records
         WHERE collection = ? AND ${whereSql} AND vector IS NOT NULL
         ORDER BY updated_at DESC, id DESC LIMIT ?`,
      )
      .all(collection, ...params, limit) as SqlRow[];
    const out: Array<{ id: number; docId: string; vector: Float32Array }> = [];
    for (const row of rows) {
      const vector = blobToVector(row.vector, rowNumber(row, "dim"));
      if (vector) {
        out.push({
          id: rowNumber(row, "id"),
          docId: rowId(row, "doc_id"),
          vector,
        });
      }
    }
    return out;
  }

  getRecordsByIds(ids: number[]): Map<number, StoredRecord> {
    const map = new Map<number, StoredRecord>();
    for (let i = 0; i < ids.length; i += 500) {
      const chunk = ids.slice(i, i + 500);
      const marks = chunk.map(() => "?").join(", ");
      const rows = this.db
        .query(`SELECT * FROM records WHERE id IN (${marks})`)
        .all(...chunk) as SqlRow[];
      for (const row of rows) {
        const record = toRecord(row, true);
        map.set(record.id, record);
      }
    }
    return map;
  }

  enforceLimit(collection: string, maxRecords: number): number[] {
    if (maxRecords <= 0) return [];
    const total = this.countRecords(collection);
    if (total <= maxRecords) return [];
    // 同毫秒写入时 updated_at 无法定序，用自增 id 兜底
    const rows = this.db
      .query(
        `SELECT id FROM records WHERE collection = ?
         ORDER BY updated_at DESC, id DESC LIMIT -1 OFFSET ?`,
      )
      .all(collection, maxRecords) as SqlRow[];
    const ids = rows.map((row) => rowNumber(row, "id"));
    if (ids.length === 0) return [];
    for (let i = 0; i < ids.length; i += 500) {
      const chunk = ids.slice(i, i + 500);
      const marks = chunk.map(() => "?").join(", ");
      this.db.run(`DELETE FROM records WHERE id IN (${marks})`, chunk);
    }
    return ids;
  }

    prepare(vector: readonly number[], metric: Metric): Float32Array {
    return normalizeForMetric(Float32Array.from(vector), metric);
  }
}

export { toRecord };
