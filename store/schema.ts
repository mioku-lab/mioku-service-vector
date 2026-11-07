import { Database } from "bun:sqlite";

interface FtsEntry {
  table: string;
}

/** 服务内部使用的 SQLite 句柄，负责建表与 FTS5 索引维护 */
export class VectorStore {
  readonly db: Database;
  private readonly ftsTables = new Map<string, FtsEntry>();
  /** FTS5 不可用时降级为纯稠密检索 */
  readonly ftsAvailable: boolean;
  /** metadata 里真正建好生成列的键，过滤时优先走索引 */
  readonly filterColumns = new Set<string>();

  constructor(dbPath: string) {
    this.db = new Database(dbPath);
    this.db.run("PRAGMA journal_mode = WAL;");
    this.db.run("PRAGMA synchronous = NORMAL;");
    this.migrate();
    this.ftsAvailable = this.probeFts();
  }

  private migrate(): void {
    this.db.run(`
      CREATE TABLE IF NOT EXISTS collections (
        name              TEXT PRIMARY KEY,
        dim               INTEGER NOT NULL DEFAULT 0,
        embed_model       TEXT NOT NULL DEFAULT '',
        metric            TEXT NOT NULL DEFAULT 'cosine',
        max_content_chars INTEGER NOT NULL DEFAULT 4000,
        max_records       INTEGER NOT NULL DEFAULT 0,
        metadata          TEXT NOT NULL DEFAULT '{}',
        created_at        INTEGER NOT NULL,
        updated_at        INTEGER NOT NULL
      );
      CREATE TABLE IF NOT EXISTS records (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        collection TEXT NOT NULL,
        doc_id     TEXT NOT NULL,
        content    TEXT NOT NULL,
        metadata   TEXT NOT NULL DEFAULT '{}',
        embed_hash TEXT NOT NULL DEFAULT '',
        dim        INTEGER NOT NULL DEFAULT 0,
        vector     BLOB,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        UNIQUE (collection, doc_id)
      );
      CREATE INDEX IF NOT EXISTS idx_records_collection
        ON records(collection, updated_at DESC);
    `);
    this.migrateFilterIndexes();
  }

  private migrateFilterIndexes(): void {
    const existing = new Set(
      (
        this.db.query("PRAGMA table_info(records)").all() as Array<{
          name?: string;
        }>
      ).map((row) => String(row.name ?? "")),
    );
    const columns: Array<[column: string, jsonKey: string]> = [
      ["group_id", "groupId"],
      ["user_id", "userId"],
    ];
    for (const [column, jsonKey] of columns) {
      if (existing.has(column)) {
        this.filterColumns.add(jsonKey);
        continue;
      }
      try {
        this.db.run(
          `ALTER TABLE records ADD COLUMN ${column} TEXT GENERATED ALWAYS AS (json_extract(metadata, '$.${jsonKey}')) STORED`,
        );
        this.db.run(
          `CREATE INDEX IF NOT EXISTS idx_records_${column} ON records(collection, ${column})`,
        );
        this.filterColumns.add(jsonKey);
      } catch {
        // 生成列不可用时忽略：过滤仍可走 json_extract
      }
    }
  }

  private probeFts(): boolean {
    try {
      this.db.run(
        "CREATE VIRTUAL TABLE IF NOT EXISTS __fts_probe USING fts5(x, tokenize='unicode61')",
      );
      this.db.run("DROP TABLE IF EXISTS __fts_probe");
      return true;
    } catch {
      return false;
    }
  }

  private tableName(collection: string): string {
    return `fts_${collection.replace(/[^A-Za-z0-9_]/g, "_")}`;
  }

    ftsTableName(collection: string): string {
    return this.ftsTable(collection);
  }

  private ftsTable(collection: string): string {
    const cached = this.ftsTables.get(collection);
    if (cached) return cached.table;
    const table = this.tableName(collection);
    // 不用 external content：本表的 tokens 是派生列，records 里不存在同名列；
    // 普通 FTS5 表支持按 rowid 增删，索引维护由本类负责。
    this.db.run(
      `CREATE VIRTUAL TABLE IF NOT EXISTS ${table} USING fts5(
         tokens,
         tokenize='unicode61'
       )`,
    );
    this.ftsTables.set(collection, { table });
    return table;
  }

  indexRow(collection: string, rowid: number, indexText: string): void {
    if (!this.ftsAvailable) return;
    const table = this.ftsTable(collection);
    this.db.run(`INSERT INTO ${table}(rowid, tokens) VALUES (?, ?)`, [
      rowid,
      indexText,
    ]);
  }

  reindexRow(collection: string, rowid: number, indexText: string): void {
    if (!this.ftsAvailable) return;
    const table = this.ftsTable(collection);
    this.db.run(`DELETE FROM ${table} WHERE rowid = ?`, [rowid]);
    this.db.run(`INSERT INTO ${table}(rowid, tokens) VALUES (?, ?)`, [
      rowid,
      indexText,
    ]);
  }

  removeIndexRows(collection: string, rowids: number[]): void {
    if (!this.ftsAvailable || rowids.length === 0) return;
    const table = this.ftsTable(collection);
    for (let i = 0; i < rowids.length; i += 500) {
      const chunk = rowids.slice(i, i + 500);
      const marks = chunk.map(() => "?").join(", ");
      this.db.run(`DELETE FROM ${table} WHERE rowid IN (${marks})`, chunk);
    }
  }

  dropFts(collection: string): void {
    this.ftsTables.delete(collection);
    this.db.run(`DROP TABLE IF EXISTS ${this.tableName(collection)}`);
  }

  clearFts(collection: string): void {
    if (!this.ftsAvailable) return;
    const table = this.ftsTable(collection);
    this.db.run(`DELETE FROM ${table}`);
  }

  transaction<T>(fn: () => T): T {
    this.db.run("BEGIN");
    try {
      const result = fn();
      this.db.run("COMMIT");
      return result;
    } catch (error) {
      this.db.run("ROLLBACK");
      throw error;
    }
  }

  close(): void {
    this.db.close();
  }
}

export const FTS_TOKENS_COLUMN = "tokens";
