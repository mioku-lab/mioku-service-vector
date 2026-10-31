const CJK =
  /[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af\u31c0-\u31ef]/;
const WORD = /[\p{L}\p{N}]/u;

export function tokenize(text: string): string[] {
  const tokens: string[] = [];
  let word = "";
  let run = "";

  const flushWord = (): void => {
    if (!word) return;
    tokens.push(word.toLowerCase());
    word = "";
  };
  const flushRun = (): void => {
    if (!run) return;
    if (run.length === 1) tokens.push(run);
    else {
      for (let i = 0; i + 2 <= run.length; i++)
        tokens.push(run.slice(i, i + 2));
    }
    run = "";
  };

  for (const ch of text) {
    if (CJK.test(ch)) {
      flushWord();
      run += ch;
    } else if (WORD.test(ch)) {
      flushRun();
      word += ch;
    } else {
      flushRun();
      flushWord();
    }
  }
  flushRun();
  flushWord();
  return tokens;
}

export function toIndexText(text: string): string {
  return tokenize(text).join(" ");
}

/**
 * 查询 → FTS5 MATCH 表达式。
 * 词元之间用 OR，命中越多 BM25 分数越高；返回空串表示查询无有效词元。
 */
export function toMatchQuery(text: string): string {
  const tokens = tokenize(text);
  if (tokens.length === 0) return "";
  return tokens.map((token) => `"${token.replace(/"/g, '""')}"`).join(" OR ");
}
