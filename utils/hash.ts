import { createHash } from "node:crypto";

/** 内容指纹：内容或 embedding 模型任一变都会变 */
export function embedHash(content: string, embedModel: string): string {
  return createHash("sha1")
    .update(embedModel)
    .update("\u0000")
    .update(content)
    .digest("hex")
    .slice(0, 32);
}
