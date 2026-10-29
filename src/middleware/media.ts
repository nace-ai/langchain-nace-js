/**
 * Prepare agent messages for Drex inside middleware.
 *
 * Middleware must not stop an agent because a conversation holds media the
 * decision model can't read, so media becomes text placeholders here instead
 * of an error (the classifier itself refuses rather than dropping anything).
 */
import { BaseMessage } from "@langchain/core/messages";

import { isImageBlock } from "../images.js";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** An image block whose bytes are inline (data URL or base64), not a remote URL. */
export function isInlineImage(block: unknown): block is Record<string, unknown> {
  if (!isImageBlock(block)) return false;
  if (block.source_type === "base64" || typeof block.base64 === "string") return true;
  if (block.type === "media" && typeof block.data === "string") return true;
  if (isRecord(block.source)) return block.source.type === "base64";
  if (block.data instanceof Uint8Array) return true;
  if (typeof block.data === "string" && block.source_type === undefined && !block.url) return true;
  const imageUrl = block.image_url;
  const url = isRecord(imageUrl) ? imageUrl.url : (imageUrl ?? block.url);
  return typeof url === "string" && url.startsWith("data:");
}

function contentOf(message: unknown): unknown {
  if (BaseMessage.isInstance(message)) return message.content;
  return isRecord(message) ? message.content : undefined;
}

/** The fields a message class needs to rebuild a copy with new content. */
function fieldsOf(message: BaseMessage): Record<string, unknown> {
  const fields: Record<string, unknown> = {};
  for (const key of [
    "name",
    "id",
    "additional_kwargs",
    "response_metadata",
    "tool_calls",
    "invalid_tool_calls",
    "usage_metadata",
    "tool_call_id",
    "status",
    "artifact",
    "role",
  ]) {
    const value = (message as unknown as Record<string, unknown>)[key];
    if (value !== undefined) fields[key] = value;
  }
  return fields;
}

/**
 * A copy of `message` with media replaced by "[... omitted]" text.
 *
 * Inline images are kept when `keepImages` is set; remote image URLs, audio,
 * video and files are always replaced. Works on message objects and on plain
 * `{ role, content }` objects.
 */
export function withoutMedia<T>(message: T, options: { keepImages: boolean }): T {
  const content = contentOf(message);
  if (!Array.isArray(content)) return message;
  let changed = false;
  const blocks = content.map((block) => {
    if (typeof block === "string" || !isRecord(block)) return block;
    const kind = block.type;
    if (kind === "text") return block;
    if (isImageBlock(block)) {
      if (options.keepImages && isInlineImage(block)) return block;
      changed = true;
      return { type: "text", text: "[image omitted]" };
    }
    if (typeof kind === "string" && /audio|video|file|document|media/.test(kind)) {
      changed = true;
      return { type: "text", text: `[${kind} omitted]` };
    }
    return block;
  });
  if (!changed) return message;
  if (BaseMessage.isInstance(message)) {
    const Ctor = message.constructor as new (fields: Record<string, unknown>) => BaseMessage;
    return new Ctor({ ...fieldsOf(message), content: blocks }) as T;
  }
  return { ...(message as Record<string, unknown>), content: blocks } as T;
}

/** The newest inline image blocks in `messages`, oldest first, at most `limit`. */
export function latestImages(messages: unknown[], limit: number): Record<string, unknown>[] {
  const found: Record<string, unknown>[] = [];
  for (let i = messages.length - 1; i >= 0 && found.length < limit; i -= 1) {
    const content = contentOf(messages[i]);
    if (!Array.isArray(content)) continue;
    for (let j = content.length - 1; j >= 0 && found.length < limit; j -= 1) {
      const block = content[j];
      if (isInlineImage(block)) found.push(block);
    }
  }
  return found.reverse();
}

/** Whether a message (object or `{ role, content }`) was written by the user. */
export function isUserMessage(message: unknown): boolean {
  if (BaseMessage.isInstance(message)) return message.getType() === "human";
  return isRecord(message) && (message.role === "user" || message.role === "human");
}
