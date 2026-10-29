/**
 * Turn state containing LangChain messages into the JSON Drex reads.
 *
 * Drex rejects media inside `state`. Images found in messages are moved to the
 * request's `images` (replaced in the text by "[Image N]"), replaced by a
 * placeholder, or refused, depending on the caller. Other media (audio, video,
 * files) is replaced by a placeholder or refused. Nothing is dropped silently.
 */
import { AIMessage, BaseMessage, ChatMessage, ToolMessage } from "@langchain/core/messages";
import type { ImageInput } from "nace-sdk";

import { NaceError, NaceImageModelError, NaceMediaError } from "./errors.js";
import { isImageBlock, toDrexImage } from "./images.js";

/** A value legal anywhere below the root. `Date` becomes an ISO string. */
export type StateValue =
  | string
  | number
  | boolean
  | null
  | Date
  | BaseMessage
  | StateValue[]
  | { [key: string]: StateValue };

/**
 * What to evaluate: text, a JSON object or array, a LangChain message, or
 * messages anywhere inside objects and arrays. Plain `{ role, content }`
 * objects are read as messages too.
 */
export type State = string | BaseMessage | StateValue[] | { [key: string]: StateValue };

export type ImageMode = "lift" | "placeholder" | "error";
export type MediaMode = "placeholder" | "error";

/** Roles LangChain accepts in a `{ role, content }` object used in place of a message. */
const MESSAGE_ROLES = new Set([
  "user",
  "human",
  "assistant",
  "ai",
  "system",
  "developer",
  "tool",
  "function",
]);

const MEDIA_TYPES = new Set([
  "audio",
  "input_audio",
  "video",
  "file",
  "document",
  "input_file",
  "media",
]);

/** Content blocks that repeat a message's `tool_calls`. */
const TOOL_CALL_BLOCKS = new Set(["tool_use", "tool_call", "function_call"]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (!isRecord(value)) return false;
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

/** Whether a plain object is a `{ role, content: [...] }` message. */
export function isDictMessage(value: unknown): value is { role: string; content: unknown[] } {
  return (
    isPlainObject(value) &&
    typeof value.role === "string" &&
    MESSAGE_ROLES.has(value.role) &&
    Array.isArray(value.content)
  );
}

function roleFor(message: BaseMessage): string {
  switch (message.getType()) {
    case "human":
      return "user";
    case "ai":
      return "assistant";
    case "system":
      return "system";
    case "tool":
      return "tool";
    case "function":
      return "function";
    case "generic":
      if (ChatMessage.isInstance(message)) return message.role;
      break;
    default:
      break;
  }
  throw new NaceError(`Unsupported message type in state: "${message.getType()}".`);
}

function json(value: unknown): string {
  try {
    return JSON.stringify(value) ?? "{}";
  } catch {
    // The native message names the offending property, which is caller data.
    throw new NaceError("Tool-call arguments in state could not be serialized.");
  }
}

class Encoder {
  readonly lifted: ImageInput[] = [];
  private readonly ancestors = new Set<object>();

  constructor(
    private readonly images: ImageMode,
    private readonly media: MediaMode,
    private readonly firstIndex: number,
  ) {}

  value(value: unknown): unknown {
    if (value === null || typeof value === "string" || typeof value === "boolean") return value;
    if (typeof value === "number") {
      if (!Number.isFinite(value)) throw new NaceError("State numbers must be finite.");
      return value;
    }
    if (value instanceof Date) return value.toISOString();
    if (typeof value !== "object") {
      throw new NaceError(`Unsupported state value of type ${typeof value}.`);
    }
    if (this.ancestors.has(value)) {
      throw new NaceError("Circular reference detected in state.");
    }
    this.ancestors.add(value);
    try {
      if (BaseMessage.isInstance(value)) return this.message(value);
      if (Array.isArray(value)) return value.map((item) => this.value(item));
      if (!isPlainObject(value)) {
        const name = (value as object).constructor?.name ?? "object";
        throw new NaceError(`Unsupported state value: ${name}.`);
      }
      if (isDictMessage(value)) return this.dictMessage(value);
      const out: Record<string, unknown> = Object.create(null);
      for (const [key, item] of Object.entries(value)) {
        if (item === undefined) continue;
        out[key] = this.value(item);
      }
      return { ...out };
    } finally {
      this.ancestors.delete(value);
    }
  }

  private message(message: BaseMessage): Record<string, unknown> {
    const out: Record<string, unknown> = { role: roleFor(message) };
    const toolCalls = AIMessage.isInstance(message) ? (message.tool_calls ?? []) : [];
    out.content = this.content(message.content, toolCalls.length > 0);
    if (toolCalls.length > 0) {
      out.tool_calls = toolCalls.map((call) => ({
        id: call.id ?? null,
        type: "function",
        function: { name: call.name, arguments: json(call.args) },
      }));
    }
    if (ToolMessage.isInstance(message)) {
      out.tool_call_id = message.tool_call_id;
      if (message.status === "error") out.status = "error";
    }
    if (message.name) out.name = message.name;
    return out;
  }

  private dictMessage(message: { role: string; content: unknown[] }): Record<string, unknown> {
    // Only the content blocks are normalized; the other keys are the caller's data.
    const out: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(message)) {
      if (item === undefined) continue;
      const value = key === "content" ? this.content(message.content, false) : this.value(item);
      Object.defineProperty(out, key, {
        value,
        enumerable: true,
        writable: true,
        configurable: true,
      });
    }
    return out;
  }

  private content(content: unknown, hasToolCalls: boolean): unknown {
    if (!Array.isArray(content)) return this.value(content);
    const parts: unknown[] = [];
    for (const block of content) {
      if (typeof block === "string") {
        parts.push({ type: "text", text: block });
        continue;
      }
      if (!isRecord(block)) {
        parts.push(this.value(block));
        continue;
      }
      const kind = block.type;
      if (kind === "text") {
        parts.push({ type: "text", text: typeof block.text === "string" ? block.text : "" });
      } else if (isImageBlock(block)) {
        parts.push(this.image(block));
      } else if (typeof kind === "string" && MEDIA_TYPES.has(kind)) {
        parts.push(this.otherMedia(kind));
      } else if (hasToolCalls && typeof kind === "string" && TOOL_CALL_BLOCKS.has(kind)) {
        // Already sent in `tool_calls`.
      } else {
        // An unrecognized block is still context: keep it as JSON.
        parts.push(this.value(block));
      }
    }
    return parts;
  }

  private image(block: Record<string, unknown>): unknown {
    if (this.images === "placeholder") return { type: "text", text: "[image omitted]" };
    if (this.images === "error") {
      throw new NaceImageModelError(
        "The state holds an image, but this model reads text only. " +
          'Use an image model such as model: "drex-max".',
      );
    }
    this.lifted.push(toDrexImage(block));
    return { type: "text", text: `[Image ${this.firstIndex + this.lifted.length - 1}]` };
  }

  private otherMedia(kind: string): unknown {
    if (this.media === "error") {
      throw new NaceMediaError(
        `Drex reads text and images only; the state holds a \`${kind}\` content block. ` +
          "Remove it, or describe it in text.",
      );
    }
    return { type: "text", text: `[${kind} omitted]` };
  }
}

/**
 * Return the state as JSON plus the images lifted out of it.
 *
 * @param images What to do with images in messages: move them to `images`
 *   ("lift"), replace them with "[image omitted]", or throw.
 * @param media What to do with audio, video and files: replace or throw.
 * @param firstIndex Number of the first lifted image, after any explicit images.
 */
export function encodeState(
  state: unknown,
  options: { images: ImageMode; media?: MediaMode; firstIndex?: number },
): { state: unknown; images: ImageInput[] } {
  if (
    state === null ||
    state === undefined ||
    typeof state === "number" ||
    typeof state === "boolean"
  ) {
    throw new NaceError("State must be a string, a JSON object or array, or LangChain messages.");
  }
  const encoder = new Encoder(options.images, options.media ?? "error", options.firstIndex ?? 1);
  return { state: encoder.value(state), images: encoder.lifted };
}

/** The text of a message (or `{ role, content }` object), with non-text blocks left out. */
export function messageText(message: unknown): string {
  if (BaseMessage.isInstance(message)) return message.text;
  if (!isRecord(message)) return "";
  const content = message.content;
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .map((block) => {
      if (typeof block === "string") return block;
      if (isRecord(block) && block.type === "text" && typeof block.text === "string") {
        return block.text;
      }
      return "";
    })
    .filter((text) => text !== "")
    .join("\n");
}
