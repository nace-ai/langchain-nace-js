/** Convert images to the entries Drex reads in a request's `images` array. */
import { existsSync, readFileSync, statSync } from "node:fs";
import { type ImageInput, imageInput } from "nace-sdk";

import { NaceImageError } from "./errors.js";

/**
 * An image: base64, a data URL, a file path, bytes (`Buffer`, `Uint8Array`,
 * `ArrayBuffer`), a LangChain image content block, an `image_url` block, a
 * `{ type: "workspace_file", ... }` reference, or an `UploadedFile` from
 * `nace-sdk`. Remote URLs are refused: Drex never fetches them.
 */
export type ImageLike =
  | string
  | Uint8Array
  | ArrayBuffer
  | { type: string; [key: string]: unknown }
  | { workspace_id: string; file_id: string; [key: string]: unknown };

const REMOTE = /^https?:\/\//i;

function remoteUrlError(): NaceImageError {
  // The URL itself stays out of the message: it can be user data or a signed link.
  return new NaceImageError(
    "Drex doesn't fetch image URLs. Download the image and pass its bytes, " +
      "or upload it with `nace-sdk` and pass the uploaded file.",
  );
}

function fromUrl(url: string): string {
  if (REMOTE.test(url)) throw remoteUrlError();
  return url;
}

function fromBase64(data: string, mimeType: unknown): string {
  return typeof mimeType === "string" && mimeType !== "" ? `data:${mimeType};base64,${data}` : data;
}

const SIGNATURES: [number[], string][] = [
  [[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a], "image/png"],
  [[0xff, 0xd8, 0xff], "image/jpeg"],
  [[0x47, 0x49, 0x46, 0x38], "image/gif"],
];

/** A data URL for base64 image bytes, typed from their first bytes. */
export function asDataUrl(image: string): string {
  if (image.startsWith("data:")) return image;
  const head = Buffer.from(image.slice(0, 24), "base64");
  let mime = SIGNATURES.find(([sig]) => sig.every((byte, i) => head[i] === byte))?.[1];
  if (
    mime === undefined &&
    head.subarray(0, 4).toString("latin1") === "RIFF" &&
    head.subarray(8, 12).toString("latin1") === "WEBP"
  ) {
    mime = "image/webp";
  }
  return `data:${mime ?? "application/octet-stream"};base64,${image}`;
}

const BASE64 = /^[A-Za-z0-9+/]+={0,2}$/;

function isBase64(value: string): boolean {
  return value.length > 0 && value.length % 4 === 0 && BASE64.test(value);
}

function isFile(path: string): boolean {
  try {
    return existsSync(path) && statSync(path).isFile();
  } catch {
    return false;
  }
}

function fromSdk(item: Parameters<typeof imageInput>[0]): ImageInput {
  try {
    return imageInput(item);
  } catch (error) {
    throw new NaceImageError(error instanceof Error ? error.message : String(error), {
      cause: error,
    });
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Whether a content block is an image: LangChain's `image` and `image_url`
 * blocks, OpenAI Responses `input_image`, Anthropic `image` with `source`, and
 * Google `media` with an image MIME type.
 */
export function isImageBlock(block: unknown): block is Record<string, unknown> {
  if (!isRecord(block)) return false;
  const kind = block.type;
  if (kind === "image" || kind === "image_url" || kind === "input_image") return true;
  const mime = block.mimeType ?? block.mime_type;
  return kind === "media" && typeof mime === "string" && mime.startsWith("image/");
}

function fromBlock(block: Record<string, unknown>): ImageInput {
  const kind = block.type;
  if (kind === "workspace_file") {
    return { ...block } as unknown as ImageInput;
  }
  if (kind === "image_url") {
    const imageUrl = block.image_url;
    const url = isRecord(imageUrl) ? imageUrl.url : imageUrl;
    if (typeof url === "string") return fromUrl(url);
  }
  if (kind === "input_image") {
    const imageUrl = block.image_url;
    const url = isRecord(imageUrl) ? imageUrl.url : imageUrl;
    if (typeof url === "string") return fromUrl(url);
  }
  if (kind === "media" && typeof block.data === "string") {
    return fromBase64(block.data, block.mimeType ?? block.mime_type);
  }
  if (kind === "image" && isRecord(block.source)) {
    // Anthropic: { type: "image", source: { type: "base64", media_type, data } | { type: "url", url } }
    const source = block.source;
    if (source.type === "base64" && typeof source.data === "string") {
      return fromBase64(source.data, source.media_type);
    }
    if (typeof source.url === "string") return fromUrl(source.url);
  }
  if (kind === "image") {
    const mimeType = block.mimeType ?? block.mime_type;
    if (block.source_type === "base64" && typeof block.data === "string") {
      return fromBase64(block.data, mimeType);
    }
    if (typeof block.base64 === "string") return fromBase64(block.base64, mimeType);
    if (typeof block.data === "string" && block.source_type === undefined && !block.url) {
      return fromBase64(block.data, mimeType);
    }
    if (block.data instanceof Uint8Array) return fromSdk(block.data);
    const url = block.url ?? (block.source_type === "url" ? block.data : undefined);
    if (typeof url === "string") return fromUrl(url);
  }
  // An `UploadedFile` from nace-sdk: the SDK turns it into a workspace_file reference.
  if (typeof block.workspace_id === "string" && typeof block.file_id === "string") {
    return fromSdk(block as unknown as Parameters<typeof imageInput>[0]);
  }
  throw new NaceImageError(`Unsupported image block (type ${JSON.stringify(kind ?? null)}).`);
}

/**
 * One `images` entry for Drex.
 *
 * @throws NaceImageError when the item isn't an image Drex can read.
 */
export function toDrexImage(item: unknown): ImageInput {
  if (typeof item === "string") {
    if (item.startsWith("data:") || REMOTE.test(item)) return fromUrl(item);
    if (isBase64(item)) return item;
    if (isFile(item)) return fromSdk(readFileSync(item));
    throw new NaceImageError(
      `${JSON.stringify(item.slice(0, 60))} is not base64, a data URL, or an existing file path.`,
    );
  }
  if (item instanceof Uint8Array || item instanceof ArrayBuffer) return fromSdk(item);
  if (isRecord(item)) return fromBlock(item);
  throw new NaceImageError(`Unsupported image type: ${item === null ? "null" : typeof item}.`);
}
