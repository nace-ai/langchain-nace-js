/**
 * Drex Router model selection (`POST /v1/route`).
 *
 * The selector has no dependency on the `langchain` or `@langchain/langgraph`
 * packages. It returns a {@link ModelSelection} and never throws for a failed
 * route call: each failure becomes a fallback. A cancelled run is the
 * exception, and it is raised again.
 */
import type { CallbackManagerForChainRun } from "@langchain/core/callbacks/manager";
import { BaseMessage } from "@langchain/core/messages";
import { Runnable, type RunnableConfig } from "@langchain/core/runnables";
import { APIError, NaceClient, NaceError as SdkNaceError } from "nace-sdk";

import {
  attributedFetch,
  DEFAULT_BASE_URL,
  env,
  hasUserinfo,
  resolveApiKey,
} from "./classifier.js";
import { NaceError } from "./errors.js";
import { isUserMessage, withoutMedia } from "./middleware/media.js";
import { CLIENT_NAME } from "./version.js";

/** Characters of route text kept, newest first. A stand-in for the token limit. */
const CHAR_BUDGET = 24_000;
/** `/v1/route` accepts at most this many messages. */
const MESSAGE_LIMIT = 1_024;
/** Default deadline for one route call. The server stops its own call much later. */
const DEFAULT_TIMEOUT_MS = 3_000;
const MAX_CANDIDATES = 255;
const MAX_ID_LENGTH = 256;
const MAX_PROVIDER_LENGTH = 128;
const MAX_TAGS = 64;
const MAX_TAG_LENGTH = 128;
/** Ids, providers and tags together, the limit the server enforces. */
const MAX_CANDIDATE_TEXT = 100_000;
const PROFILE_KEYS = new Set(["id", "provider", "priceIn", "priceOut", "context", "tags"]);
const PRIORITIES = ["quality", "balanced", "cheapest"] as const;
const REQUEST_ID = /^req_[A-Za-z0-9]{8,64}$/;
const DEADLINE = Symbol("deadline");
const CALLER_ABORT = Symbol("caller-abort");

/** How Drex weighs the candidates. Sent as `selection_priority`. */
export type SelectionPriority = (typeof PRIORITIES)[number];

/** When the selector asks Drex. `"run"` asks once; `"model_call"` asks every time. */
export type SelectOn = "run" | "model_call";

/** How much of the conversation is sent. */
export type SelectionContextMode = "conversation" | "latest_human";

/**
 * Why `model` is the one in use. `selected` is the only reason that uses
 * Drex's choice.
 */
export type SelectionReason =
  | "selected"
  | "low_probability"
  | "single_candidate"
  | "no_input"
  | "no_eligible_candidate"
  | "timeout"
  | "connection_error"
  | "rate_limited"
  | "unavailable"
  | "auth_error"
  | "no_funds"
  | "invalid_request"
  | "invalid_response";

/** Metadata for one candidate. Prices are USD per million tokens. The id is separate. */
export interface CandidateProfile {
  provider: string;
  priceIn: number;
  priceOut: number;
  context: number;
  tags: string[];
}

/** A text turn accepted by `/v1/route`. */
export interface RouteMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

/**
 * The selection kept in agent state under `modelSelection`.
 *
 * `model` is the candidate in use, or `null` when the agent's own model is in
 * use. `selected` is the id Drex returned, or `null`. `fallback` is true when
 * `model` is not that id. Nothing here is message text.
 */
export interface ModelSelection {
  model: string | null;
  selected: string | null;
  probabilities: Record<string, number>;
  fallback: boolean;
  reason: SelectionReason;
  priority: SelectionPriority;
  /** The candidate ids sent, in binding order. Empty when nothing was sent. */
  candidates: string[];
  requestId: string | null;
  latencyMs: number;
  /**
   * The latest human message's id, or `count:<n>` when that message has no id.
   * Set when the selection is stored. No message text.
   */
  anchor: string;
}

/** A bare id, or a profile. A bare id must be one the server already knows. */
export type SelectorCandidate = string | (CandidateProfile & { id: string });

/** What one route exchange ended in, before it becomes a {@link ModelSelection}. */
export type RouteOutcome =
  | {
      type: "response";
      selected: string;
      probabilities: Record<string, number>;
      requestId: string | null;
      latencyMs: number;
    }
  | { type: "invalid_response"; requestId: string | null; latencyMs: number }
  | { type: "status"; status: number; requestId: string | null; latencyMs: number }
  | { type: "timeout"; requestId: string | null; latencyMs: number }
  | { type: "connection_error"; latencyMs: number }
  | { type: "no_input" }
  | { type: "no_eligible_candidate" }
  | { type: "single_candidate"; id: string };

/** The facts the outcome mapping needs that are not part of the exchange itself. */
export interface OutcomeContext {
  priority: SelectionPriority;
  /** Ids sent, in order. Empty when no route call was made. */
  sentIds: readonly string[];
  minProbability?: number;
  /** An eligible fallback id, or `null` to keep the agent's own model. */
  fallbackId: string | null;
  /** Copied onto the selection. Empty when the caller has no message ids. */
  anchor?: string;
}

/** A candidate the filter can accept or reject. `context` is the profile window, in tokens. */
export interface CandidateConstraint {
  id: string;
  images: boolean;
  tools: boolean;
  context?: number;
}

export interface NaceModelSelectorFields {
  /** Defaults to `NACE_API_KEY`. A custom `baseUrl` needs its own key. */
  apiKey?: string;
  /** Defaults to `NACE_BASE_URL`, else `https://console.nace.ai`. OpenRouter is refused. */
  baseUrl?: string;
  /** Deadline for the route call, in milliseconds. Default: 3000. */
  timeout?: number;
  /** Default: `"balanced"`. */
  priority?: SelectionPriority;
  /** Below this probability, use the fallback. Omit to accept any valid probability. */
  minProbability?: number;
  /** Candidates used when a call does not pass its own. */
  candidates?: readonly SelectorCandidate[];
  /** A custom `fetch`, for proxies and tests. */
  fetch?: typeof fetch;
  /** Allow use in a browser, exposing the API key. Default: false. */
  dangerouslyAllowBrowser?: boolean;
}

export interface ModelSelectOptions {
  /** Replaces the constructor's candidates for this call. */
  candidates?: readonly SelectorCandidate[];
  priority?: SelectionPriority;
  minProbability?: number;
  /** Eligible fallback id, or `null` to keep the caller's own model. Default: `null`. */
  fallback?: string | null;
  /** The run's signal. Aborting it cancels the route call and raises. */
  signal?: AbortSignal;
  /** Deadline for this call, in milliseconds. */
  timeout?: number;
  callbacks?: RunnableConfig["callbacks"];
  tags?: string[];
  metadata?: Record<string, unknown>;
}

const WARN_REASONS = new Set<SelectionReason>([
  "low_probability",
  "no_eligible_candidate",
  "timeout",
  "connection_error",
  "rate_limited",
  "unavailable",
  "auth_error",
  "no_funds",
  "invalid_request",
  "invalid_response",
]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (!isRecord(value)) return false;
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

const SELECTION_REASONS: ReadonlySet<string> = new Set<SelectionReason>([
  "selected",
  "low_probability",
  "single_candidate",
  "no_input",
  "no_eligible_candidate",
  "timeout",
  "connection_error",
  "rate_limited",
  "unavailable",
  "auth_error",
  "no_funds",
  "invalid_request",
  "invalid_response",
]);

/**
 * A checkpoint record the middleware may store again.
 *
 * Every field has to match the documented shape. The result is a new object
 * of those fields. A record that fails any check is ignored, and the caller
 * selects again. A throwing getter is ignored too.
 */
export function readStoredSelection(value: unknown): ModelSelection | undefined {
  try {
    if (!isPlainObject(value)) return undefined;
    const anchor = value.anchor;
    if (typeof anchor !== "string") return undefined;
    const reason = value.reason;
    if (typeof reason !== "string" || !SELECTION_REASONS.has(reason)) return undefined;
    const requestId = value.requestId;
    if (requestId != null && (typeof requestId !== "string" || !REQUEST_ID.test(requestId))) {
      return undefined;
    }
    const candidates = value.candidates;
    if (!Array.isArray(candidates)) return undefined;
    const ids = candidates.filter((id): id is string => typeof id === "string");
    if (ids.length !== candidates.length) return undefined;
    const selected = value.selected;
    let cleanSelected: string | null;
    if (selected === null) {
      cleanSelected = null;
    } else if (typeof selected === "string" && ids.includes(selected)) {
      cleanSelected = selected;
    } else {
      return undefined;
    }
    const probabilities = value.probabilities;
    if (!isPlainObject(probabilities)) return undefined;
    const cleanProbabilities: Record<string, number> = {};
    for (const [key, probability] of Object.entries(probabilities)) {
      if (
        typeof probability !== "number" ||
        !Number.isFinite(probability) ||
        probability < 0 ||
        probability > 1
      ) {
        return undefined;
      }
      cleanProbabilities[key] = probability;
    }
    if (typeof value.fallback !== "boolean") return undefined;
    const model = value.model;
    let cleanModel: string | null;
    if (model === null) cleanModel = null;
    else if (typeof model === "string") cleanModel = model;
    else return undefined;
    // `selected` uses the chosen id. A fallback record must not claim that reason.
    if (reason === "selected" && (cleanModel !== cleanSelected || value.fallback !== false)) {
      return undefined;
    }
    const priority = value.priority;
    if (priority !== "quality" && priority !== "balanced" && priority !== "cheapest") {
      return undefined;
    }
    const latencyMs = value.latencyMs;
    if (typeof latencyMs !== "number" || !Number.isFinite(latencyMs)) return undefined;
    return {
      model: cleanModel,
      selected: cleanSelected,
      probabilities: cleanProbabilities,
      fallback: value.fallback,
      reason: reason as SelectionReason,
      priority,
      candidates: ids,
      requestId: typeof requestId === "string" ? requestId : null,
      latencyMs,
      anchor,
    };
  } catch {
    return undefined;
  }
}

/** The error a cancelled run raises again. */
export function cancellationError(signal: AbortSignal): Error {
  return signal.reason instanceof Error
    ? signal.reason
    : new DOMException("This operation was aborted", "AbortError");
}

function assertPriority(priority: SelectionPriority): void {
  if (!PRIORITIES.includes(priority)) {
    throw new NaceError('`priority` must be "quality", "balanced", or "cheapest".');
  }
}

function assertTimeout(timeout: number): void {
  if (!(timeout > 0)) {
    throw new NaceError("`timeout` must be a positive number of milliseconds.");
  }
}

function assertProbability(value: number): void {
  if (!(typeof value === "number" && value >= 0 && value <= 1)) {
    throw new NaceError("`minProbability` must be a number from 0 to 1.");
  }
}

/** Unicode code points, not UTF-16 units. */
/** A provider or a tag is blank when it has no non-whitespace character. */
function hasVisibleCharacter(value: string): boolean {
  return /\S/.test(value);
}

function codePoints(value: string): number {
  let count = 0;
  for (const _char of value) count += 1;
  return count;
}

function assertProfile(profile: CandidateProfile & { id?: string }, id: string): void {
  const tags = profile.tags;
  const keysOk = Object.keys(profile).every((key) => PROFILE_KEYS.has(key));
  const providerOk =
    typeof profile.provider === "string" &&
    hasVisibleCharacter(profile.provider) &&
    codePoints(profile.provider) <= MAX_PROVIDER_LENGTH;
  const pricesOk =
    typeof profile.priceIn === "number" &&
    Number.isFinite(profile.priceIn) &&
    profile.priceIn >= 0 &&
    typeof profile.priceOut === "number" &&
    Number.isFinite(profile.priceOut) &&
    profile.priceOut >= 0;
  const contextOk =
    typeof profile.context === "number" &&
    Number.isSafeInteger(profile.context) &&
    profile.context > 0;
  const tagsOk =
    Array.isArray(tags) &&
    tags.length <= MAX_TAGS &&
    tags.every(
      (tag) =>
        typeof tag === "string" && hasVisibleCharacter(tag) && codePoints(tag) <= MAX_TAG_LENGTH,
    );
  if (!keysOk || !providerOk || !pricesOk || !contextOk || !tagsOk) {
    throw new NaceError(`Candidate "${id}" has an invalid profile.`);
  }
}

function assertId(id: string): void {
  if (id.trim() === "") throw new NaceError("A candidate id must not be blank.");
  if (codePoints(id) > MAX_ID_LENGTH) {
    throw new NaceError(`A candidate id must be at most ${MAX_ID_LENGTH} code points.`);
  }
}

function candidateText(candidate: SelectorCandidate): number {
  if (typeof candidate === "string") return codePoints(candidate);
  return (
    codePoints(candidate.id) +
    codePoints(candidate.provider) +
    candidate.tags.reduce((sum, tag) => sum + codePoints(tag), 0)
  );
}

interface WireCandidate {
  id: string;
  provider: string;
  price_in: number;
  price_out: number;
  context: number;
  tags: string[];
}

/** Checks the candidates and returns the wire form, in the given order. */
export function wireCandidates(
  candidates: readonly SelectorCandidate[],
): (string | WireCandidate)[] {
  if (candidates.length === 0) {
    throw new NaceError("`candidates` must contain at least one model.");
  }
  if (candidates.length > MAX_CANDIDATES) {
    throw new NaceError(`\`candidates\` must contain at most ${MAX_CANDIDATES} models.`);
  }
  const seen = new Map<string, true>();
  const wire: (string | WireCandidate)[] = [];
  for (const candidate of candidates) {
    if (typeof candidate === "string") {
      assertId(candidate);
      if (seen.has(candidate)) throw new NaceError(`Candidate "${candidate}" is listed twice.`);
      seen.set(candidate, true);
      wire.push(candidate);
      continue;
    }
    if (!isRecord(candidate) || typeof candidate.id !== "string") {
      throw new NaceError("A candidate profile needs a string `id`.");
    }
    assertId(candidate.id);
    if (seen.has(candidate.id)) throw new NaceError(`Candidate "${candidate.id}" is listed twice.`);
    seen.set(candidate.id, true);
    assertProfile(candidate, candidate.id);
    wire.push({
      id: candidate.id,
      provider: candidate.provider,
      price_in: candidate.priceIn,
      price_out: candidate.priceOut,
      context: candidate.context,
      tags: [...candidate.tags],
    });
  }
  const text = candidates.reduce((sum, candidate) => sum + candidateText(candidate), 0);
  if (text > MAX_CANDIDATE_TEXT) {
    throw new NaceError(`Candidate text must be at most ${MAX_CANDIDATE_TEXT} code points.`);
  }
  return wire;
}

function candidateId(candidate: string | WireCandidate): string {
  return typeof candidate === "string" ? candidate : candidate.id;
}

function contentOf(message: unknown): unknown {
  if (BaseMessage.isInstance(message)) return message.content;
  return isRecord(message) ? message.content : undefined;
}

function roleOf(message: unknown): "system" | "user" | "assistant" | "tool" | "other" {
  if (BaseMessage.isInstance(message)) {
    switch (message.getType()) {
      case "system":
        return "system";
      case "human":
        return "user";
      case "ai":
        return "assistant";
      case "tool":
        return "tool";
      default:
        return "other";
    }
  }
  if (!isRecord(message)) return "other";
  const role = message.role ?? message.type;
  if (role === "system") return "system";
  if (role === "user" || role === "human") return "user";
  if (role === "assistant" || role === "ai") return "assistant";
  if (role === "tool") return "tool";
  return "other";
}

/** Text of a message, with media replaced by the placeholders `withoutMedia` uses. */
function textOf(message: unknown): string {
  const prepared = withoutMedia(message, { keepImages: false });
  const content = contentOf(prepared);
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const block of content) {
    if (typeof block === "string") {
      parts.push(block);
      continue;
    }
    if (!isRecord(block) || block.type !== "text" || typeof block.text !== "string") continue;
    parts.push(block.text);
  }
  return parts.join("\n");
}

function mimeOf(block: Record<string, unknown>): unknown {
  if (typeof block.mimeType === "string") return block.mimeType;
  if (typeof block.mime_type === "string") return block.mime_type;
  const source = block.source;
  if (!isRecord(source)) return undefined;
  if (typeof source.media_type === "string") return source.media_type;
  if (typeof source.mime_type === "string") return source.mime_type;
  if (typeof source.mimeType === "string") return source.mimeType;
  return undefined;
}

/** An image the capability filter cares about, including provider-native blocks. */
function isRouteImageBlock(block: unknown): boolean {
  if (!isRecord(block)) return false;
  if (block.image != null) return true;
  const kind = block.type;
  if (kind === "image" || kind === "image_url" || kind === "input_image") return true;
  if (kind !== "media" && kind !== "file") return false;
  const mime = mimeOf(block);
  return typeof mime === "string" && mime.toLowerCase().startsWith("image/");
}

/** How deep an image scan walks. A deeper container may still hold an image. */
const IMAGE_SCAN_DEPTH = 8;

function containsImage(value: unknown, depth: number, seen: WeakSet<object>): boolean {
  if (typeof value !== "object" || value === null) return false;
  // Fail closed: a container past the limit is treated as an image.
  if (depth > IMAGE_SCAN_DEPTH) return true;
  if (seen.has(value)) return false;
  seen.add(value);
  if (Array.isArray(value)) {
    return value.some((entry) => containsImage(entry, depth + 1, seen));
  }
  if (!isRecord(value)) return false;
  if (isRouteImageBlock(value)) return true;
  for (const nested of Object.values(value)) {
    if (containsImage(nested, depth + 1, seen)) return true;
  }
  return false;
}

function hasImage(messages: readonly unknown[]): boolean {
  for (const message of messages) {
    if (containsImage(message, 0, new WeakSet())) return true;
  }
  return false;
}

function isHumanMessage(message: unknown): boolean {
  return roleOf(message) === "user" || isUserMessage(message);
}

function idOf(message: unknown): unknown {
  if (BaseMessage.isInstance(message)) return message.id;
  return isRecord(message) ? message.id : undefined;
}

/**
 * FNV-1a over the text. The digest is not a secret and it is not the text.
 */
function textDigest(text: string): string {
  let hash = 0x811c9dc5;
  for (const char of text) {
    hash ^= char.codePointAt(0) ?? 0;
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

/**
 * The anchor stored with a selection: the latest human message's id, or
 * `count:<n>` when that id is missing, plus a digest of that message's text.
 * The value never includes message text. A new id or a new digest starts a
 * new selection.
 */
export function selectionAnchor(messages: readonly unknown[]): string {
  let humans = 0;
  let latest: unknown;
  for (const message of messages) {
    if (!isHumanMessage(message)) continue;
    humans += 1;
    latest = message;
  }
  const digest = textDigest(latest === undefined ? "" : textOf(latest));
  const id = latest === undefined ? undefined : idOf(latest);
  if (typeof id === "string" && id.length > 0) return `${id}:${digest}`;
  return `count:${humans}:${digest}`;
}

function lastHumanMessage(messages: readonly unknown[]): unknown | undefined {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (isHumanMessage(message)) return message;
  }
  return undefined;
}

function toolText(tool: unknown): string {
  if (!isRecord(tool)) return "";
  const name = typeof tool.name === "string" ? tool.name : "";
  const description = typeof tool.description === "string" ? tool.description : "";
  const raw = tool.schema ?? tool.parameters;
  let schema = "";
  if (raw !== undefined) {
    try {
      schema = JSON.stringify(raw);
    } catch {
      schema = "";
    }
  }
  return `${name}${description}${schema}`;
}

/** Characters of the request divided by 3, a stand-in for the token count. */
export function estimateRequestTokens(
  systemText: string,
  messages: readonly unknown[],
  tools: readonly unknown[],
): number {
  let chars = systemText.length;
  for (const message of messages) chars += textOf(message).length;
  for (const tool of tools) chars += toolText(tool).length;
  return Math.ceil(chars / 3);
}

/** Text with media placeholders removed. Empty means the turn has no real text. */
function realText(message: unknown): string {
  const kept: string[] = [];
  for (const line of textOf(message).split("\n")) {
    const trimmed = line.trim();
    if (trimmed === "" || /^\[[^\]\n]+ omitted\]$/.test(trimmed)) continue;
    kept.push(line);
  }
  return kept.join("\n").trim();
}

function lastUserIndex(messages: readonly RouteMessage[]): number {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]!;
    if (message.role === "user" && message.content.trim() !== "") return index;
  }
  return -1;
}

/**
 * Keeps the latest human message, then fills the rest of the budget with the
 * newest other turns first. The result stays in chronological order. The latest
 * human message is cut to the budget when it is longer.
 */
function fitBudget(messages: readonly RouteMessage[]): RouteMessage[] {
  const latest = lastUserIndex(messages);
  if (latest === -1) return [];
  const fitted = messages.map((message, index) =>
    index === latest && message.content.length > CHAR_BUDGET
      ? { role: message.role, content: message.content.slice(-CHAR_BUDGET) }
      : message,
  );
  const include = new Set<number>([latest]);
  let used = fitted[latest]!.content.length;
  for (let index = fitted.length - 1; index >= 0 && include.size < MESSAGE_LIMIT; index -= 1) {
    if (include.has(index)) continue;
    const message = fitted[index]!;
    if (used + message.content.length > CHAR_BUDGET) continue;
    include.add(index);
    used += message.content.length;
  }
  return fitted.filter((_, index) => include.has(index));
}

function promptText(systemMessage: unknown, systemPrompt: string | undefined): string {
  if (systemMessage !== undefined && systemMessage !== "") {
    const text = textOf(systemMessage);
    if (text.trim() !== "") return text;
  }
  return typeof systemPrompt === "string" && systemPrompt.trim() !== "" ? systemPrompt : "";
}

/**
 * The route messages for one model request.
 *
 * Returns `null` when no human text remains, so the caller makes no route call.
 * Tool calls and tool results are dropped. Images become placeholders.
 */
export function projectRouteMessages(
  request: { systemMessage?: unknown; systemPrompt?: string; messages: readonly unknown[] },
  context: SelectionContextMode,
): RouteMessage[] | null {
  const projected: RouteMessage[] = [];
  const prompt = promptText(request.systemMessage, request.systemPrompt);
  const latest = lastHumanMessage(request.messages);
  // Placeholders such as "[image omitted]" are not text. An empty latest human
  // turn is no_input in both context modes, so an earlier turn is not sent.
  if (latest === undefined || realText(latest) === "") return null;
  if (prompt !== "") projected.push({ role: "system", content: prompt });
  for (const message of request.messages) {
    const role = roleOf(message);
    if (role === "tool" || role === "other") continue;
    const text = textOf(message);
    if (text.trim() === "") continue;
    if (role === "user" || isUserMessage(message)) {
      projected.push({ role: "user", content: text });
      continue;
    }
    if (role === "assistant") projected.push({ role: "assistant", content: text });
    if (role === "system") projected.push({ role: "system", content: text });
  }
  if (lastUserIndex(projected) === -1) return null;
  const scoped =
    context === "latest_human"
      ? projected.filter(
          (message, index) => message.role === "system" || index === lastUserIndex(projected),
        )
      : projected;
  const fitted = fitBudget(scoped);
  return lastUserIndex(fitted) === -1 ? null : fitted;
}

/**
 * Eligible candidate ids, in binding order.
 *
 * A candidate that refuses images or tools is dropped when the request has
 * them. A profile whose context window is smaller than the estimated request
 * is dropped too.
 */
export function filterEligibleCandidates(
  candidates: readonly CandidateConstraint[],
  request: {
    messages: readonly unknown[];
    tools: readonly unknown[];
    systemText?: string;
    /** Set when the agent asked for structured output, which can bind a tool. */
    responseFormat?: unknown;
  },
): string[] {
  const image = hasImage(request.messages);
  const tools = request.tools.length > 0 || request.responseFormat != null;
  const tokens = estimateRequestTokens(request.systemText ?? "", request.messages, request.tools);
  const eligible: string[] = [];
  for (const candidate of candidates) {
    if (!candidate.images && image) continue;
    if (!candidate.tools && tools) continue;
    if (candidate.context !== undefined && candidate.context < tokens) continue;
    eligible.push(candidate.id);
  }
  return eligible;
}

function requestIdOf(value: unknown, secrets: readonly string[]): string | null {
  if (typeof value !== "string" || !REQUEST_ID.test(value)) return null;
  for (const secret of secrets) {
    if (secret !== "" && value.includes(secret)) return null;
  }
  return value;
}

/**
 * Whether a route body is a selection for exactly the ids that were sent.
 * Bindings are a `Map`, so an id such as `__proto__` matches only when it was sent.
 */
export function validateRouteResponse(
  body: unknown,
  sentIds: readonly string[],
  secrets: readonly string[] = [],
):
  | {
      ok: true;
      selected: string;
      probabilities: Record<string, number>;
      requestId: string | null;
    }
  | { ok: false; requestId: string | null } {
  const requestId = isRecord(body) ? requestIdOf(body.request_id, secrets) : null;
  if (!isRecord(body) || typeof body.selected !== "string" || !isRecord(body.probabilities)) {
    return { ok: false, requestId };
  }
  const allowed = new Map(sentIds.map((id) => [id, true]));
  if (!allowed.has(body.selected)) return { ok: false, requestId };
  const keys = Object.keys(body.probabilities);
  if (keys.length !== sentIds.length) return { ok: false, requestId };
  const probabilities: Record<string, number> = {};
  for (const id of sentIds) {
    if (!Object.hasOwn(body.probabilities, id)) return { ok: false, requestId };
    const value = Object.getOwnPropertyDescriptor(body.probabilities, id)?.value;
    if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) {
      return { ok: false, requestId };
    }
    Object.defineProperty(probabilities, id, {
      value,
      enumerable: true,
      writable: true,
      configurable: true,
    });
  }
  for (const key of keys) {
    if (!allowed.has(key)) return { ok: false, requestId };
  }
  return { ok: true, selected: body.selected, probabilities, requestId };
}

function reasonForStatus(status: number): SelectionReason {
  if (status === 429) return "rate_limited";
  if (status === 401 || status === 403) return "auth_error";
  if (status === 402) return "no_funds";
  if (status === 408) return "timeout";
  if (status >= 500) return "unavailable";
  return "invalid_request";
}

function ownNumber(record: Record<string, number>, key: string): number {
  const value = Object.getOwnPropertyDescriptor(record, key)?.value;
  return typeof value === "number" ? value : Number.NaN;
}

/**
 * Maps one exchange outcome to a {@link ModelSelection}.
 * This is the only place an outcome becomes a reason.
 */
export function selectionFromOutcome(
  outcome: RouteOutcome,
  context: OutcomeContext,
): ModelSelection {
  let reason: SelectionReason;
  let model: string | null;
  let selected: string | null = null;
  let probabilities: Record<string, number> = {};
  let requestId: string | null = null;
  let latencyMs = 0;
  let candidates = [...context.sentIds];
  switch (outcome.type) {
    case "no_input":
      reason = "no_input";
      model = null;
      candidates = [];
      break;
    case "no_eligible_candidate":
      reason = "no_eligible_candidate";
      model = context.fallbackId;
      candidates = [];
      break;
    case "single_candidate":
      reason = "single_candidate";
      model = outcome.id;
      candidates = [outcome.id];
      break;
    case "response": {
      const probability = ownNumber(outcome.probabilities, outcome.selected);
      const low = context.minProbability !== undefined && probability < context.minProbability;
      reason = low ? "low_probability" : "selected";
      model = low ? context.fallbackId : outcome.selected;
      selected = outcome.selected;
      probabilities = outcome.probabilities;
      requestId = outcome.requestId;
      latencyMs = outcome.latencyMs;
      break;
    }
    case "invalid_response":
      reason = "invalid_response";
      model = context.fallbackId;
      requestId = outcome.requestId;
      latencyMs = outcome.latencyMs;
      break;
    case "timeout":
      reason = "timeout";
      model = context.fallbackId;
      requestId = outcome.requestId;
      latencyMs = outcome.latencyMs;
      break;
    case "connection_error":
      reason = "connection_error";
      model = context.fallbackId;
      latencyMs = outcome.latencyMs;
      break;
    case "status":
      reason = reasonForStatus(outcome.status);
      model = context.fallbackId;
      requestId = outcome.requestId;
      latencyMs = outcome.latencyMs;
      break;
  }
  return {
    model,
    selected,
    probabilities,
    fallback: model !== selected,
    reason,
    priority: context.priority,
    candidates,
    requestId,
    latencyMs,
    anchor: context.anchor ?? "",
  };
}

function logFallback(
  selection: ModelSelection,
  outcome: RouteOutcome,
  knownIds: readonly string[],
): void {
  if (!WARN_REASONS.has(selection.reason)) return;
  const status = outcome.type === "status" ? outcome.status : undefined;
  const ids = selection.candidates.length > 0 ? selection.candidates : knownIds;
  const statusPart = status === undefined ? "" : ` status=${status}`;
  const requestPart = selection.requestId ? ` request_id=${selection.requestId}` : "";
  // The reason, the status, the request id and the candidate ids. No message and no body.
  console.warn(
    `[@nace-ai/langchain-nace] Drex model selection fell back: ${selection.reason}${statusPart}${requestPart}. Candidates: ${ids.join(", ")}.`,
  );
}

/** Maps the outcome and writes the one fallback warning. */
export function settleSelection(
  outcome: RouteOutcome,
  context: OutcomeContext,
  knownIds: readonly string[] = context.sentIds,
): ModelSelection {
  const selection = selectionFromOutcome(outcome, context);
  logFallback(selection, outcome, knownIds);
  return selection;
}

function outcomeFromError(
  error: unknown,
  latencyMs: number,
  secrets: readonly string[],
): RouteOutcome {
  // Only this exchange's own deadline is a timeout. A timeout raised by the
  // caller or the SDK before that deadline is a connection error.
  if (error instanceof APIError) {
    return {
      type: "status",
      status: error.status,
      requestId: requestIdOf(error.requestId, secrets),
      latencyMs,
    };
  }
  return { type: "connection_error", latencyMs };
}

function outcomeFromResponse(
  value: { data: unknown; requestId?: string | null },
  sentIds: readonly string[],
  latencyMs: number,
  secrets: readonly string[],
): RouteOutcome {
  const validated = validateRouteResponse(value.data, sentIds, secrets);
  const requestId = requestIdOf(value.requestId, secrets) ?? validated.requestId;
  if (!validated.ok) return { type: "invalid_response", requestId, latencyMs };
  return {
    type: "response",
    selected: validated.selected,
    probabilities: validated.probabilities,
    requestId,
    latencyMs,
  };
}

/**
 * Owns one route exchange: send, wait, deadline, abort, and the cleanup of the
 * timer and the abort listener.
 *
 * Node 20.0 has no `AbortSignal.any`, so the run signal and the deadline are
 * combined here. A response that loses the race is discarded. When the run
 * signal is aborted the cancellation is raised and the handler must not run.
 */
export async function exchangeRoute(
  client: NaceClient,
  params: {
    messages: RouteMessage[];
    candidates: (string | WireCandidate)[];
    selection_priority: SelectionPriority;
  },
  options: { signal?: AbortSignal; timeoutMs: number; secrets?: readonly string[] },
): Promise<RouteOutcome> {
  try {
    return await runExchange(client, params, options);
  } catch {
    const signal = options.signal;
    // Cleanup or the SDK can throw a different error after the caller aborts.
    // The abort still wins, and the handler must not run.
    if (signal?.aborted) throw cancellationError(signal);
    return { type: "connection_error", latencyMs: 0 };
  }
}

async function runExchange(
  client: NaceClient,
  params: {
    messages: RouteMessage[];
    candidates: (string | WireCandidate)[];
    selection_priority: SelectionPriority;
  },
  options: { signal?: AbortSignal; timeoutMs: number; secrets?: readonly string[] },
): Promise<RouteOutcome> {
  const secrets = options.secrets ?? [];
  const runSignal = options.signal;
  if (runSignal?.aborted) throw cancellationError(runSignal);
  const controller = new AbortController();
  let timedOut = false;
  let noteAbort: (() => void) | undefined;
  const callerAbort = new Promise<typeof CALLER_ABORT>((resolve) => {
    noteAbort = () => resolve(CALLER_ABORT);
  });
  const onAbort = () => {
    controller.abort();
    noteAbort?.();
  };
  if (runSignal) runSignal.addEventListener("abort", onAbort);
  const started = Date.now();
  let timer: ReturnType<typeof setTimeout> | undefined;
  let pending: Promise<unknown> = Promise.resolve();
  try {
    const route = client
      .route(params, {
        signal: controller.signal,
        retry: { maxRetries: 0 },
        timeout: options.timeoutMs,
      })
      .withResponse();
    pending = route;
    const deadline = new Promise<typeof DEADLINE>((resolve) => {
      timer = setTimeout(() => {
        timedOut = true;
        controller.abort();
        resolve(DEADLINE);
      }, options.timeoutMs);
    });
    const winner = await Promise.race([
      route.then(
        (value) => ({ kind: "value" as const, value }),
        (error: unknown) => ({ kind: "error" as const, error }),
      ),
      deadline,
      callerAbort,
    ]);
    const latencyMs = Date.now() - started;
    if (winner === CALLER_ABORT) {
      if (!runSignal) return { type: "connection_error", latencyMs };
      throw cancellationError(runSignal);
    }
    if (runSignal?.aborted) throw cancellationError(runSignal);
    // An error or a body that arrives once the deadline has passed is a timeout,
    // even when the timer callback has not run yet.
    if (winner === DEADLINE || timedOut || latencyMs >= options.timeoutMs) {
      return { type: "timeout", requestId: null, latencyMs };
    }
    if (winner.kind === "error") return outcomeFromError(winner.error, latencyMs, secrets);
    return outcomeFromResponse(
      winner.value,
      params.candidates.map(candidateId),
      latencyMs,
      secrets,
    );
  } finally {
    if (timer !== undefined) clearTimeout(timer);
    try {
      runSignal?.removeEventListener("abort", onAbort);
    } catch {
      // Cleanup must not replace an abort or a settled outcome.
    }
    pending.catch(() => {});
  }
}

function stampRequestId(
  runManager: CallbackManagerForChainRun | undefined,
  requestId: string | null,
): void {
  if (!runManager || !requestId) return;
  const metadata = (runManager as unknown as { metadata?: Record<string, unknown> }).metadata;
  if (!metadata) return;
  try {
    metadata.nace_request_id = requestId;
  } catch {
    // A frozen metadata object cannot take the id. The selection still has it.
  }
}

interface TraceInput {
  candidates: string[];
  priority: SelectionPriority;
  messageCount: number;
  characterCount: number;
}

/** Traces one selection. The run input is the summary, never the message text. */
class SelectionTrace extends Runnable<TraceInput, ModelSelection> {
  static lc_name(): string {
    return "NaceModelSelector";
  }

  lc_namespace: string[] = ["langchain", "selectors", "nace"];

  constructor(
    private readonly body: (runManager?: CallbackManagerForChainRun) => Promise<ModelSelection>,
  ) {
    super();
  }

  invoke(input: TraceInput, options?: RunnableConfig): Promise<ModelSelection> {
    return this._callWithConfig((_input, _config, runManager) => this.body(runManager), input, {
      ...options,
      runName: "NaceModelSelector",
    });
  }
}

/** `openrouter.ai` and any subdomain, ignoring case and a trailing dot. */
function isOpenRouterHost(baseUrl: string): boolean {
  let hostname: string;
  try {
    hostname = new URL(baseUrl).hostname;
  } catch {
    return false;
  }
  hostname = hostname.replace(/\.$/, "").toLowerCase();
  return hostname === "openrouter.ai" || hostname.endsWith(".openrouter.ai");
}

/** Drops request and error bodies. The SDK would otherwise print them at debug. */
function redactLogValue(value: unknown): unknown {
  if (
    value == null ||
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return value;
  }
  return "[redacted]";
}

const routeLogger = {
  debug(message: string, ...args: unknown[]) {
    console.debug(message, ...args.map(redactLogValue));
  },
  info(message: string, ...args: unknown[]) {
    console.info(message, ...args.map(redactLogValue));
  },
  warn(message: string, ...args: unknown[]) {
    console.warn(message, ...args.map(redactLogValue));
  },
  error(message: string, ...args: unknown[]) {
    console.error(message, ...args.map(redactLogValue));
  },
};

/**
 * Asks Drex Router which candidate should answer.
 *
 * `select` sends text turns and candidate ids to `POST /v1/route` and returns
 * the selection. It does not call a chat model. One attempt is made. A failed
 * call uses `fallback` (or no model, when `fallback` is null).
 */
export class NaceModelSelector {
  readonly timeout: number;

  readonly priority: SelectionPriority;

  readonly minProbability: number | undefined;

  readonly baseUrl: string;

  readonly #client: NaceClient;

  readonly #candidates: readonly SelectorCandidate[];

  readonly #secrets: readonly string[];

  constructor(fields: NaceModelSelectorFields = {}) {
    const explicitBaseUrl = fields.baseUrl?.trim() || undefined;
    let baseUrl = explicitBaseUrl ?? env("NACE_BASE_URL") ?? DEFAULT_BASE_URL;
    while (baseUrl.endsWith("/")) baseUrl = baseUrl.slice(0, -1);
    if (hasUserinfo(baseUrl)) {
      throw new NaceError(
        "`baseUrl` must not contain a user name or password; pass the key as `apiKey`.",
      );
    }
    if (isOpenRouterHost(baseUrl)) {
      throw new NaceError(
        "Model selection is not available through OpenRouter; use a Nace endpoint.",
      );
    }
    const apiKey = resolveApiKey(fields.apiKey, baseUrl, explicitBaseUrl !== undefined);
    const timeout = fields.timeout ?? DEFAULT_TIMEOUT_MS;
    assertTimeout(timeout);
    const priority = fields.priority ?? "balanced";
    assertPriority(priority);
    if (fields.minProbability !== undefined) assertProbability(fields.minProbability);
    const candidates = fields.candidates ?? [];
    if (candidates.length > 0) wireCandidates(candidates);
    let client: NaceClient;
    try {
      client = new NaceClient({
        apiKey,
        baseURL: baseUrl,
        timeout,
        retry: { maxRetries: 0, timeoutMs: null },
        clientInfo: CLIENT_NAME,
        dangerouslyAllowBrowser: fields.dangerouslyAllowBrowser ?? false,
        fetch: attributedFetch(fields.fetch ?? ((input, init) => globalThis.fetch(input, init))),
        logger: routeLogger,
      });
    } catch (error) {
      if (error instanceof SdkNaceError) throw new NaceError(error.message, { cause: error });
      throw error;
    }
    this.timeout = timeout;
    this.priority = priority;
    this.minProbability = fields.minProbability;
    this.baseUrl = baseUrl;
    this.#client = client;
    this.#candidates = candidates;
    this.#secrets = secretValues(apiKey, client);
  }

  /**
   * Select a candidate for these text turns.
   *
   * The traced inputs are the candidate ids, the priority, and the message and
   * character counts. The traced output is the {@link ModelSelection}.
   */
  invoke(messages: readonly RouteMessage[], options?: ModelSelectOptions): Promise<ModelSelection> {
    return this.select(messages, options);
  }

  async select(
    messages: readonly RouteMessage[],
    options?: ModelSelectOptions,
  ): Promise<ModelSelection> {
    const priority = options?.priority ?? this.priority;
    const minProbability = options?.minProbability ?? this.minProbability;
    const timeout = options?.timeout ?? this.timeout;
    assertPriority(priority);
    assertTimeout(timeout);
    if (minProbability !== undefined) assertProbability(minProbability);
    const wire = wireCandidates(options?.candidates ?? this.#candidates);
    const sentIds = wire.map(candidateId);
    const fallbackId = options?.fallback ?? null;
    if (fallbackId !== null && !sentIds.includes(fallbackId)) {
      throw new NaceError("The fallback is not one of the candidates.");
    }
    const turns = normalizeTurns(messages);
    const traceInput = {
      candidates: sentIds,
      priority,
      messageCount: turns.length,
      characterCount: turns.reduce((sum, message) => sum + message.content.length, 0),
    };
    return new SelectionTrace(async (runManager) => {
      const context: OutcomeContext = { priority, sentIds, minProbability, fallbackId };
      if (lastUserIndex(turns) === -1) {
        return settleSelection({ type: "no_input" }, { ...context, sentIds: [] }, sentIds);
      }
      if (wire.length === 1) {
        const only = sentIds[0]!;
        return settleSelection(
          { type: "single_candidate", id: only },
          { ...context, sentIds: [only] },
          sentIds,
        );
      }
      const outcome = await exchangeRoute(
        this.#client,
        { messages: turns, candidates: wire, selection_priority: priority },
        { signal: options?.signal, timeoutMs: timeout, secrets: this.#secrets },
      );
      const selection = settleSelection(outcome, context, sentIds);
      stampRequestId(runManager, selection.requestId);
      return selection;
    }).invoke(traceInput, {
      callbacks: options?.callbacks,
      tags: options?.tags,
      metadata: options?.metadata,
      signal: options?.signal,
    });
  }
}

function secretValues(apiKey: string, client: NaceClient): string[] {
  const values: string[] = [];
  if (apiKey !== "") values.push(apiKey);
  const headers = client.defaultHeaders;
  for (const value of Object.values(headers)) {
    if (typeof value === "string" && value !== "") values.push(value);
  }
  return values;
}

function normalizeTurns(messages: readonly RouteMessage[]): RouteMessage[] {
  if (messages.length > MESSAGE_LIMIT) {
    throw new NaceError(`At most ${MESSAGE_LIMIT} route messages can be sent.`);
  }
  return messages.map((message) => {
    if (message.role !== "system" && message.role !== "user" && message.role !== "assistant") {
      throw new NaceError("A route message role must be system, user, or assistant.");
    }
    if (typeof message.content !== "string") {
      throw new NaceError("A route message's content must be a string.");
    }
    return { role: message.role, content: message.content };
  });
}
