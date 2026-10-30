/**
 * Errors thrown by langchain-nace.
 *
 * Every class extends LangChain's `LangChainError` and is brand-checked, so
 * `NaceRateLimitError.isInstance(error)` works even when two copies of the
 * package are installed. Messages never include state, questions or response
 * bodies, so they are safe to log.
 */
import { ns as baseNs, LangChainError, stampRetryable } from "@langchain/core/errors";
import * as sdk from "nace-sdk";

const ns = baseNs.sub("nace");

export const BILLING_URL = "https://console.nace.ai/dashboard/billing";

/** A `{ path, message }` validation issue from Drex. */
export interface Issue {
  path: string;
  message: string;
}

function defineHidden(target: object, key: string, value: unknown): void {
  Object.defineProperty(target, key, {
    value,
    enumerable: false,
    writable: false,
    configurable: true,
  });
}

/** Base class for every error this package throws. */
export class NaceError extends ns.brand(LangChainError) {
  override readonly name: string = "NaceError";

  constructor(message?: string, options?: { cause?: unknown }) {
    super(message);
    if (options?.cause !== undefined) {
      // Non-enumerable: an SDK error's body would otherwise reach JSON logs.
      Object.defineProperty(this, "cause", {
        value: options.cause,
        enumerable: false,
        writable: true,
        configurable: true,
      });
    }
  }
}

/** An image can't be sent as given (a remote URL, an unknown block, ...). */
export class NaceImageError extends ns.brand(NaceError, "image") {
  override readonly name: string = "NaceImageError";
}

/** Images were sent to a model that reads text only. */
export class NaceImageModelError extends ns.brand(NaceImageError, "image-model") {
  override readonly name: string = "NaceImageModelError";
}

/** State holds media Drex can't read (audio, video, files). */
export class NaceMediaError extends ns.brand(NaceError, "media") {
  override readonly name: string = "NaceMediaError";
}

/**
 * A `batch` input that was never sent, because an earlier request in the batch
 * failed in a way every later request would too (401, 402, 403) or the batch was
 * cancelled. `cause` is that failure.
 */
export class NaceBatchStoppedError extends ns.brand(NaceError, "batch-stopped") {
  override readonly name: string = "NaceBatchStoppedError";
}

/** Drex answered with an unsuccessful HTTP status. */
export class NaceAPIError extends ns.brand(NaceError, "api") {
  override readonly name: string = "NaceAPIError";

  /** The HTTP status. */
  readonly status: number;

  /** Drex's request id; quote it in support requests. */
  readonly requestId: string | undefined;

  /** The class of failure Drex reports, such as `invalid_request_error`. */
  readonly type: string | undefined;

  /** Where the request failed validation. */
  issues: Issue[];

  /** The response body. Non-enumerable, so it never reaches logs by accident. */
  declare readonly body: unknown;

  /** The response headers. Non-enumerable. */
  declare readonly headers: Headers | undefined;

  constructor(
    message: string,
    fields: {
      status: number;
      requestId?: string | undefined;
      type?: string | undefined;
      issues?: Issue[];
      body?: unknown;
      headers?: Headers;
      cause?: unknown;
    },
  ) {
    super(message, { cause: fields.cause });
    this.status = fields.status;
    this.requestId = fields.requestId;
    this.type = fields.type;
    this.issues = fields.issues ?? [];
    defineHidden(this, "body", fields.body);
    defineHidden(this, "headers", fields.headers);
  }

  /** Alias for `status`, matching `@langchain/typesafe`. */
  get statusCode(): number {
    return this.status;
  }
}

/** HTTP 400. */
export class NaceBadRequestError extends ns.brand(NaceAPIError, "bad-request") {
  override readonly name: string = "NaceBadRequestError";
}

/** HTTP 401: the API key is missing, wrong or revoked. */
export class NaceAuthenticationError extends ns.brand(NaceAPIError, "authentication") {
  override readonly name: string = "NaceAuthenticationError";
}

/** HTTP 402: the account has no available funds. */
export class NaceInsufficientCreditError extends ns.brand(NaceAPIError, "insufficient-credit") {
  override readonly name: string = "NaceInsufficientCreditError";
}

/** HTTP 403. */
export class NacePermissionDeniedError extends ns.brand(NaceAPIError, "permission-denied") {
  override readonly name: string = "NacePermissionDeniedError";
}

/** HTTP 404. */
export class NaceNotFoundError extends ns.brand(NaceAPIError, "not-found") {
  override readonly name: string = "NaceNotFoundError";
}

/** HTTP 422: Drex rejected the request; `issues` says where. */
export class NaceUnprocessableEntityError extends ns.brand(NaceAPIError, "unprocessable-entity") {
  override readonly name: string = "NaceUnprocessableEntityError";
}

/** HTTP 429, after the retries ran out. */
export class NaceRateLimitError extends ns.brand(NaceAPIError, "rate-limit") {
  override readonly name: string = "NaceRateLimitError";

  /** The server's retry delay in milliseconds, when it sent one. */
  retryAfterMs: number | undefined;
}

/** HTTP 5xx, after the retries ran out. */
export class NaceInternalServerError extends ns.brand(NaceAPIError, "internal-server") {
  override readonly name: string = "NaceInternalServerError";
}

/** HTTP 529: Drex is overloaded; retried with `retry-after-ms` before throwing. */
export class NaceOverloadedError extends ns.brand(NaceInternalServerError, "overloaded") {
  override readonly name: string = "NaceOverloadedError";
}

/** Drex could not be reached. */
export class NaceAPIConnectionError extends ns.brand(NaceError, "api-connection") {
  override readonly name: string = "NaceAPIConnectionError";
}

/** The request timed out. */
export class NaceAPITimeoutError extends ns.brand(NaceAPIConnectionError, "api-timeout") {
  override readonly name: string = "NaceAPITimeoutError";

  /** The per-attempt timeout that ran out. */
  timeoutMs: number | undefined;
}

/** Drex answered 200 with a body this package can't read. */
export class NaceAPIResponseValidationError extends ns.brand(NaceError, "api-response-validation") {
  override readonly name: string = "NaceAPIResponseValidationError";
}

type APIErrorClass = new (
  message: string,
  fields: ConstructorParameters<typeof NaceAPIError>[1],
) => NaceAPIError;

const BY_STATUS: Record<number, APIErrorClass> = {
  400: NaceBadRequestError,
  401: NaceAuthenticationError,
  402: NaceInsufficientCreditError,
  403: NacePermissionDeniedError,
  404: NaceNotFoundError,
  422: NaceUnprocessableEntityError,
  429: NaceRateLimitError,
  529: NaceOverloadedError,
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function issuesFrom(list: unknown): Issue[] {
  if (!Array.isArray(list)) return [];
  const issues: Issue[] = [];
  for (const item of list) {
    if (!isRecord(item)) continue;
    const message = item.message ?? item.msg;
    if (typeof message !== "string" || message === "") continue;
    let path = item.path ?? item.loc ?? "";
    if (Array.isArray(path)) {
      path = path.filter((part) => part !== "body").join(".");
    }
    issues.push({ path: String(path), message });
  }
  return issues;
}

/**
 * Make a gateway's raw error text readable.
 *
 * OpenRouter passes upstream errors through as text such as
 * `HTTP 422: {"detail": "..."}` or a JSON list of schema issues. Returns a
 * short message and any issues found in it.
 */
export function unwrapGatewayMessage(raw: string): { message: string; issues: Issue[] } {
  let text = raw.trim();
  const wrapped = /^HTTP \d{3}: ([\s\S]*)$/.exec(text);
  if (wrapped?.[1] !== undefined) {
    text = wrapped[1].trim();
  }
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    return { message: text, issues: [] };
  }
  if (isRecord(data)) {
    const inner = data.detail ?? data.message ?? data.error;
    if (typeof inner === "string") return { message: inner, issues: [] };
    if (isRecord(inner) && typeof inner.message === "string") {
      return { message: inner.message, issues: issuesFrom(inner.issues) };
    }
    data = inner;
  }
  const issues = issuesFrom(data);
  if (issues.length > 0) return { message: "the request is invalid", issues };
  return { message: text, issues: [] };
}

/** The error message from a body, without echoing the rest of the body. */
function bodyMessage(body: unknown): string | undefined {
  if (typeof body === "string") return body.trim() || undefined;
  if (!isRecord(body)) return undefined;
  const { error, message, detail } = body;
  if (typeof error === "string") return error;
  if (isRecord(error) && typeof error.message === "string") return error.message;
  if (typeof message === "string") return message;
  if (typeof detail === "string") return detail;
  if (isRecord(detail) && typeof detail.message === "string") return detail.message;
  return undefined;
}

function bodyIssues(body: unknown): Issue[] {
  if (!isRecord(body)) return [];
  if (isRecord(body.error)) {
    const issues = issuesFrom(body.error.issues);
    if (issues.length > 0) return issues;
  }
  return issuesFrom(body.detail);
}

function describe(
  provider: string,
  status: number,
  message: string | undefined,
  issues: Issue[],
  requestId: string | undefined,
): string {
  let text = `${provider} returned ${status}`;
  if (message) text += `: ${message}`;
  const lines = issues.map((issue) => `${issue.path || "(request)"}: ${issue.message}`);
  // Drex's top-level message often repeats its only issue; don't print it twice.
  const repeated = issues.length === 1 && issues[0]?.message === message;
  if (lines.length > 0 && !repeated && !(lines.length === 1 && lines[0] === message)) {
    text += `\n${lines.map((line) => `  ${line}`).join("\n")}`;
  }
  if (status === 402) text += `\nAdd funds at ${BILLING_URL}`;
  if (requestId) text += `\n(request_id=${requestId})`;
  return text;
}

/**
 * Map a `nace-sdk` error to this package's classes.
 *
 * `provider` names who answered (`"Drex"`, or `"OpenRouter"` on that backend),
 * so the message points the user to the right place.
 */
export function convertError(error: unknown, provider = "Drex"): unknown {
  if (NaceError.isInstance(error)) return error;
  if (error instanceof sdk.APITimeoutError) {
    const converted = new NaceAPITimeoutError(
      `${provider} did not answer within ${error.timeoutMs} ms.`,
      { cause: error },
    );
    converted.timeoutMs = error.timeoutMs;
    return stampRetryable(converted, true);
  }
  if (error instanceof sdk.APIConnectionError) {
    return stampRetryable(
      new NaceAPIConnectionError(`Could not reach ${provider}: ${error.message}`, { cause: error }),
      true,
    );
  }
  if (error instanceof sdk.APIError) {
    const status = error.status;
    const Class = BY_STATUS[status] ?? (status >= 500 ? NaceInternalServerError : NaceAPIError);
    let message = bodyMessage(error.body);
    let issues = issuesFrom(error.issues);
    if (issues.length === 0) issues = bodyIssues(error.body);
    if (message !== undefined && (issues.length === 0 || /^HTTP \d{3}: /.test(message))) {
      const unwrapped = unwrapGatewayMessage(message);
      message = unwrapped.message;
      if (issues.length === 0) issues = unwrapped.issues;
    }
    const converted = new Class(describe(provider, status, message, issues, error.requestId), {
      status,
      requestId: error.requestId,
      type: error.type,
      issues,
      body: error.body,
      headers: error.headers,
      cause: error,
    });
    if (converted instanceof NaceRateLimitError && error instanceof sdk.RateLimitError) {
      converted.retryAfterMs = error.retryAfterMs;
    }
    const retryable = status === 408 || status === 429 || status >= 500;
    return stampRetryable(converted, retryable);
  }
  if (error instanceof sdk.NaceError) {
    return new NaceError(error.message, { cause: error });
  }
  return error;
}
