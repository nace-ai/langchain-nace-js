/** NDI (Nace Document Intelligence): parse, extract and workspace search. */
import { readFile } from "node:fs/promises";
import { basename } from "node:path";

import {
  type DocumentJob,
  JobTimeoutError,
  NaceClient,
  NaceError as SdkNaceError,
  type Source,
  TERMINAL_STATUSES,
  type UploadedFile,
  workspaceFile,
} from "nace-sdk";

import {
  attributedFetch,
  DEFAULT_BASE_URL,
  env,
  hasUserinfo,
  isOpenRouter,
  resolveApiKey,
} from "../classifier.js";
import { clip, convertError, NaceAPIError, NaceAPITimeoutError, NaceError } from "../errors.js";
import { CLIENT_NAME } from "../version.js";

/** The provider name in NDI error messages. */
export const NDI = "NDI";
/** How long an NDI job may take, in milliseconds. Default: 300000. */
export const DEFAULT_JOB_TIMEOUT_MS = 300_000;
/** One HTTP attempt; a create holds the request for up to 60 s. */
const HTTP_TIMEOUT_MS = 90_000;
/** The prefix of the path a one-off upload goes to; a UUID keeps paths unique. */
const UPLOAD_PATH_PREFIX = "langchain-nace";

/** Connection fields that the three NDI components share. */
export interface NaceNdiFields {
  /** Defaults to `NACE_API_KEY`. The rules of "Credentials and custom providers" in the README apply. */
  apiKey?: string;
  /** Defaults to `NACE_BASE_URL`, else `https://console.nace.ai`. */
  baseUrl?: string;
  /** How long to wait for the NDI job, in milliseconds. Default: 300000. */
  timeout?: number;
  /** Retries of one HTTP request (408, 429, 5xx). Default: 2. */
  maxRetries?: number;
  /** A custom `fetch`, for proxies and tests. */
  fetch?: typeof fetch;
}

/** What the NDI components take as a document: an https URL, a local path, or bytes. */
export type NdiSource = string | Blob | Uint8Array;

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** A `File`'s own name, when the bytes came as one. */
export function fileNameOf(content: Blob | Uint8Array): string | undefined {
  return typeof File !== "undefined" && content instanceof File && content.name
    ? content.name
    : undefined;
}

/** The last path segment of a URL, percent-decoded, as the file's name. */
function urlFileName(url: string): string {
  let segment: string = "";
  try {
    segment = new URL(url).pathname.split("/").pop() ?? "";
    segment = decodeURIComponent(segment);
  } catch {
    // Keep the raw segment; a broken encoding still names a file.
  }
  return segment;
}

/**
 * A URL without the parts that can hold a credential: the query, the fragment,
 * the user name and the password. Parsed with the URL class, so a path with
 * spaces or other unusual characters is still redacted (and comes back
 * percent-encoded). The scheme keeps the case it was written with.
 */
export function redactUrl(url: string): string {
  try {
    const parsed = new URL(url);
    parsed.search = "";
    parsed.hash = "";
    parsed.username = "";
    parsed.password = "";
    const shown = parsed.toString();
    // The URL class lowercases the scheme; keep it as the caller wrote it.
    const scheme = /^[A-Za-z][A-Za-z0-9+.-]*:\/\//.exec(url)?.[0];
    return scheme === undefined ? shown : scheme + shown.slice(scheme.length);
  } catch {
    // Not a parseable URL: still cut at the first query or fragment.
    return url.split(/[?#]/)[0] ?? url;
  }
}

/**
 * Drops the query and fragment of every URL in a text: a signed link's query
 * is a credential. A URL can hold spaces in its path, so a run may span spaces
 * (it never reaches a second URL) and must end at its query or fragment; each
 * URL found is redacted with the URL class, in `redactUrl`.
 */
export function stripUrlQueries(text: string): string {
  return text.replace(/https?:\/\/[^\s?#]+(?:\s(?!https?:\/\/)[^\s?#]+)*[?#]\S*/gi, (matched) =>
    redactUrl(matched),
  );
}

/** The NDI error for a job that did not succeed: kind, id, status and why. */
function failedJobError(job: DocumentJob): NaceError {
  const error = job.error;
  const code = isRecord(error) && typeof error.code === "string" ? error.code : undefined;
  const message = isRecord(error) && typeof error.message === "string" ? error.message : undefined;
  // NDI can repeat the source URL; the query of a signed link is a credential.
  const detail = stripUrlQueries([code, message].filter((part) => part !== undefined).join(": "));
  return new NaceError(
    clip(`NDI ${job.kind} job ${job.job_id} ${job.status}${detail ? `: ${detail}` : ""}`, 500),
  );
}

/**
 * The connection the NDI components share. Resolves credentials exactly like
 * `NaceClassifier` (NDI is never available through OpenRouter), uploads bytes
 * with an upload token, and deletes the job it created and the file it uploaded
 * after the read — a failed clean-up logs one `console.warn` line, never an error.
 */
export class NdiClient {
  readonly timeout: number;

  readonly #client: NaceClient;

  // Kept for the error scrub only; a private field never reaches serialization.
  readonly #apiKey: string;

  constructor(fields: NaceNdiFields = {}) {
    const explicitBaseUrl = fields.baseUrl?.trim() || undefined;
    let baseUrl = explicitBaseUrl ?? env("NACE_BASE_URL") ?? DEFAULT_BASE_URL;
    while (baseUrl.endsWith("/")) baseUrl = baseUrl.slice(0, -1);
    if (hasUserinfo(baseUrl)) {
      throw new NaceError(
        "`baseUrl` must not contain a user name or password; pass the key as `apiKey`.",
      );
    }
    if (isOpenRouter(baseUrl)) {
      throw new NaceError("NDI is not available through OpenRouter; use a Nace endpoint.");
    }
    const apiKey = resolveApiKey(fields.apiKey, baseUrl, explicitBaseUrl !== undefined);
    if (fields.timeout !== undefined && !(fields.timeout > 0)) {
      throw new NaceError("`timeout` must be a positive number of milliseconds.");
    }
    if (fields.maxRetries !== undefined && !(fields.maxRetries >= 0)) {
      throw new NaceError("`maxRetries` must be zero or more.");
    }
    this.timeout = fields.timeout ?? DEFAULT_JOB_TIMEOUT_MS;
    const maxRetries = fields.maxRetries ?? 2;
    let client: NaceClient;
    try {
      client = new NaceClient({
        apiKey,
        baseURL: baseUrl,
        timeout: HTTP_TIMEOUT_MS,
        // Like the classifier: `timeout` bounds each attempt, `maxRetries`
        // their number, and the SDK's total retry budget is turned off.
        retry: { maxRetries, timeoutMs: null },
        clientInfo: CLIENT_NAME,
        // Looked up at call time, so a polyfill installed after construction is
        // still used.
        fetch: attributedFetch(fields.fetch ?? ((input, init) => globalThis.fetch(input, init))),
      });
    } catch (error) {
      if (error instanceof SdkNaceError) throw new NaceError(error.message, { cause: error });
      throw error;
    }
    this.#client = client;
    this.#apiKey = apiKey;
  }

  /** Runs one SDK call and converts its errors. */
  async call<T>(fn: (client: NaceClient) => Promise<T>): Promise<T> {
    try {
      return await fn(this.#client);
    } catch (error) {
      const converted = convertError(error, NDI);
      // NDI can repeat a source URL, and a provider can echo the key it was
      // sent: the query of a signed link and the key itself are credentials.
      // Scrub both on the error and its whole cause chain.
      const key = this.#apiKey;
      const scrub = (text: string): string =>
        stripUrlQueries(key ? text.split(key).join("***") : text);
      let level: unknown = converted;
      for (let depth = 0; depth < 5 && level instanceof Error; depth += 1) {
        const scrubbed = scrub(level.message);
        if (scrubbed !== level.message) {
          try {
            // A plain assignment would throw on a getter-only message (a
            // native DOMException, an AbortError); defining an own property
            // does not, and a failure to set is ignored.
            Object.defineProperty(level, "message", {
              value: scrubbed,
              writable: true,
              configurable: true,
            });
          } catch {
            // Leave a message that cannot be set as it is.
          }
        }
        level = level.cause;
      }
      throw converted;
    }
  }

  /**
   * Gives `use` an NDI source for `source`. Deletes the uploaded file after `use`
   * (also after an error).
   */
  async withSource<T>(
    source: NdiSource,
    options: { fileName?: string; workspaceId?: string; signal?: AbortSignal },
    use: (source: Source) => Promise<T>,
  ): Promise<T> {
    if (typeof source === "string") {
      if (source.includes("://")) {
        if (!/^https:\/\//i.test(source)) {
          throw new NaceError("Only https:// URLs are supported.");
        }
        const fileName = options.fileName ?? urlFileName(source);
        if (!/\.[A-Za-z0-9]+$/.test(fileName)) {
          // Never the URL itself: the query of a signed link is a credential.
          throw new NaceError(
            "The source URL has no file name with an extension, such as .pdf. NDI picks the " +
              "parser from the extension. NaceDocumentLoader takes the name as fileName.",
          );
        }
        return use({ type: "url", url: source, file_name: fileName });
      }
      let bytes: Buffer;
      try {
        bytes = await readFile(source);
      } catch (error) {
        throw new NaceError(`Could not read the file: ${basename(source)}`, { cause: error });
      }
      return this.#uploaded(bytes, options.fileName ?? basename(source), options, use);
    }
    const fileName = options.fileName ?? fileNameOf(source);
    if (fileName === undefined) {
      throw new NaceError(
        "`source` bytes need `fileName` with the file's extension, " +
          "or a `File`, which carries its own name.",
      );
    }
    return this.#uploaded(source, fileName, options, use);
  }

  /**
   * Creates a job, waits for it, gives the finished job to `use`. Deletes the job
   * after `use` (also after an error or a timeout). The create and the polls run
   * under a signal that aborts at the deadline, `started + timeout`, which `use`
   * also receives for calls that must not outlive it.
   */
  async withJob<T>(
    create: (client: NaceClient, waitSeconds: number, signal: AbortSignal) => Promise<DocumentJob>,
    options: { signal?: AbortSignal },
    use: (job: DocumentJob, deadline: number) => Promise<T>,
  ): Promise<T> {
    const started = Date.now();
    const deadline = started + this.timeout;
    // Never `wait_seconds: 0`: zero-retention accounts refuse it, and they give
    // the result only in the create response.
    const waitSeconds = Math.max(1, Math.min(60, Math.floor(this.timeout / 1000)));
    let jobId: string | undefined;
    // The deadline is also a signal: it stops the create and the polls at the
    // configured timeout, the request in flight included, and no poll starts
    // after it. The caller's signal aborts it too; the clean-up delete never
    // carries it, so a job that outlived the deadline is still cancelled.
    const controller = new AbortController();
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, this.timeout);
    const caller = options.signal;
    const onCallerAbort = () => controller.abort();
    if (caller !== undefined) {
      if (caller.aborted) controller.abort();
      else caller.addEventListener("abort", onCallerAbort);
    }
    const signal = controller.signal;
    try {
      let job: DocumentJob;
      try {
        job = await this.call((client) => create(client, waitSeconds, signal));
      } catch (error) {
        // A refused create keeps the request and the error under an id in this
        // header; NDI's docs say to delete that id like any job.
        if (NaceAPIError.isInstance(error)) {
          const kept = error.headers?.get("x-drex-document-job")?.trim();
          if (kept) jobId = kept;
        }
        throw error;
      }
      jobId = job.job_id;
      if (!TERMINAL_STATUSES.has(String(job.status))) {
        const id = jobId;
        const left = Math.max(1, this.timeout - (Date.now() - started));
        job = await this.call((client) =>
          client.jobs
            .wait(id, { timeout: left, raiseOnFailure: false, signal })
            .catch((error: unknown) => {
              if (error instanceof JobTimeoutError) {
                throw this.#jobTimeout(id);
              }
              throw error;
            }),
        );
      }
      if (job.status !== "succeeded") throw failedJobError(job);
      if (job.result === null || job.result === undefined) {
        // `not_retained` on a zero-retention account whose job outlasted the
        // create wait, or `expired`: only the create response held the result.
        const state = job.result_state;
        throw new NaceError(
          `NDI ${job.kind} job ${job.job_id} finished, but its result is not available ` +
            `(${typeof state === "string" && state !== "" ? state : "no result"}). ` +
            "A zero-retention account gets the result only in the create response: " +
            "parse fewer pages or use a smaller document.",
        );
      }
      if (Date.now() > deadline) {
        // The SDK's wait checks its deadline only after each poll, so a result
        // that arrives after it still returns. The configured `timeout` bounds
        // the whole operation, a late result included.
        throw this.#jobTimeout(job.job_id);
      }
      return await use(job, deadline);
    } catch (error) {
      // The deadline stopped the request in flight: that abort is this timeout.
      if (timedOut) throw this.#jobTimeout(jobId);
      throw error;
    } finally {
      clearTimeout(timer);
      caller?.removeEventListener("abort", onCallerAbort);
      const id = jobId;
      if (id !== undefined) {
        // A delete also cancels a job that still runs. A clean-up failure must
        // not replace the result or the first error.
        try {
          await this.call((client) => client.jobs.delete(id));
        } catch {
          console.warn(`[langchain-nace] Could not delete NDI job ${id}.`);
        }
      }
    }
  }

  /**
   * Downloads one of a job's files (the full content of a truncated parse),
   * bounded by the deadline of the whole operation: the SDK's download holds no
   * deadline of its own, and a body that never ends would hang the read.
   */
  async download(jobId: string, path: string, deadline: number): Promise<Uint8Array> {
    const left = deadline - Date.now();
    if (left <= 0) throw this.#jobTimeout(jobId);
    const controller = new AbortController();
    let end: () => void = () => {};
    // The abort frees the request; the race ends the wait even when a body
    // that never ends ignores the abort.
    const late = new Promise<never>((_, reject) => {
      const timer = setTimeout(() => {
        controller.abort();
        reject(this.#jobTimeout(jobId));
      }, left);
      end = () => clearTimeout(timer);
    });
    const read = this.call((client) =>
      client.jobs.download(jobId, path, { signal: controller.signal }),
    );
    // A rejection that lands after the deadline ended the read must not go
    // unhandled.
    read.catch(() => {});
    try {
      return await Promise.race([read, late]);
    } finally {
      end();
    }
  }

  /** The NDI timeout error for a job (or its create) that ran past the configured `timeout`. */
  #jobTimeout(jobId: string | undefined): NaceAPITimeoutError {
    const who = jobId === undefined ? "create" : `job ${jobId}`;
    const error = new NaceAPITimeoutError(`The NDI ${who} did not finish in ${this.timeout} ms.`);
    error.timeoutMs = this.timeout;
    return error;
  }

  /** Uploads bytes and gives `use` the workspace file, deleting it afterwards. */
  async #uploaded<T>(
    content: Blob | Uint8Array,
    fileName: string,
    options: { workspaceId?: string; signal?: AbortSignal },
    use: (source: Source) => Promise<T>,
  ): Promise<T> {
    const uploaded: UploadedFile = await this.call((client) =>
      client.documents.upload(
        content,
        {
          file_name: fileName,
          workspace_id: options.workspaceId,
          // A unique path prevents a `path_conflict` with a different file of
          // the same name.
          path: `${UPLOAD_PATH_PREFIX}/${crypto.randomUUID()}/${fileName}`,
        },
        { signal: options.signal },
      ),
    );
    try {
      return await use(workspaceFile(uploaded));
    } finally {
      try {
        await this.call((client) =>
          client.documents.deleteFile(uploaded.workspace_id, uploaded.file_id),
        );
      } catch {
        console.warn(`[langchain-nace] Could not delete NDI file ${uploaded.file_id}.`);
      }
    }
  }
}
