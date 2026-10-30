import { AsyncLocalStorage } from "node:async_hooks";

import {
  Runnable,
  type RunnableBatchOptions,
  type RunnableConfig,
} from "@langchain/core/runnables";
import { getEnvironmentVariable } from "@langchain/core/utils/env";
import { type ImageInput, NaceClient, NaceError as SdkNaceError } from "nace-sdk";

import {
  convertError,
  NaceAPIError,
  NaceBatchStoppedError,
  NaceError,
  NaceImageError,
  NaceImageModelError,
} from "./errors.js";
import { asDataUrl, type ImageLike, toDrexImage } from "./images.js";
import { encodeState, type State } from "./state.js";
import {
  type ClassificationResponse,
  parseResponse,
  type Question,
  type Questions,
  type RawQuestion,
  toWire,
  validateQuestions,
} from "./types.js";
import { CLIENT_NAME } from "./version.js";

/**
 * The default model is Drex Max, moved to the newest flagship with each Drex
 * release. Pin a model with `model` or `NACE_DEFAULT_DECISION_MODEL`.
 */
export const DEFAULT_MODEL = "drex-max";
export const DEFAULT_BASE_URL = "https://console.nace.ai";
/** Drex may take up to 55 s to answer, so each attempt gets 60 s by default. */
export const DEFAULT_TIMEOUT_MS = 60_000;
const DEFAULT_MAX_CONCURRENCY = 8;

/** Request fields the classifier sets itself; `modelKwargs` can't override them. */
const RESERVED_BODY_FIELDS = ["model", "state", "questions", "images"] as const;

/**
 * Released, pinned versions known to read text only. Facts about pinned
 * versions never change; every other model id is sent as given and Drex decides.
 */
const TEXT_ONLY_MODELS = new Set(["drex-v1.5", "drex-v2.0"]);

/** Whether `model` is a released version known to read text only. */
export function isTextOnly(model: string): boolean {
  const name = model.split("/").pop() ?? model;
  return TEXT_ONLY_MODELS.has(name.replace(/^~/, ""));
}

/** An env var, with an empty or blank value read as unset. */
function env(name: string): string | undefined {
  const value = getEnvironmentVariable(name)?.trim();
  return value ? value : undefined;
}

function isOpenRouter(baseUrl: string): boolean {
  try {
    return new URL(baseUrl).hostname.replace(/\.$/, "") === "openrouter.ai";
  } catch {
    return false;
  }
}

const NACE_HOSTS = new Set(["console.nace.ai", "drex.nace.ai"]);

function isNaceDestination(baseUrl: string): boolean {
  try {
    const url = new URL(baseUrl);
    return (
      url.protocol === "https:" &&
      url.port === "" &&
      !url.username &&
      !url.password &&
      NACE_HOSTS.has(url.hostname.replace(/\.$/, ""))
    );
  } catch {
    return false;
  }
}

function hasUserinfo(baseUrl: string): boolean {
  try {
    const url = new URL(baseUrl);
    return url.username !== "" || url.password !== "";
  } catch {
    return false;
  }
}

/**
 * The key for `baseUrl`. An explicit `apiKey` always wins. Otherwise the key
 * follows the endpoint: `OPENROUTER_API_KEY` for the exact OpenRouter host (never
 * the Nace key), and `NACE_API_KEY` for the Nace hosts or an endpoint that also
 * came from the environment (`NACE_BASE_URL`, e.g. on-prem). A custom `baseUrl`
 * passed in code needs its own `apiKey`.
 */
function resolveApiKey(
  explicit: string | undefined,
  baseUrl: string,
  explicitBaseUrl: boolean,
): string {
  if (explicit !== undefined) {
    const key = explicit.trim();
    if (!key) throw new NaceError("`apiKey` must not be empty.");
    return key;
  }
  if (isOpenRouter(baseUrl)) {
    const key = env("OPENROUTER_API_KEY");
    if (!key) throw new NaceError("OpenRouter needs `apiKey` or `OPENROUTER_API_KEY`.");
    return key;
  }
  if (explicitBaseUrl && !isNaceDestination(baseUrl)) {
    throw new NaceError(
      "A custom `baseUrl` needs an explicit `apiKey`. To use `NACE_API_KEY`, " +
        "set the endpoint with `NACE_BASE_URL` instead of `baseUrl`.",
    );
  }
  const key = env("NACE_API_KEY");
  if (!key) {
    throw new NaceError("A Drex API key is required. Pass `apiKey` or set `NACE_API_KEY`.");
  }
  return key;
}

function providerHeaders(baseUrl: string, headers?: HeadersInit): Headers {
  const result = new Headers(headers);
  if (!isNaceDestination(baseUrl)) {
    result.delete("x-vercel-protection-bypass");
    result.delete("x-vercel-set-bypass-cookie");
  }
  return result;
}

export interface NaceClassifierFields {
  /**
   * The questions to ask, keyed by the id their answers come back under.
   * Optional here when every call passes `questions` in its options.
   */
  questions?: Questions;
  /**
   * Drex model id, sent as given. Defaults to `NACE_DEFAULT_DECISION_MODEL`,
   * else `drex-max` (reads images; up to 512 questions per request).
   * `drex-v2.0` is text-only and has the same question-count limit.
   */
  model?: string;
  /**
   * Provider key. Defaults to `NACE_API_KEY` for the Nace hosts and for an
   * endpoint set with `NACE_BASE_URL`; `OPENROUTER_API_KEY` for OpenRouter.
   */
  apiKey?: string;
  /**
   * API root. Defaults to `NACE_BASE_URL`, else `https://console.nace.ai`.
   * Also works with OpenRouter (`https://openrouter.ai/api/v1`) and other
   * TypeSafe-compatible hosts.
   */
  baseUrl?: string;
  /**
   * Per-attempt timeout in **milliseconds**. Defaults to 60000, because Drex may
   * take up to 55 s; a smaller value makes slow decisions time out.
   */
  timeout?: number;
  /** Retries for 408, 429, 5xx and 529, honouring `retry-after-ms`. Defaults to 2. */
  maxRetries?: number;
  /** Requests in flight during `batch`, unless the call sets its own. Defaults to 8. */
  maxConcurrency?: number;
  /** Custom transport, for tests, proxies or shared connection pools. */
  fetch?: typeof fetch;
  /** Allow use in a browser, exposing the API key to page users. Defaults to `false`. */
  dangerouslyAllowBrowser?: boolean;
  /** Send OpenRouter's request format. Inferred from `baseUrl`. */
  openrouter?: boolean;
  /** Extra top-level request body fields, for options Drex adds later. */
  modelKwargs?: Record<string, unknown>;
  /** Extra request headers. Transport-only: omitted from inspection and serialization. */
  defaultHeaders?: Record<string, string>;
}

export interface NaceClassifierCallOptions extends RunnableConfig {
  /**
   * Images every question sees, in order, as "Image 1", "Image 2", and so on.
   * Images found in messages are numbered after these. Image models only (such
   * as `drex-max`).
   */
  images?: ImageLike[];
  /** Questions for this call, in place of the constructor's. */
  questions?: Questions;
}

/**
 * Ask Drex typed questions about text, JSON, messages or images.
 *
 * Questions about one state are answered in one request. Answers keep their
 * full probabilities, so thresholds stay in your code.
 *
 * Any model id is accepted and sent as given; Drex enforces each model's
 * limits and the error explains what to change. Images are refused before
 * sending only for models known to read text only (`drex-v1.5`, `drex-v2.0`).
 *
 * @example
 * ```ts
 * const classifier = new NaceClassifier({
 *   questions: {
 *     refund: { type: "noul", instructions: "Is the customer asking for a refund?" },
 *     team: {
 *       type: "choice",
 *       instructions: "Which team should handle this?",
 *       criteria: { billing: "Payments", technical: "Bugs" },
 *     },
 *   },
 * });
 * const response = await classifier.invoke("I was charged twice for March.");
 * response.nouls.refund.noul;           // probability of yes
 * response.choices.team.probabilities;  // every label's probability
 * ```
 */
export class NaceClassifier extends Runnable<
  State,
  ClassificationResponse,
  NaceClassifierCallOptions
> {
  static lc_name(): string {
    return "NaceClassifier";
  }

  lc_serializable = true;

  lc_namespace = ["langchain", "classifiers", "nace"];

  get lc_secrets(): Record<string, string> {
    return {
      apiKey: isNaceDestination(this.baseUrl)
        ? "NACE_API_KEY"
        : isOpenRouter(this.baseUrl)
          ? "OPENROUTER_API_KEY"
          : "NACE_CUSTOM_API_KEY",
    };
  }

  readonly questions: Questions | undefined;

  readonly model: string;

  readonly baseUrl: string;

  readonly timeout: number;

  readonly maxRetries: number;

  readonly maxConcurrency: number;

  readonly openrouter: boolean;

  readonly modelKwargs: Record<string, unknown>;

  /** Non-enumerable, so `console.log(classifier)` never prints it. */
  declare protected apiKey: string;

  declare protected client: NaceClient;

  constructor(fields: NaceClassifierFields = {}) {
    // `lc_kwargs` is enumerable and serialized. Custom headers can hold arbitrary
    // credentials, so keep them transport-only alongside the key and fetch.
    const { apiKey: _key, fetch: _fetch, defaultHeaders: _headers, ...kwargs } = fields;
    const explicitBaseUrl = fields.baseUrl?.trim() || undefined;
    let baseUrl = explicitBaseUrl ?? env("NACE_BASE_URL") ?? DEFAULT_BASE_URL;
    while (baseUrl.endsWith("/")) baseUrl = baseUrl.slice(0, -1);
    const defaultHeaders = Object.fromEntries(providerHeaders(baseUrl, fields.defaultHeaders));
    super({ ...kwargs, baseUrl });

    if (hasUserinfo(baseUrl)) {
      throw new NaceError(
        "`baseUrl` must not contain a user name or password; pass the key as `apiKey`.",
      );
    }
    const apiKey = resolveApiKey(fields.apiKey, baseUrl, explicitBaseUrl !== undefined);
    const model = (fields.model ?? env("NACE_DEFAULT_DECISION_MODEL") ?? DEFAULT_MODEL).trim();
    if (!model) {
      throw new NaceError("`model` must not be empty.");
    }
    if (fields.timeout !== undefined && !(fields.timeout > 0)) {
      throw new NaceError("`timeout` must be a positive number of milliseconds.");
    }
    if (fields.maxRetries !== undefined && !(fields.maxRetries >= 0)) {
      throw new NaceError("`maxRetries` must be zero or more.");
    }
    for (const key of RESERVED_BODY_FIELDS) {
      if (fields.modelKwargs !== undefined && Object.hasOwn(fields.modelKwargs, key)) {
        throw new NaceError(
          `\`modelKwargs\` can't set "${key}"; it is set from the model, state, questions and images.`,
        );
      }
    }
    this.questions =
      fields.questions === undefined ? undefined : validateQuestions(fields.questions);
    this.model = model;

    this.baseUrl = baseUrl;
    this.openrouter = fields.openrouter ?? isOpenRouter(baseUrl);
    this.timeout = fields.timeout ?? DEFAULT_TIMEOUT_MS;
    this.maxRetries = fields.maxRetries ?? 2;
    this.maxConcurrency = fields.maxConcurrency ?? DEFAULT_MAX_CONCURRENCY;
    this.modelKwargs = { ...fields.modelKwargs };

    Object.defineProperty(this, "apiKey", {
      value: apiKey,
      enumerable: false,
      writable: false,
      configurable: true,
    });

    const headers: Record<string, string> = { ...defaultHeaders };
    if (this.openrouter) {
      headers["http-referer"] ??= "https://nace.ai";
      headers["x-openrouter-title"] ??= "Drex for LangChain";
    }
    // Looked up at call time, so a later `vi.spyOn(globalThis, "fetch")` or a
    // polyfill installed after construction is still used.
    const inner = fields.fetch ?? ((input, init) => globalThis.fetch(input, init));
    // nace-sdk sets `User-Agent: nace-sdk/<version>` after merging our headers,
    // so the attribution `User-Agent` is set here, on the way out.
    const withUserAgent = (input: string, init?: RequestInit) => {
      const outgoing = providerHeaders(input, init?.headers);
      outgoing.set("User-Agent", CLIENT_NAME);
      // Do not let fetch forward provider-specific headers through a redirect.
      // Use the final API URL explicitly instead.
      return inner(input, { ...init, headers: outgoing, redirect: "error" });
    };

    let client: NaceClient;
    try {
      client = new NaceClient({
        apiKey,
        baseURL: this.sdkBaseUrl,
        defaultModel: model,
        timeout: this.timeout,
        // The SDK's total retry budget (30 s) would skip the retry after a slow
        // failed attempt; `timeout` bounds each attempt and `maxRetries` their
        // number, so the budget is turned off.
        retry: { maxRetries: this.maxRetries, timeoutMs: null },
        clientInfo: CLIENT_NAME,
        defaultHeaders: headers,
        dangerouslyAllowBrowser: fields.dangerouslyAllowBrowser ?? false,
        fetch: withUserAgent,
      });
    } catch (error) {
      if (error instanceof SdkNaceError) throw new NaceError(error.message, { cause: error });
      throw error;
    }
    Object.defineProperty(this, "client", {
      value: client,
      enumerable: false,
      writable: false,
      configurable: true,
    });
  }

  /** Who answers, for error messages. */
  get provider(): string {
    return this.openrouter ? "OpenRouter" : "Drex";
  }

  /** OpenRouter documents `.../api/v1`; the SDK appends `/v1/systemone` itself. */
  private get sdkBaseUrl(): string {
    if (this.openrouter && this.baseUrl.endsWith("/v1")) return this.baseUrl.slice(0, -3);
    return this.baseUrl;
  }

  /**
   * Answer one request.
   *
   * @throws NaceImageModelError when images go to a text-only model (nothing is sent).
   * @throws NaceAPIError (a subclass per status) when Drex refuses the request.
   * @throws NaceAPIConnectionError when Drex can't be reached.
   */
  async invoke(
    input: State,
    options?: Partial<NaceClassifierCallOptions>,
  ): Promise<ClassificationResponse> {
    const config = {
      ...options,
      metadata: {
        ...options?.metadata,
        ls_provider: "nace",
        ls_model_name: this.model,
        ls_model_type: "chat",
      },
    };
    // The config `_callWithConfig` hands over carries the run's signal, which also
    // covers `timeout` and a cancelled parent run.
    const run = (state: State, runConfig?: { signal?: AbortSignal }) =>
      this.classify(state, { ...options, signal: runConfig?.signal ?? options?.signal });
    return this._callWithConfig(run, input, {
      ...config,
      runType: "llm",
    } as Partial<NaceClassifierCallOptions> & { runType: string });
  }

  /**
   * Answer many requests with LangChain's own `batch`: each input goes through
   * `invoke` (so an overridden `invoke` applies), at most `maxConcurrency` (default
   * 8) at a time, in input order. After a 401, 402 or 403, or once the batch has
   * rejected, no new request is sent: requests already sent finish, including the
   * SDK's own retries of them, and their results are discarded when the batch
   * rejects. An input that was not sent fails with a `NaceBatchStoppedError` in its
   * own run. A 401, 402 or 403 stops the batch it happens in, and the batches nested
   * in it. Another batch running at the same time stops at its own first 401, 402 or
   * 403. Such a failure means the classifier's key or account can't be used, so
   * dispatch stops at once. After any other failure the batch stops when it rejects,
   * so an input dispatched in between (for example while the failed input's error
   * callbacks run) can still be sent. A batch an `invoke` override starts with
   * `this.batch` runs inside the batch it came from: it stops with it, and a terminal
   * failure inside it stops that batch too; batches started on their own stay
   * independent. An `invoke` override that recovers from an error keeps the batch
   * going, except after a 401, 402 or 403, which always stop it. A cancelled input
   * fails alone.
   */
  override async batch(
    inputs: State[],
    options?: Partial<NaceClassifierCallOptions> | Partial<NaceClassifierCallOptions>[],
    batchOptions?: RunnableBatchOptions & { returnExceptions?: false },
  ): Promise<ClassificationResponse[]>;

  override async batch(
    inputs: State[],
    options?: Partial<NaceClassifierCallOptions> | Partial<NaceClassifierCallOptions>[],
    batchOptions?: RunnableBatchOptions & { returnExceptions: true },
  ): Promise<(ClassificationResponse | Error)[]>;

  override async batch(
    inputs: State[],
    options?: Partial<NaceClassifierCallOptions> | Partial<NaceClassifierCallOptions>[],
    batchOptions?: RunnableBatchOptions,
  ): Promise<(ClassificationResponse | Error)[]> {
    // A batch started while one of this classifier's batches is running (an `invoke`
    // override answering its input with `this.batch`) runs inside that batch's fate.
    const active = batchContext.getStore();
    const state: BatchState = {
      owner: this,
      parent: active?.owner === this ? active : undefined,
      stopped: false,
      stops: new WeakSet(),
    };
    // Everything below runs in this batch's context, which `classify` reads right
    // before it sends: the stop reaches it whatever an overridden `invoke` does with
    // the options, and every input keeps its own run.
    return batchContext.run(state, () =>
      super
        .batch(inputs, options, {
          ...batchOptions,
          maxConcurrency: batchOptions?.maxConcurrency ?? this.maxConcurrency,
        })
        .catch((error: unknown) => {
          // Another input's stop may reject first; report the failure behind it, which
          // may be an enclosing batch's.
          const reason = state.stops.has(error as object) ? (error as Error).cause : error;
          state.stopped = true;
          state.cause ??= reason;
          throw reason;
        }),
    );
  }

  /** The request body fields for one call, before the SDK adds `model` defaults. */
  prepare(
    state: State,
    options?: Pick<NaceClassifierCallOptions, "images" | "questions">,
  ): {
    state: unknown;
    questions: Record<string, Record<string, unknown>>;
    images: ImageInput[];
  } {
    const source = options?.questions ?? this.questions;
    if (source === undefined) {
      throw new NaceError(
        "No questions to ask. Pass `questions` to the constructor or in the call options.",
      );
    }
    const checked = options?.questions === undefined ? source : validateQuestions(source);
    const textOnly = isTextOnly(this.model);
    const explicit = (options?.images ?? []).map((image) => toDrexImage(image));
    if (explicit.length > 0 && textOnly) {
      throw new NaceImageModelError(
        `${this.model} reads text only, so it can't take images. ` +
          'Use an image model such as model: "drex-max".',
      );
    }
    const encoded = encodeState(state, {
      images: textOnly ? "error" : "lift",
      firstIndex: explicit.length + 1,
    });
    const questions: Record<string, Record<string, unknown>> = {};
    for (const [id, question] of Object.entries(checked)) {
      Object.defineProperty(questions, id, {
        value: toWire(question as Question | RawQuestion),
        enumerable: true,
        writable: true,
        configurable: true,
      });
    }
    let images = [...explicit, ...encoded.images];
    let wireState = encoded.state;
    if (images.length > 0 && this.openrouter) {
      wireState = openRouterState(wireState, images);
      images = [];
    }
    return { state: wireState, questions, images };
  }

  protected async classify(
    state: State,
    options?: Partial<NaceClassifierCallOptions>,
  ): Promise<ClassificationResponse> {
    const prepared = this.prepare(state, options);
    const request: Record<string, unknown> = {
      ...this.modelKwargs,
      model: this.model,
      state: prepared.state,
      questions: prepared.questions,
    };
    if (prepared.images.length > 0) request.images = prepared.images;
    const batch = batchContext.getStore();
    const current = batch?.owner === this ? batch : undefined;
    const stopped = current ? stoppedBatch(current) : undefined;
    if (current && stopped) {
      const stop = new NaceBatchStoppedError(
        "Not sent: an earlier request in this batch failed, and this one would too.",
        { cause: stopped.cause },
      );
      current.stops.add(stop);
      throw stop;
    }
    // A cancelled input is not sent.
    options?.signal?.throwIfAborted();
    try {
      const { data, requestId } = await this.client
        // The SDK's request type is narrower than what Drex accepts (structured
        // or future question types); the wire form here is Drex's own.
        .systemOne(request as never, { signal: options?.signal })
        .withResponse();
      return parseResponse(data, requestId ?? undefined, prepared.questions);
    } catch (error) {
      const converted = convertError(error, this.provider);
      // Recorded before the error leaves the request, so before any callback runs.
      if (current && stopsBatch(converted)) {
        // The key or the account can't be used: this batch and the ones it runs in stop.
        for (let level: BatchState | undefined = current; level; level = level.parent) {
          level.stopped = true;
          level.cause ??= converted;
        }
      }
      throw converted;
    }
  }
}

/** Failures every later request in a batch would repeat: the key or the account. */
const BATCH_STOP_STATUSES = new Set([401, 402, 403]);

function stopsBatch(error: unknown): boolean {
  return NaceAPIError.isInstance(error) && BATCH_STOP_STATUSES.has(error.status);
}

/** What `batch` shares with the requests it dispatches. */
interface BatchState {
  owner: NaceClassifier;
  /** The batch of this classifier this one was started inside, if any. */
  parent?: BatchState;
  stopped: boolean;
  /** The failure that stopped the batch. */
  cause?: unknown;
  /** The stop errors this batch raised, to tell them from its inputs' own failures. */
  stops: WeakSet<object>;
}

/** The batch a request belongs to, if any; set by `batch`, read by `classify`. */
const batchContext = new AsyncLocalStorage<BatchState>();

/** The batch, or the nearest enclosing one, that has stopped. */
function stoppedBatch(batch: BatchState): BatchState | undefined {
  for (let level: BatchState | undefined = batch; level; level = level.parent) {
    if (level.stopped) return level;
  }
  return undefined;
}

/** OpenRouter takes images as `image_url` parts of a `state` content array. */
function openRouterState(state: unknown, images: ImageInput[]): unknown[] {
  const parts: unknown[] = [
    { type: "text", text: typeof state === "string" ? state : JSON.stringify(state) },
  ];
  for (const image of images) {
    if (typeof image !== "string") {
      throw new NaceImageError(
        "OpenRouter accepts inline images only, not uploaded workspace files.",
      );
    }
    parts.push({ type: "image_url", image_url: { url: asDataUrl(image) } });
  }
  return parts;
}
