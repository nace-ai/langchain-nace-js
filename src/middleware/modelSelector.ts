/**
 * Pick the chat model for an agent run with Drex Router (`POST /v1/route`).
 *
 * Selection runs in `wrapModelCall`, where the system message, the tools and
 * the messages are all visible. The decision is kept in checkpointed state
 * under `modelSelection`, with an `anchor` that names the human turn it belongs
 * to. There is no `beforeAgent` hook: on langchain 1.3 that hook's update
 * shadows later state updates.
 *
 * @experimental The API may change without notice.
 */
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { Command } from "@langchain/langgraph";
import { createMiddleware, initChatModel } from "langchain";
import * as z from "zod/v4";

import { NaceError } from "../errors.js";
import {
  type CandidateConstraint,
  type CandidateProfile,
  cancellationError,
  filterEligibleCandidates,
  type ModelSelection,
  type ModelSelectOptions,
  NaceModelSelector,
  type NaceModelSelectorFields,
  projectRouteMessages,
  type SelectionContextMode,
  type SelectionPriority,
  type SelectOn,
  type SelectorCandidate,
  selectionAnchor,
  settleSelection,
  wireCandidates,
} from "../selection.js";

/** A model the selector can pick, and when it may be picked. */
export interface ModelCandidate {
  /** A chat model, or a model name accepted by `initChatModel`. */
  model: string | BaseChatModel;
  /** Sent instead of a bare id when Drex should see this model's profile. */
  profile?: CandidateProfile;
  /** When false, the candidate is skipped for a request that contains an image. Default: true. */
  images?: boolean;
  /** When false, the candidate is skipped for a request that has tools. Default: true. */
  tools?: boolean;
}

export interface NaceModelSelectorMiddlewareConfig {
  /** Candidate ids mapped to the models the agent may run. At most 255. */
  candidates: Record<string, ModelCandidate>;
  /** Default: `"balanced"`. */
  priority?: SelectionPriority;
  /**
   * The candidate used when Drex does not choose one. `null` keeps the agent's
   * own model. Default: `null`.
   */
  fallback?: string | null;
  /** Below this probability, use the fallback. */
  minProbability?: number;
  /** `"run"` selects once per run. `"model_call"` selects on every model call. Default: `"run"`. */
  selectOn?: SelectOn;
  /** `"conversation"` sends the turns. `"latest_human"` sends the system text and the latest human turn. */
  context?: SelectionContextMode;
  /** Deadline for the route call, in milliseconds. Default: 3000. */
  timeout?: number;
  /** A configured selector. When omitted, one is built from the fields below. */
  selector?: NaceModelSelector;
  /** Passed to the selector built when `selector` is omitted. */
  apiKey?: NaceModelSelectorFields["apiKey"];
  baseUrl?: NaceModelSelectorFields["baseUrl"];
  fetch?: NaceModelSelectorFields["fetch"];
}

interface Binding {
  id: string;
  model: string | BaseChatModel;
  images: boolean;
  tools: boolean;
  profile?: CandidateProfile;
}

const stateSchema = z.object({
  modelSelection: z.custom<ModelSelection | null>().optional(),
});

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Select the model for each agent run with Drex Router.
 *
 * The first model call of a human turn sends the conversation and the eligible
 * candidates to Drex. Later model calls of that turn reuse the choice when the
 * anchor matches and the model is still eligible, unless `selectOn` is
 * `"model_call"`. A failed route call uses `fallback`.
 *
 * @example
 * ```ts
 * const selector = naceModelSelectorMiddleware({
 *   candidates: {
 *     fast: { model: "anthropic:claude-haiku-5-5" },
 *     strong: { model: "anthropic:claude-sonnet-5-5" },
 *   },
 *   fallback: "fast",
 * });
 * ```
 */
export function naceModelSelectorMiddleware(config: NaceModelSelectorMiddlewareConfig) {
  const selectOn = config.selectOn ?? "run";
  if (selectOn !== "run" && selectOn !== "model_call") {
    throw new NaceError('naceModelSelectorMiddleware: `selectOn` must be "run" or "model_call".');
  }
  const contextMode = config.context ?? "conversation";
  if (contextMode !== "conversation" && contextMode !== "latest_human") {
    throw new NaceError(
      'naceModelSelectorMiddleware: `context` must be "conversation" or "latest_human".',
    );
  }
  const bindings = new Map<string, Binding>();
  for (const [id, candidate] of Object.entries(config.candidates ?? {})) {
    if (
      !isRecord(candidate) ||
      (typeof candidate.model !== "string" && !isRecord(candidate.model))
    ) {
      throw new NaceError(
        `naceModelSelectorMiddleware: candidate "${id}" needs a model name or a chat model.`,
      );
    }
    if (typeof candidate.model === "string" && candidate.model.trim() === "") {
      throw new NaceError(
        `naceModelSelectorMiddleware: candidate "${id}" has an empty model name.`,
      );
    }
    if (candidate.images !== undefined && typeof candidate.images !== "boolean") {
      throw new NaceError(
        `naceModelSelectorMiddleware: candidate "${id}" has an invalid images flag.`,
      );
    }
    if (candidate.tools !== undefined && typeof candidate.tools !== "boolean") {
      throw new NaceError(
        `naceModelSelectorMiddleware: candidate "${id}" has an invalid tools flag.`,
      );
    }
    bindings.set(id, {
      id,
      model: candidate.model as string | BaseChatModel,
      images: candidate.images !== false,
      tools: candidate.tools !== false,
      ...(candidate.profile !== undefined
        ? { profile: candidate.profile as CandidateProfile }
        : {}),
    });
  }
  if (bindings.size === 0) {
    throw new NaceError(
      "naceModelSelectorMiddleware: `candidates` must contain at least one model.",
    );
  }
  if (bindings.size > 255) {
    throw new NaceError(
      "naceModelSelectorMiddleware: `candidates` must contain at most 255 models.",
    );
  }
  wireCandidates(
    [...bindings.values()].map((binding) =>
      binding.profile ? { id: binding.id, ...binding.profile } : binding.id,
    ),
  );
  if (config.fallback != null && !bindings.has(config.fallback)) {
    throw new NaceError(
      `naceModelSelectorMiddleware: fallback "${config.fallback}" is not one of the candidates.`,
    );
  }
  const selector =
    config.selector ??
    new NaceModelSelector({
      apiKey: config.apiKey,
      baseUrl: config.baseUrl,
      fetch: config.fetch,
      timeout: config.timeout,
      priority: config.priority,
      minProbability: config.minProbability,
    });
  const priority = config.priority ?? selector.priority;
  const minProbability = config.minProbability ?? selector.minProbability;
  const timeout = config.timeout ?? selector.timeout;

  const models = new Map<string, Promise<BaseChatModel>>();
  function modelFor(id: string): Promise<BaseChatModel> {
    const cached = models.get(id);
    if (cached) return cached;
    const declared = bindings.get(id)!.model;
    const pending = (
      typeof declared === "string"
        ? (initChatModel(declared) as unknown as Promise<BaseChatModel>)
        : Promise.resolve(declared)
    ).catch((error: unknown) => {
      models.delete(id);
      throw error;
    });
    models.set(id, pending);
    return pending;
  }

  /**
   * The only call to the model handler. One `wrapModelCall` reaches it once,
   * except when the run was cancelled.
   */
  async function callModel<Request extends { model: unknown }>(
    request: Request,
    handler: (request: Request) => Promise<unknown> | unknown,
    selection: ModelSelection,
  ) {
    const id = selection.model;
    const outgoing =
      id !== null && bindings.has(id)
        ? ({ ...request, model: await modelFor(id) } as Request)
        : request;
    const response = await handler(outgoing);
    return new Command({ update: { modelSelection: selection, messages: [response] } });
  }

  interface AssessedRequest {
    messages?: unknown[];
    tools?: unknown[];
    systemMessage?: unknown;
    systemPrompt?: string;
    responseFormat?: unknown;
    runtime?: { signal?: AbortSignal };
  }

  function assess(request: AssessedRequest) {
    const messages = request.messages ?? [];
    const tools = request.tools ?? [];
    const anchor = selectionAnchor(messages);
    const projected = projectRouteMessages(
      { systemMessage: request.systemMessage, systemPrompt: request.systemPrompt, messages },
      contextMode,
    );
    const filterMessages =
      request.systemMessage === undefined ? messages : [request.systemMessage, ...messages];
    const constraints: CandidateConstraint[] = [...bindings.values()].map((binding) => ({
      id: binding.id,
      images: binding.images,
      tools: binding.tools,
      ...(binding.profile ? { context: binding.profile.context } : {}),
    }));
    const eligible = filterEligibleCandidates(constraints, {
      messages: filterMessages,
      tools,
      ...(request.responseFormat != null ? { responseFormat: request.responseFormat } : {}),
    });
    const fallbackId =
      config.fallback != null && eligible.includes(config.fallback) ? config.fallback : null;
    return { messages, anchor, projected, eligible, fallbackId, tools };
  }

  function isPlainObject(value: unknown): value is Record<string, unknown> {
    if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
    const proto = Object.getPrototypeOf(value);
    return proto === Object.prototype || proto === null;
  }

  /**
   * A checkpoint record is reused only when it is a plain object, its anchor
   * matches, and `model` is null or a configured candidate that is still eligible.
   * Any other shape is ignored. A throwing getter must not escape.
   */
  function reusable(
    stored: unknown,
    anchor: string,
    eligible: readonly string[],
  ): ModelSelection | undefined {
    try {
      if (!isPlainObject(stored) || stored.anchor !== anchor) return undefined;
      const model = stored.model;
      if (model === null) return stored as unknown as ModelSelection;
      if (typeof model !== "string" || !bindings.has(model) || !eligible.includes(model)) {
        return undefined;
      }
      return stored as unknown as ModelSelection;
    } catch {
      return undefined;
    }
  }

  async function choose(request: AssessedRequest): Promise<ModelSelection> {
    const signal = request.runtime?.signal;
    if (signal?.aborted) throw cancellationError(signal);
    const { anchor, projected, eligible, fallbackId } = assess(request);
    const knownIds = [...bindings.keys()];
    const base = { priority, sentIds: eligible, minProbability, fallbackId, anchor };
    let selection: ModelSelection;
    if (projected === null) {
      selection = settleSelection({ type: "no_input" }, { ...base, sentIds: [] }, knownIds);
    } else if (eligible.length === 0) {
      selection = settleSelection(
        { type: "no_eligible_candidate" },
        { ...base, sentIds: [] },
        knownIds,
      );
    } else if (eligible.length === 1) {
      selection = settleSelection({ type: "single_candidate", id: eligible[0]! }, base, knownIds);
    } else {
      const candidates: SelectorCandidate[] = eligible.map((id) => {
        const binding = bindings.get(id)!;
        return binding.profile ? { id, ...binding.profile } : id;
      });
      const call: ModelSelectOptions = {
        candidates,
        priority,
        minProbability,
        fallback: fallbackId,
        timeout,
        ...(signal ? { signal } : {}),
      };
      selection = await selector.select(projected, call);
    }
    return { ...selection, anchor };
  }

  return createMiddleware({
    name: "NaceModelSelectorMiddleware",
    stateSchema,

    async wrapModelCall(request, handler) {
      const signal = request.runtime?.signal;
      if (signal?.aborted) throw cancellationError(signal);
      const { anchor, eligible } = assess(request);
      const stored = (request.state as { modelSelection?: unknown }).modelSelection;
      const selection =
        selectOn === "run"
          ? (reusable(stored, anchor, eligible) ?? (await choose(request)))
          : await choose(request);
      return callModel(request, handler, selection);
    },
  });
}
