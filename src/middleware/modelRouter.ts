/** Experimental model-routing middleware powered by Drex. */
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { createMiddleware, initChatModel } from "langchain";
import * as z from "zod/v4";

import { isTextOnly, NaceClassifier, type NaceClassifierFields } from "../classifier.js";
import { NaceAPIError, NaceError } from "../errors.js";
import type { State } from "../state.js";
import type { Choice, ChoiceAnswer, JsonValue, QuestionContent } from "../types.js";
import { isUserMessage, withoutMedia } from "./media.js";

const QUESTION_ID = "model_route";
/** Statuses after which routing is retried once with images replaced by placeholders. */
const IMAGE_RETRY_STATUSES = new Set([400, 413, 422]);

/** A model the router can pick, and when to pick it. */
export interface ModelChoice {
  /** A chat model, or a model name accepted by `initChatModel` (e.g. `"anthropic:claude-haiku-5-5"`). */
  model: string | BaseChatModel;
  /** The kind of task this model suits. Becomes the Choice criterion. */
  criteria: JsonValue;
}

/**
 * The routing decision kept in agent state under `modelRoute`. When Drex
 * answered, `choice`, `confidence` and `probabilities` are its answer.
 */
export interface NaceModelRoute {
  /** The choice whose model this run uses. */
  route: string;
  /** What Drex picked, or `null` when Drex couldn't be asked. */
  choice: string | null;
  confidence: number | null;
  probabilities: Record<string, number>;
  /** Whether `route` is `escalateTo` instead of Drex's pick. */
  escalated: boolean;
  reason: "confident" | "low_confidence" | "error";
}

export interface NaceModelRouterMiddlewareConfig {
  /** Route names mapped to models, cheapest first. */
  choices: Record<string, ModelChoice>;
  /** The routing question, e.g. "Pick the cheapest model that can do this task well." */
  instructions: QuestionContent;
  /** Below this confidence, escalate. Default 0.6. */
  minConfidence?: number;
  /** The route used when escalating. Defaults to the last choice. */
  escalateTo?: string;
  /** A configured classifier. Questions are passed per call, so any classifier works. */
  classifier?: NaceClassifier;
  /** Settings for the classifier the router builds when `classifier` is not given. */
  classifierOptions?: Omit<NaceClassifierFields, "questions">;
  /** Drex model id shortcut. Defaults to `drex-max`. */
  model?: string;
}

const stateSchema = z.object({
  modelRoute: z.custom<NaceModelRoute | null>().optional(),
});

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Pick the model for each agent run with a Drex `Choice`.
 *
 * Before each run, Drex reads the latest user message and picks one of
 * `choices`. When its confidence is below `minConfidence`, it picks a label
 * that isn't a choice, or Drex can't be reached, the run uses `escalateTo` (by
 * default the last choice, so list choices from cheapest to strongest). With no
 * user message, the agent's own model is used. The decision and its
 * probabilities are kept in state under `modelRoute`.
 *
 * Drop-in for `modelRouterMiddleware` from `@langchain/typesafe`: its
 * `choices`, `instructions` and `classifierOptions` work here.
 *
 * @experimental The API may change without notice.
 *
 * @example
 * ```ts
 * const router = naceModelRouterMiddleware({
 *   choices: {
 *     fast: { model: "anthropic:claude-haiku-5-5", criteria: "Simple, well-scoped tasks." },
 *     strong: { model: "anthropic:claude-sonnet-5-5", criteria: "Hard, multi-step work." },
 *   },
 *   instructions: "Pick the cheapest model that can do this task well.",
 * });
 * ```
 */
export function naceModelRouterMiddleware(config: NaceModelRouterMiddlewareConfig) {
  const routes = Object.keys(config.choices ?? {});
  if (routes.length === 0) {
    throw new NaceError("naceModelRouterMiddleware: `choices` must contain at least one model.");
  }
  for (const route of routes) {
    const choice = config.choices[route];
    if (!isRecord(choice) || (typeof choice.model !== "string" && !isRecord(choice.model))) {
      throw new NaceError(
        `naceModelRouterMiddleware: choice "${route}" needs a model name or a chat model.`,
      );
    }
    if (typeof choice.model === "string" && choice.model.trim() === "") {
      throw new NaceError(`naceModelRouterMiddleware: choice "${route}" has an empty model name.`);
    }
  }
  if (typeof config.instructions === "string" && config.instructions.trim() === "") {
    throw new NaceError("naceModelRouterMiddleware: `instructions` must not be empty.");
  }
  const escalateTo = config.escalateTo ?? routes[routes.length - 1]!;
  if (!Object.hasOwn(config.choices, escalateTo)) {
    throw new NaceError(
      `naceModelRouterMiddleware: escalateTo "${escalateTo}" is not one of the choices.`,
    );
  }
  const minConfidence = config.minConfidence ?? 0.6;
  const criteria: Record<string, JsonValue> = {};
  for (const route of routes) {
    Object.defineProperty(criteria, route, {
      value: config.choices[route]!.criteria,
      enumerable: true,
      writable: true,
      configurable: true,
    });
  }
  const question: Choice = { type: "choice", instructions: config.instructions, criteria };
  const classifier =
    config.classifier ??
    new NaceClassifier({
      ...config.classifierOptions,
      ...(config.model !== undefined ? { model: config.model } : {}),
    });
  const keepImages = !isTextOnly(classifier.model);

  // Cache the promise, not the model, so concurrent runs share one
  // `initChatModel` call; a failure is evicted so it isn't cached.
  const models = new Map<string, Promise<BaseChatModel>>();
  function modelFor(route: string): Promise<BaseChatModel> {
    const cached = models.get(route);
    if (cached) return cached;
    const declared = config.choices[route]!.model;
    const pending = (
      typeof declared === "string"
        ? (initChatModel(declared) as unknown as Promise<BaseChatModel>)
        : Promise.resolve(declared)
    ).catch((error: unknown) => {
      models.delete(route);
      throw error;
    });
    models.set(route, pending);
    return pending;
  }

  function decide(answer: ChoiceAnswer | undefined): NaceModelRoute {
    if (answer === undefined) {
      return {
        route: escalateTo,
        choice: null,
        confidence: null,
        probabilities: {},
        escalated: true,
        reason: "error",
      };
    }
    const confident =
      answer.confidence >= minConfidence && Object.hasOwn(config.choices, answer.choice);
    return {
      route: confident ? answer.choice : escalateTo,
      choice: answer.choice,
      confidence: answer.confidence,
      probabilities: { ...answer.probabilities },
      escalated: !confident,
      reason: confident ? "confident" : "low_confidence",
    };
  }

  return createMiddleware({
    name: "NaceModelRouterMiddleware",
    stateSchema,

    async beforeAgent(state, runtime) {
      const signal = (runtime as { signal?: AbortSignal } | undefined)?.signal;
      const messages = (state as { messages?: unknown[] }).messages ?? [];
      const latest = [...messages].reverse().find(isUserMessage);
      if (latest === undefined) return { modelRoute: null };
      const attempts = [withoutMedia(latest, { keepImages })];
      if (keepImages) {
        const text = withoutMedia(latest, { keepImages: false });
        if (text !== attempts[0]) attempts.push(text);
      }
      let answer: ChoiceAnswer | undefined;
      for (const [index, attempt] of attempts.entries()) {
        try {
          const response = await classifier.invoke(attempt as State, {
            questions: { [QUESTION_ID]: question },
            runName: "NaceModelRouter",
            metadata: { nace_min_confidence: minConfidence, nace_escalate_to: escalateTo },
            ...(signal ? { signal } : {}),
          });
          answer = response.choices[QUESTION_ID];
          break;
        } catch (error) {
          const retry =
            index + 1 < attempts.length &&
            NaceAPIError.isInstance(error) &&
            IMAGE_RETRY_STATUSES.has(error.status);
          if (retry) continue;
          // A cancelled run stays cancelled rather than escalating.
          if (signal?.aborted) throw error;
          const name = isRecord(error) && typeof error.name === "string" ? error.name : "Error";
          console.warn(`[@nace-ai/langchain-nace] Drex routing failed; escalating (${name}).`);
          break;
        }
      }
      return { modelRoute: decide(answer) };
    },

    async wrapModelCall(request, handler) {
      const route = (request.state as { modelRoute?: NaceModelRoute | null }).modelRoute;
      if (!route) return handler(request);
      return handler({ ...request, model: await modelFor(route.route) });
    },
  });
}
