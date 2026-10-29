import {
  Runnable,
  type RunnableConfig,
  type RunnableInterface,
  RunnableLambda,
} from "@langchain/core/runnables";

import { NaceClassifier, type NaceClassifierFields } from "./classifier.js";
import { NaceError } from "./errors.js";
import type { State } from "./state.js";
import type { Choice, ChoiceAnswer } from "./types.js";

const QUESTION_ID = "route";

/** Drex was less confident than `minConfidence` and no fallback was set. */
export class NaceLowConfidenceError extends NaceError {
  override readonly name: string = "NaceLowConfidenceError";
}

/** Which route Drex picked, with the full answer. */
export interface RouteDecision {
  /** The selected label, or the fallback when confidence was too low. */
  route: string;
  /** Drex's answer, with every label's probability. */
  answer: ChoiceAnswer;
  /** Whether `route` is the fallback because confidence was below the threshold. */
  fellBack: boolean;
}

export interface NaceRouterFields<RunInput = State> {
  /** The routing question. Its labels are the routes. */
  question: Choice;
  /** A configured classifier. Defaults to one built from `classifierOptions`. */
  classifier?: NaceClassifier;
  /** Settings for the classifier the router builds when `classifier` is not given. */
  classifierOptions?: Omit<NaceClassifierFields, "questions">;
  /** Turns the input into Drex state. Defaults to the input itself. */
  state?: (input: RunInput) => State;
  /** Below this confidence, use `fallback` (or throw when there is none). */
  minConfidence?: number;
  /** The route to use when confidence is below `minConfidence`. */
  fallback?: string;
}

/**
 * Pick a route for an input with a Drex `Choice` question.
 *
 * Use it on its own, or with `branch()` in place of `RunnableBranch` conditions
 * or an output parser that routes on generated text.
 *
 * @example
 * ```ts
 * const router = new NaceRouter({
 *   question: {
 *     type: "choice",
 *     instructions: "Which team should answer?",
 *     criteria: { billing: "Payments", technical: "Bugs" },
 *   },
 *   state: (input: { text: string }) => input.text,
 *   minConfidence: 0.6,
 *   fallback: "human",
 * });
 * const chain = router.branch({ billing, technical, human: escalate }, escalate);
 * await chain.invoke({ text: "I was charged twice" });
 * ```
 */
export class NaceRouter<RunInput = State> extends Runnable<RunInput, RouteDecision> {
  static lc_name(): string {
    return "NaceRouter";
  }

  lc_namespace = ["langchain", "classifiers", "nace"];

  readonly question: Choice;

  readonly classifier: NaceClassifier;

  readonly minConfidence: number;

  readonly fallback: string | undefined;

  protected readonly toState: ((input: RunInput) => State) | undefined;

  constructor(fields: NaceRouterFields<RunInput>) {
    // `lc_kwargs` is enumerable, so `console.log(router)` would print a key in
    // `classifierOptions`; the classifier and the state function stay out too.
    const { classifier: _c, classifierOptions: _o, state: _s, ...kwargs } = fields;
    super(kwargs);
    this.question = fields.question;
    this.classifier = fields.classifier ?? new NaceClassifier(fields.classifierOptions);
    this.minConfidence = fields.minConfidence ?? 0;
    this.fallback = fields.fallback;
    this.toState = fields.state;
  }

  protected decide(answer: ChoiceAnswer): RouteDecision {
    if (answer.confidence >= this.minConfidence) {
      return { route: answer.choice, answer, fellBack: false };
    }
    if (this.fallback === undefined) {
      throw new NaceLowConfidenceError(
        `Drex picked "${answer.choice}" with confidence ${answer.confidence.toFixed(2)}, ` +
          `below minConfidence ${this.minConfidence}, and no fallback is set.`,
      );
    }
    return { route: this.fallback, answer, fellBack: true };
  }

  async invoke(input: RunInput, options?: Partial<RunnableConfig>): Promise<RouteDecision> {
    const state = this.toState ? this.toState(input) : (input as unknown as State);
    const response = await this.classifier.invoke(state, {
      ...options,
      runName: options?.runName ?? "NaceRouter",
      metadata: {
        ...options?.metadata,
        nace_min_confidence: this.minConfidence,
        nace_fallback: this.fallback ?? null,
      },
      questions: { [QUESTION_ID]: this.question },
    });
    const answer = response.choices[QUESTION_ID];
    if (answer === undefined) {
      throw new NaceError(`Drex returned no answer for the routing question.`);
    }
    return this.decide(answer);
  }

  /** Run `routes[decision.route]` (or `defaultRoute`) on the original input. */
  branch<RunOutput>(
    routes: Record<string, RunnableInterface<RunInput, RunOutput>>,
    defaultRoute: RunnableInterface<RunInput, RunOutput>,
  ): Runnable<RunInput, RunOutput> {
    return RunnableLambda.from(async (input: RunInput, config?: RunnableConfig) => {
      const decision = await this.invoke(input, config);
      const target = Object.hasOwn(routes, decision.route) ? routes[decision.route] : undefined;
      return (target ?? defaultRoute).invoke(input, config);
    }).withConfig({ runName: "NaceRouterBranch" }) as Runnable<RunInput, RunOutput>;
  }
}
