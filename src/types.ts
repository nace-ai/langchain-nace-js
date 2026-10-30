/**
 * Question and answer types for Drex.
 *
 * Names and shapes follow `@langchain/typesafe`, so code written for
 * `TypeSafeClassifier` runs on `NaceClassifier` after changing the import.
 */
import * as z from "zod/v4";

import { clip, NaceAPIResponseValidationError, NaceError } from "./errors.js";

/** Any value expressible in JSON. */
export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

/** Content for a question's `instructions` or a criterion: text or JSON. */
export type QuestionContent = string | { [key: string]: JsonValue } | JsonValue[];

/** Descriptions of a Noul's two outcomes. The keys are literally `true` and `false`. */
export interface NoulCriteria {
  true?: JsonValue;
  false?: JsonValue;
}

/** A yes/no question; the answer is the probability of yes. */
export interface Noul {
  type: "noul";
  /** The question. Optional when `criteria` describes an outcome. */
  instructions?: QuestionContent;
  criteria?: NoulCriteria;
}

/** Pick one label. `criteria` maps each label to a description (or `null`). */
export interface Choice {
  type: "choice";
  criteria: Record<string, JsonValue>;
  instructions?: QuestionContent;
}

/** Rate on an ordered rubric. `criteria` is lowest first; level `i` scores `i`. */
export interface Score {
  type: "score";
  criteria: JsonValue[];
  instructions?: QuestionContent;
}

/**
 * A question type this package doesn't model yet. Sent as given (with
 * `instructions` defaulted to `""`), so new Drex question types work without a
 * package release.
 */
export interface RawQuestion {
  type: string;
  instructions?: QuestionContent | null;
  [key: string]: unknown;
}

export type Question = Noul | Choice | Score;

/** Questions keyed by the id their answers come back under. */
export type Questions = Record<string, Question | RawQuestion>;

/** P(yes) for a Noul. */
export interface NoulAnswer {
  type: "noul";
  noul: number;
}

/** The selected label with every label's probability. */
export interface ChoiceAnswer {
  type: "choice";
  choice: string;
  probabilities: Record<string, number>;
  /** How sure the model is; not the same number as the top probability. */
  confidence: number;
}

/**
 * The expected level with its rubric and level probabilities. `legend` and
 * `probabilities` are keyed by level index (`"0"`, `"1"`, ...), so
 * `answer.legend[0]` works.
 */
export interface ScoreAnswer {
  type: "score";
  /** Expected level; may fall between two levels. */
  score: number;
  legend: Record<string, JsonValue>;
  probabilities: Record<string, number>;
  confidence: number;
}

/** An answer of a type this package version doesn't model yet. */
export interface RawAnswer {
  type: string;
  [key: string]: unknown;
}

export type Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer;

/** Token usage reported by Drex. */
export interface Usage {
  /** Billed input tokens. */
  inputTokens?: number;
  /** Output tokens (not billed). */
  outputTokens?: number;
  /** Image tokens, already included in `inputTokens`. */
  imageTokens?: number;
}

/** Answers and metadata from one Drex request. */
export interface ClassificationResponse {
  /** The model that answered. */
  model: string;
  /** Every answer, keyed by question id. */
  answers: Record<string, Answer | RawAnswer>;
  usage: Usage;
  /** Drex's request id; quote it in support requests. */
  requestId?: string;
  evaluationTimeMs?: number;
  /** The response body as sent, including fields this version doesn't model. Non-enumerable. */
  readonly raw: Record<string, unknown>;
  /** Noul answers only, keyed by question id. Non-enumerable. */
  readonly nouls: Record<string, NoulAnswer>;
  /** Choice answers only, keyed by question id. Non-enumerable. */
  readonly choices: Record<string, ChoiceAnswer>;
  /** Score answers only, keyed by question id. Non-enumerable. */
  readonly scores: Record<string, ScoreAnswer>;
}

/** Alias kept for parity with the Python package. */
export type ClassifierResponse = ClassificationResponse;

const jsonValue = z.json();
const content = z.union([z.string(), z.record(z.string(), jsonValue), z.array(jsonValue)]);

const noulSchema = z
  .object({
    type: z.literal("noul"),
    instructions: content.nullish(),
    // Strict, so `{ True: ... }` is refused rather than silently not sent. The
    // message names no key: keys are the caller's.
    criteria: z
      .strictObject(
        { true: jsonValue.optional(), false: jsonValue.optional() },
        { error: "Noul criteria may only have the keys `true` and `false`." },
      )
      .nullish(),
  })
  .refine(
    (q) =>
      (q.instructions !== undefined && q.instructions !== null) ||
      (q.criteria !== undefined && q.criteria !== null),
    { message: "Noul question must have criteria or instructions." },
  );

const choiceSchema = z.object({
  type: z.literal("choice"),
  criteria: z
    .custom<Record<string, JsonValue>>(
      (c) => typeof c === "object" && c !== null && !Array.isArray(c) && Object.keys(c).length >= 1,
      { message: "Choice question must have at least 1 option." },
    )
    .pipe(z.record(z.string(), jsonValue)),
  instructions: content.nullish(),
});

const scoreSchema = z.object({
  type: z.literal("score"),
  criteria: z.array(jsonValue).min(1, "Score question must have at least 1 level."),
  instructions: content.nullish(),
});

const KNOWN: Record<string, z.ZodType> = {
  noul: noulSchema,
  choice: choiceSchema,
  score: scoreSchema,
};

/**
 * Check a questions map, naming the question id in any error. Error messages
 * describe the expected shape and never echo the caller's values.
 */
export function validateQuestions(questions: unknown): Questions {
  if (typeof questions !== "object" || questions === null || Array.isArray(questions)) {
    throw new NaceError("questions must be an object mapping ids to questions.");
  }
  const ids = Object.keys(questions);
  if (ids.length === 0) {
    throw new NaceError("At least one question is required.");
  }
  for (const id of ids) {
    const question = (questions as Record<string, unknown>)[id];
    if (typeof question !== "object" || question === null || Array.isArray(question)) {
      throw new NaceError(`Invalid question "${id}": must be an object with a \`type\`.`);
    }
    const type = (question as { type?: unknown }).type;
    if (typeof type !== "string" || type === "") {
      throw new NaceError(`Invalid question "${id}": \`type\` must be a non-empty string.`);
    }
    const schema = Object.hasOwn(KNOWN, type) ? KNOWN[type] : undefined;
    if (schema === undefined) continue;
    let result: ReturnType<typeof schema.safeParse>;
    try {
      result = schema.safeParse(question);
    } catch {
      throw new NaceError(
        `Invalid question "${id}": criteria or instructions are too deeply nested, or circular.`,
      );
    }
    if (!result.success) {
      const issue = result.error.issues[0];
      const path = issue?.path.join(".") ?? "";
      throw new NaceError(
        `Invalid question "${id}": ${path ? `${path}: ` : ""}${issue?.message ?? "invalid"}`,
      );
    }
  }
  return questions as Questions;
}

/** Drex reads text: structured content is sent as compact JSON text. */
function asText(value: unknown): unknown {
  if (value === undefined || value === null || typeof value === "string") return value;
  return JSON.stringify(value);
}

function instructionsText(value: unknown): string {
  const text = asText(value);
  return typeof text === "string" ? text : "";
}

/**
 * A question as Drex reads it.
 *
 * Field order is `type, instructions, criteria`. `instructions` is always a
 * string (`""` when omitted), because Drex rejects `null` and OpenRouter
 * requires the key. Structured instructions and criteria become JSON text.
 * `null` choice descriptions are kept. Unknown question types are sent as given.
 */
export function toWire(question: Question | RawQuestion): Record<string, unknown> {
  const wire: Record<string, unknown> = Object.create(null);
  wire.type = question.type;
  wire.instructions = instructionsText(question.instructions);
  if (question.type === "noul") {
    const criteria = (question as Noul).criteria;
    if (criteria !== undefined && criteria !== null) {
      const out: Record<string, unknown> = {};
      if (criteria.true !== undefined && criteria.true !== null) out.true = asText(criteria.true);
      if (criteria.false !== undefined && criteria.false !== null)
        out.false = asText(criteria.false);
      if (Object.keys(out).length > 0) wire.criteria = out;
    }
    return { ...wire };
  }
  if (question.type === "choice") {
    const out: Record<string, unknown> = Object.create(null);
    for (const [label, description] of Object.entries((question as Choice).criteria)) {
      out[label] = asText(description);
    }
    wire.criteria = out;
    return { ...wire };
  }
  if (question.type === "score") {
    wire.criteria = (question as Score).criteria.map((level) => asText(level));
    return { ...wire };
  }
  for (const [key, value] of Object.entries(question)) {
    if (key !== "type" && key !== "instructions") wire[key] = value;
  }
  return { ...wire };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Set an own property, so a key such as `__proto__` is kept rather than lost. */
function setOwn(target: Record<string, unknown>, key: string, value: unknown): void {
  Object.defineProperty(target, key, {
    value,
    enumerable: true,
    writable: true,
    configurable: true,
  });
}

function finite(value: unknown, field: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new NaceAPIResponseValidationError(
      `Drex returned an answer with an invalid \`${field}\`.`,
    );
  }
  return value;
}

function bounded(value: unknown, field: string, max = 1): number {
  const number = finite(value, field);
  if (number < 0 || number > max) {
    throw new NaceAPIResponseValidationError(
      `Drex returned an answer with an out-of-range \`${field}\`.`,
    );
  }
  return number;
}

function numbers(value: unknown, field: string): Record<string, number> {
  const out: Record<string, number> = {};
  if (Array.isArray(value)) {
    value.forEach((p, i) => {
      setOwn(out, String(i), bounded(p, field));
    });
    return out;
  }
  if (!isRecord(value)) {
    throw new NaceAPIResponseValidationError(`Drex returned an answer without \`${field}\`.`);
  }
  for (const [key, p] of Object.entries(value)) setOwn(out, key, bounded(p, field));
  return out;
}

function parseAnswer(raw: unknown): Answer | RawAnswer {
  if (!isRecord(raw) || typeof raw.type !== "string") {
    throw new NaceAPIResponseValidationError("Drex returned an answer without a `type`.");
  }
  if (raw.type === "noul") {
    return { type: "noul", noul: bounded(raw.noul, "noul") };
  }
  if (raw.type === "choice") {
    if (typeof raw.choice !== "string") {
      throw new NaceAPIResponseValidationError("Drex returned a choice answer without `choice`.");
    }
    return {
      type: "choice",
      choice: raw.choice,
      probabilities: numbers(raw.probabilities, "probabilities"),
      confidence: bounded(raw.confidence, "confidence"),
    };
  }
  if (raw.type === "score") {
    const legend: Record<string, JsonValue> = {};
    if (Array.isArray(raw.legend)) {
      raw.legend.forEach((level, i) => {
        setOwn(legend, String(i), level);
      });
    } else if (isRecord(raw.legend)) {
      for (const [key, level] of Object.entries(raw.legend)) setOwn(legend, key, level);
    }
    const levels = Object.keys(legend);
    if (levels.length === 0 || levels.some((_, i) => !Object.hasOwn(legend, String(i)))) {
      throw new NaceAPIResponseValidationError("Drex returned an invalid score legend.");
    }
    return {
      type: "score",
      score: bounded(raw.score, "score", levels.length - 1),
      legend,
      probabilities: numbers(raw.probabilities, "probabilities"),
      confidence: bounded(raw.confidence, "confidence"),
    };
  }
  return { ...raw, type: raw.type };
}

function hide(target: object, key: string, get: () => unknown): void {
  Object.defineProperty(target, key, { enumerable: false, configurable: true, get });
}

function pick<T extends Answer>(answers: Record<string, Answer | RawAnswer>, type: T["type"]) {
  const out: Record<string, T> = {};
  for (const [id, answer] of Object.entries(answers)) {
    if (answer.type === type) setOwn(out, id, answer);
  }
  return out;
}

/** A question id for a message: quoted, escaped, and cut to 64 characters. */
function quoted(id: string): string {
  return JSON.stringify(clip(id, 65));
}

/**
 * Check that the answers fit the questions that were asked. Extra answer ids and
 * answers to question types this package doesn't know are left alone.
 *
 * Messages name the question and the question's own type, never text the response
 * supplied (an answer's type, a choice, a legend), so a bad response can't push its
 * content into logs.
 */
function checkAnswered(
  answers: Record<string, Answer | RawAnswer>,
  asked: Record<string, Record<string, unknown>>,
): void {
  for (const [id, question] of Object.entries(asked)) {
    const name = quoted(id);
    if (!Object.hasOwn(answers, id)) {
      throw new NaceAPIResponseValidationError(`Drex returned no answer for question ${name}.`);
    }
    const answer = answers[id] as Answer | RawAnswer;
    const type = question.type;
    if (type !== "noul" && type !== "choice" && type !== "score") continue;
    if (answer.type !== type) {
      throw new NaceAPIResponseValidationError(
        `Drex's answer to question ${name} has the wrong type (expected ${type}).`,
      );
    }
    const criteria = question.criteria;
    if (type === "choice" && isRecord(criteria)) {
      if (!Object.hasOwn(criteria, (answer as ChoiceAnswer).choice)) {
        throw new NaceAPIResponseValidationError(
          `Drex answered question ${name} with a choice that was not offered.`,
        );
      }
    }
    if (type === "score" && Array.isArray(criteria)) {
      const levels = Object.keys((answer as ScoreAnswer).legend).length;
      if (levels !== criteria.length) {
        throw new NaceAPIResponseValidationError(
          `Drex answered question ${name} with a score legend of ${levels} levels, ` +
            `but the rubric has ${criteria.length}.`,
        );
      }
    }
  }
}

/**
 * Build a response from a Drex response body, keeping unknown answer types.
 *
 * With `asked` (the questions as sent), every asked question must have an answer
 * of its own type, a choice must be one of the labels offered, and a score's
 * legend must be as long as the rubric asked.
 */
export function parseResponse(
  body: unknown,
  requestId?: string,
  asked?: Record<string, Record<string, unknown>>,
): ClassificationResponse {
  if (!isRecord(body)) {
    throw new NaceAPIResponseValidationError("Drex returned a body that isn't a JSON object.");
  }
  const answers: Record<string, Answer | RawAnswer> = {};
  const rawAnswers = isRecord(body.answers) ? body.answers : {};
  for (const [id, raw] of Object.entries(rawAnswers)) {
    setOwn(answers, id, parseAnswer(raw));
  }
  if (asked !== undefined) checkAnswered(answers, asked);
  const rawUsage = isRecord(body.usage) ? body.usage : {};
  const usage: Usage = {};
  if (typeof rawUsage.input_tokens === "number") usage.inputTokens = rawUsage.input_tokens;
  if (typeof rawUsage.output_tokens === "number") usage.outputTokens = rawUsage.output_tokens;
  if (typeof rawUsage.image_tokens === "number") usage.imageTokens = rawUsage.image_tokens;
  const response: Record<string, unknown> = {
    model: typeof body.model === "string" ? body.model : "",
    answers,
    usage,
  };
  const id = requestId ?? (typeof body.request_id === "string" ? body.request_id : undefined);
  if (id !== undefined) response.requestId = id;
  if (typeof body.evaluation_time_ms === "number") {
    response.evaluationTimeMs = body.evaluation_time_ms;
  }
  hide(response, "raw", () => body);
  hide(response, "nouls", () => pick<NoulAnswer>(answers, "noul"));
  hide(response, "choices", () => pick<ChoiceAnswer>(answers, "choice"));
  hide(response, "scores", () => pick<ScoreAnswer>(answers, "score"));
  return response as unknown as ClassificationResponse;
}
