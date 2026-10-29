import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { NaceClassifier, type NaceClassifierFields } from "../../src/index.js";

/** Live tests run against prod unless NACE_BASE_URL_LIVE is set (empty means prod). */
export const KEY = process.env.NACE_API_KEY_LIVE?.trim() || undefined;
export const BASE_URL = process.env.NACE_BASE_URL_LIVE?.trim() || undefined;
export const OPENROUTER_KEY = process.env.OPENROUTER_API_KEY_LIVE?.trim() || undefined;
export const ANTHROPIC_KEY = process.env.ANTHROPIC_API_KEY?.trim() || undefined;
export const FIREWORKS_KEY = process.env.FIREWORKS_API_KEY?.trim() || undefined;

export const MODELS = ["drex-v2.0", "drex-max"] as const;

export function fixture(name: string): string {
  return fileURLToPath(new URL(`../fixtures/${name}`, import.meta.url));
}

export const RED = readFileSync(fixture("red.png")).toString("base64");
export const BLUE = readFileSync(fixture("blue.png")).toString("base64");

export const TICKET =
  "I was charged twice for my March invoice and I'm really annoyed. Please refund one of them.";

export const WHICH_RED = {
  red: {
    type: "choice" as const,
    instructions: "Which image is red?",
    criteria: { "Image 1": null, "Image 2": null },
  },
};

export function live(model: string, fields: Omit<NaceClassifierFields, "model"> = {}) {
  return new NaceClassifier({ apiKey: KEY, baseUrl: BASE_URL, model, ...fields });
}

/** Choice and score probabilities sum to 1; usage and a request id are present. */
export function checkResponse(r: Awaited<ReturnType<NaceClassifier["invoke"]>>): void {
  if (!r.requestId) throw new Error("no request id");
  if (!((r.usage.inputTokens ?? 0) > 0)) throw new Error("no input tokens");
  for (const answer of [...Object.values(r.choices), ...Object.values(r.scores)]) {
    const total = Object.values(answer.probabilities).reduce((a, b) => a + b, 0);
    if (Math.abs(total - 1) > 1e-2) throw new Error(`probabilities sum to ${total}`);
  }
}
