/** Live Drex Router checks. Synthetic prompts only. */
import { describe, expect, it } from "vitest";

import { NaceModelSelector, type SelectorCandidate } from "../../src/index.js";
import { BASE_URL, KEY } from "./live.js";

const FAST = "openai/gpt-4o-mini";
const STRONG = "anthropic/claude-haiku-5.5";
const CUSTOM_ID = "my-org/fine-tuned-support-model";
const TURNS = [
  { role: "user" as const, content: "I was charged twice for March. Please refund one charge." },
];

function selector(candidates: SelectorCandidate[]) {
  return new NaceModelSelector({
    apiKey: KEY,
    baseUrl: BASE_URL,
    timeout: 60_000,
    candidates,
  });
}

function check(selection: Awaited<ReturnType<NaceModelSelector["select"]>>, ids: string[]) {
  expect(selection.reason).toBe("selected");
  expect(ids).toContain(selection.selected);
  expect(Object.keys(selection.probabilities).sort()).toEqual([...ids].sort());
  const sum = Object.values(selection.probabilities).reduce((total, value) => total + value, 0);
  expect(Math.abs(sum - 1)).toBeLessThan(0.02);
}

describe.skipIf(!KEY)("Drex Router model selection", () => {
  it("selects between two known ids", async () => {
    const selection = await selector([FAST, STRONG]).select(TURNS);
    check(selection, [FAST, STRONG]);
  });

  it("selects with one custom profile", async () => {
    const selection = await selector([
      {
        id: CUSTOM_ID,
        provider: "my-org",
        priceIn: 0.4,
        priceOut: 1.6,
        context: 128_000,
        tags: ["support", "chat"],
      },
      FAST,
    ]).select(TURNS);
    check(selection, [CUSTOM_ID, FAST]);
    expect(selection.probabilities).toHaveProperty(CUSTOM_ID);
  });
});
