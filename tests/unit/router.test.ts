import { RunnableLambda } from "@langchain/core/runnables";
import { describe, expect, it } from "vitest";

import {
  type Choice,
  NaceClassifier,
  NaceLowConfidenceError,
  NaceRouter,
  naceConditionalEdge,
} from "../../src/index.js";
import { fakeDrex, offline } from "./helpers.js";

const TEAM: Choice = {
  type: "choice",
  instructions: "Which team should answer?",
  criteria: { billing: "Payments", technical: "Bugs" },
};

function answering(choice: string, confidence: number) {
  return fakeDrex([
    (body) => ({
      body: {
        model: body.model,
        answers: {
          route: {
            type: "choice",
            choice,
            confidence,
            probabilities: { billing: 0.5, technical: 0.5 },
          },
        },
      },
    }),
  ]);
}

describe("NaceRouter", () => {
  it("routes to Drex's choice and keeps the full answer", async () => {
    const drex = answering("technical", 0.9);
    const router = new NaceRouter({
      question: TEAM,
      classifier: new NaceClassifier(offline(drex.fetch)),
      state: (input: { text: string }) => input.text,
      minConfidence: 0.6,
    });
    const decision = await router.invoke({ text: "The app crashes" });
    expect(decision.route).toBe("technical");
    expect(decision.fellBack).toBe(false);
    expect(decision.answer.probabilities).toEqual({ billing: 0.5, technical: 0.5 });
    expect(drex.requests[0]!.body.state).toBe("The app crashes");
    expect(drex.requests[0]!.body.questions.route.criteria).toEqual(TEAM.criteria);
  });

  it("falls back below minConfidence, or throws without a fallback", async () => {
    const fallback = new NaceRouter({
      question: TEAM,
      classifierOptions: offline(answering("billing", 0.3).fetch),
      minConfidence: 0.6,
      fallback: "human",
    });
    expect(await fallback.invoke("hmm")).toMatchObject({ route: "human", fellBack: true });
    const strict = new NaceRouter({
      question: TEAM,
      classifierOptions: offline(answering("billing", 0.3).fetch),
      minConfidence: 0.6,
    });
    await expect(strict.invoke("hmm")).rejects.toThrow(NaceLowConfidenceError);
  });

  it("branch() runs the chosen route on the original input", async () => {
    const router = new NaceRouter({
      question: TEAM,
      classifierOptions: offline(answering("billing", 0.95).fetch),
      state: (input: { text: string }) => input.text,
    });
    const chain = router.branch(
      {
        billing: RunnableLambda.from((x: { text: string }) => `billing:${x.text}`),
        technical: RunnableLambda.from((x: { text: string }) => `tech:${x.text}`),
      },
      RunnableLambda.from(() => "default"),
    );
    expect(await chain.invoke({ text: "charged twice" })).toBe("billing:charged twice");
  });
});

describe("naceConditionalEdge", () => {
  it("returns the label, or human_review below the threshold", async () => {
    const confident = naceConditionalEdge({
      question: TEAM,
      state: (s: { text: string }) => s.text,
      minConfidence: 0.7,
      classifierOptions: offline(answering("technical", 0.9).fetch),
    });
    expect(await confident({ text: "crash" })).toBe("technical");
    const unsure = naceConditionalEdge({
      question: TEAM,
      state: (s: { text: string }) => s.text,
      minConfidence: 0.7,
      classifierOptions: offline(answering("technical", 0.4).fetch),
    });
    expect(await unsure({ text: "help" })).toBe("human_review");
  });
});
