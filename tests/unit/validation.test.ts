/** Responses must answer the questions that were asked; errors must stay small. */
import { describe, expect, it } from "vitest";

import { NaceAPIResponseValidationError, NaceClassifier } from "../../src/index.js";
import { fakeDrex, offline, okBody } from "./helpers.js";

const choice = (criteria: Record<string, string | null>) => ({
  type: "choice" as const,
  instructions: "?",
  criteria,
});
const picked = (label: string) => ({
  type: "choice",
  choice: label,
  confidence: 1,
  probabilities: { [label]: 1 },
});

function classifier(questions: Record<string, unknown>, answers: Record<string, unknown>) {
  const server = fakeDrex([(body: any) => ({ body: { ...okBody(body), answers } })]);
  return new NaceClassifier(offline(server.fetch, { questions }));
}

describe("answers match the questions asked", () => {
  it.each([
    ["a missing answer", {}, /no answer for question "q"/],
    ["the wrong type", { q: picked("a") }, /"choice" answer for question "q".*"noul"/],
  ])("refuses %s", async (_name, answers, message) => {
    const c = classifier({ q: { type: "noul", instructions: "?" } }, answers);
    const error = await c.invoke("x").catch((e) => e);
    expect(error).toBeInstanceOf(NaceAPIResponseValidationError);
    expect(error.message).toMatch(message);
  });

  it("refuses a choice that was not offered", async () => {
    const c = classifier({ q: choice({ a: null, b: null }) }, { q: picked("__end__") });
    const error = await c.invoke("x").catch((e) => e);
    expect(error).toBeInstanceOf(NaceAPIResponseValidationError);
    expect(error.message).toMatch(/"q"/);
    expect(error.message).not.toContain("__end__");
  });

  it("refuses a score legend longer than the rubric", async () => {
    const legend = Array.from({ length: 100 }, (_, i) => `level ${i}`);
    const c = classifier(
      { q: { type: "score", instructions: "?", criteria: ["a", "b", "c", "d", "e"] } },
      { q: { type: "score", score: 4, legend, confidence: 1, probabilities: { "4": 1 } } },
    );
    const error = await c.invoke("x").catch((e) => e);
    expect(error).toBeInstanceOf(NaceAPIResponseValidationError);
    expect(error.message).toMatch(/"q"/);
  });

  it("accepts extra answer ids, offered choices and a matching legend", async () => {
    const server = fakeDrex([
      (body: any) => ({
        body: okBody(body, {
          q: picked("b"),
          s: {
            type: "score",
            score: 1,
            legend: ["a", "b"],
            confidence: 1,
            probabilities: { "1": 1 },
          },
          n: { type: "noul", noul: 0.5 },
          extra: { type: "noul", noul: 0.1 },
        }),
      }),
    ]);
    const c = new NaceClassifier(
      offline(server.fetch, {
        questions: {
          q: choice({ a: null, b: null }),
          s: { type: "score", instructions: "?", criteria: ["lo", "hi"] },
          n: { type: "noul", instructions: "?" },
        },
      }),
    );
    const response = await c.invoke("x");
    expect(response.choices.q?.choice).toBe("b");
    expect(Object.keys(response.answers)).toContain("extra");
  });

  it("passes unknown question types through unchecked", async () => {
    const server = fakeDrex([
      (body: any) => ({
        body: okBody(body, { w: { type: "ranking", order: ["x", "y"] } }),
      }),
    ]);
    const c = new NaceClassifier(
      offline(server.fetch, { questions: { w: { type: "ranking", instructions: "?" } } as never }),
    );
    const response = await c.invoke("x");
    expect(response.answers.w).toMatchObject({ type: "ranking", order: ["x", "y"] });
  });
});
