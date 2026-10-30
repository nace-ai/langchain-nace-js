/** Responses must answer the questions that were asked; errors must stay small. */
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { createAgent } from "langchain";
import { describe, expect, it } from "vitest";
import * as z from "zod/v4";

import {
  type ClassificationResponse,
  NaceAPIResponseValidationError,
  NaceClassifier,
  NaceError,
} from "../../src/index.js";
import { naceAutoModeMiddleware, verdictFromResponse } from "../../src/middleware/index.js";
import { parseResponse } from "../../src/types.js";
import { fakeDrex, offline, okBody, ScriptedChatModel } from "./helpers.js";

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

describe("gate rubric", () => {
  const wideRisk = (levels: number, score: number) => ({
    type: "score",
    score,
    legend: Array.from({ length: levels }, (_, i) => `level ${i}`),
    confidence: 1,
    probabilities: { [String(score)]: 1 },
  });
  const gateResponse = (risk: unknown): ClassificationResponse =>
    parseResponse({
      model: "drex-max",
      answers: {
        risk,
        destructive: { type: "noul", noul: 0 },
        matches_request: { type: "noul", noul: 1 },
      },
    });

  it("reads the risk against the rubric that was asked, not the legend that came back", () => {
    expect(() => verdictFromResponse(gateResponse(wideRisk(100, 4)), 5)).toThrow(NaceError);
    expect(() => verdictFromResponse(gateResponse(wideRisk(100, 4)), 5)).toThrow(/rubric/);
    expect(verdictFromResponse(gateResponse(wideRisk(5, 4)), 5).risk).toBe(1);
    // Without a rubric the legend is all there is, as before.
    expect(verdictFromResponse(gateResponse(wideRisk(100, 4))).risk).toBeCloseTo(4 / 99);
  });

  /** A classifier that skips response checking, so only the gate's own check is tested. */
  class Unchecked extends NaceClassifier {
    override async invoke(): Promise<ClassificationResponse> {
      return gateResponse(wideRisk(100, 4));
    }
  }

  async function run(classifier: NaceClassifier) {
    const ran: string[] = [];
    const wipe = tool(
      async () => {
        ran.push("wipe");
        return "gone";
      },
      { name: "wipe_disk", description: "Wipe the disk.", schema: z.object({}) },
    );
    const agent = createAgent({
      model: new ScriptedChatModel([
        new AIMessage({ content: "", tool_calls: [{ id: "c1", name: "wipe_disk", args: {} }] }),
        new AIMessage("done"),
      ]),
      tools: [wipe],
      middleware: [naceAutoModeMiddleware({ classifier })],
    });
    const result = await agent.invoke({ messages: [new HumanMessage("Wipe it")] });
    return { ran, last: String(result.messages.at(-2)?.content) };
  }

  it("blocks when the legend does not match the gate's rubric", async () => {
    const { ran, last } = await run(new Unchecked(offline(fakeDrex().fetch)));
    expect(ran).toEqual([]);
    expect(last).toMatch(/its risk could not be checked/);
  });
});
