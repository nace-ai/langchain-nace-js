/** Adversarial probes for the public release review; all credentials are fake. */
import { inspect } from "node:util";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { MemorySaver } from "@langchain/langgraph";
import { createAgent } from "langchain";
import { expect, it } from "vitest";
import * as z from "zod/v4";
import { NaceAPIResponseValidationError, NaceClassifier } from "../../src/index.js";
import { naceAutoModeMiddleware } from "../../src/middleware/index.js";
import { fakeDrex, offline, okBody, ScriptedChatModel } from "./helpers.js";

it.each(["X-Vercel-Protection-Bypass", "X-Proxy-Key"])(
  "keeps %s private while sending configured headers",
  async (header) => {
    const server = fakeDrex();
    const classifier = new NaceClassifier({
      apiKey: "fake-key",
      fetch: server.fetch,
      questions: { q: { type: "noul", instructions: "?" } },
      defaultHeaders: {
        [header]: "FAKE_PRIVATE_HEADER_REVIEW",
        "X-App-Version": "test-version",
      },
    });
    expect.soft(inspect(classifier, { depth: 8 })).not.toContain("FAKE_PRIVATE_HEADER_REVIEW");
    expect.soft(JSON.stringify(classifier)).not.toContain("FAKE_PRIVATE_HEADER_REVIEW");
    expect(classifier.toJSON()).not.toHaveProperty("kwargs.default_headers");
    await classifier.invoke("x");
    expect(server.requests[0]!.headers.get(header)).toBe("FAKE_PRIVATE_HEADER_REVIEW");
    expect(server.requests[0]!.headers.get("X-App-Version")).toBe("test-version");
  },
);

function answer(risk: number, destructive: number, matches: number) {
  return (body: any) => ({
    body: okBody(body, {
      risk: {
        type: "score",
        score: risk,
        legend: ["a", "b", "c", "d", "e"],
        confidence: 0.9,
        probabilities: { "0": 1 },
      },
      destructive: { type: "noul", noul: destructive },
      matches_request: { type: "noul", noul: matches },
    }),
  });
}

it("checks a reused call again when a new user turn changes permission", async () => {
  const drex = fakeDrex([answer(0, 0, 1), answer(4, 1, 0)]);
  let ran = 0;
  const send = tool(
    async () => {
      ran++;
      return "sent";
    },
    {
      name: "send_report",
      description: "Send report",
      schema: z.object({}),
    },
  );
  const call = () =>
    new AIMessage({ content: "", tool_calls: [{ id: "reused", name: "send_report", args: {} }] });
  const agent = createAgent({
    model: new ScriptedChatModel([call(), new AIMessage("done"), call(), new AIMessage("done")]),
    tools: [send],
    checkpointer: new MemorySaver(),
    middleware: [naceAutoModeMiddleware({ classifierOptions: offline(drex.fetch) })],
  });
  const config = { configurable: { thread_id: "same-conversation" } };
  await agent.invoke({ messages: [new HumanMessage("Send the report")] }, config);
  await agent.invoke({ messages: [new HumanMessage("Do not send the report again")] }, config);
  expect.soft(ran).toBe(1);
  expect.soft(drex.requests).toHaveLength(2);
});

it("blocks a tool call on out-of-range decision probabilities", async () => {
  const drex = fakeDrex([answer(-1, -1, 2)]);
  let ran = false;
  const send = tool(
    async () => {
      ran = true;
      return "sent";
    },
    {
      name: "send_report",
      description: "Send report",
      schema: z.object({}),
    },
  );
  const agent = createAgent({
    model: new ScriptedChatModel([
      new AIMessage({ content: "", tool_calls: [{ id: "bad", name: "send_report", args: {} }] }),
      new AIMessage("done"),
    ]),
    tools: [send],
    middleware: [naceAutoModeMiddleware({ classifierOptions: offline(drex.fetch) })],
  });
  await agent.invoke({ messages: [new HumanMessage("Read the report")] });
  expect(ran).toBe(false);
});

it("preserves special answer IDs in typed response accessors", async () => {
  const drex = fakeDrex([
    (body) => ({ body: okBody(body, JSON.parse('{"__proto__":{"type":"noul","noul":0.4}}')) }),
  ]);
  const classifier = new NaceClassifier(
    offline(drex.fetch, {
      questions: JSON.parse('{"__proto__":{"type":"noul","instructions":"?"}}'),
    }),
  );
  const response = await classifier.invoke("x");
  expect(Object.keys(response.answers)).toEqual(["__proto__"]);
  expect(Object.keys(response.nouls)).toEqual(["__proto__"]);
});

const choice = { type: "choice", choice: "a", confidence: 1, probabilities: { a: 1 } };
const score = {
  type: "score",
  score: 1,
  legend: ["low", "high"],
  confidence: 1,
  probabilities: { "1": 1 },
};

it.each([
  { type: "noul", noul: -0.01 },
  { type: "noul", noul: 1.01 },
  { ...choice, confidence: -0.01 },
  { ...choice, confidence: 1.01 },
  { ...choice, probabilities: { a: -0.01 } },
  { ...choice, probabilities: { a: 1.01 } },
  { ...score, score: -1 },
  { ...score, score: 2 },
  { ...score, legend: [] },
  { ...score, legend: { "1": "high" } },
])("rejects invalid answer ranges or rubrics: %j", async (raw) => {
  const server = fakeDrex([(body) => ({ body: okBody(body, { q: raw }) })]);
  const classifier = new NaceClassifier(
    offline(server.fetch, {
      questions: { q: { type: "noul", instructions: "?" } },
    }),
  );
  await expect(classifier.invoke("x")).rejects.toBeInstanceOf(NaceAPIResponseValidationError);
});

it("accepts valid boundary values and preserves all typed special answer IDs", async () => {
  const raw = JSON.parse(
    JSON.stringify({
      noul: { type: "noul", noul: 0 },
      choice,
      score: {
        ...score,
        score: 0,
        legend: ["only level"],
        confidence: 0,
        probabilities: { "0": 1 },
      },
    }),
  );
  for (const [kind, value] of Object.entries(raw)) {
    const answers = Object.fromEntries([
      ["__proto__", value],
      ["constructor", value],
    ]);
    const server = fakeDrex([(body) => ({ body: okBody(body, answers) })]);
    const response = await new NaceClassifier(
      offline(server.fetch, {
        questions: { q: { type: "noul", instructions: "?" } },
      }),
    ).invoke("x");
    const typed =
      kind === "noul" ? response.nouls : kind === "choice" ? response.choices : response.scores;
    expect(Object.keys(typed)).toEqual(["__proto__", "constructor"]);
    expect(Object.getPrototypeOf(typed)).toBe(Object.prototype);
  }
});
