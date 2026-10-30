/** Responses must answer the questions that were asked; errors must stay small. */
import { inspect } from "node:util";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { createAgent } from "langchain";
import { describe, expect, it } from "vitest";
import * as z from "zod/v4";
import {
  BILLING_URL,
  type ClassificationResponse,
  NaceAPIResponseValidationError,
  NaceBadRequestError,
  NaceClassifier,
  NaceError,
  NaceInsufficientCreditError,
  NaceUnprocessableEntityError,
  naceConditionalEdge,
  OPENROUTER_CREDITS_URL,
} from "../../src/index.js";
import { naceAutoModeMiddleware, verdictFromResponse } from "../../src/middleware/index.js";
import { parseResponse } from "../../src/types.js";
import { fakeDrex, offline, okBody, PNG_1x1, ScriptedChatModel } from "./helpers.js";

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
    [
      "the wrong type",
      { q: picked("a") },
      /^Drex's answer to question "q" has the wrong type \(expected noul\)\.$/,
    ],
  ])("refuses %s", async (_name, answers, message) => {
    const c = classifier({ q: { type: "noul", instructions: "?" } }, answers);
    const error = await c.invoke("x").catch((e) => e);
    expect(error).toBeInstanceOf(NaceAPIResponseValidationError);
    expect(error.message).toMatch(message);
  });

  it("keeps a response's own text out of the message", async () => {
    const huge = "x".repeat(200_000);
    const c = classifier({ q: { type: "noul", instructions: "?" } }, { q: { type: huge } });
    const error = await c.invoke("x").catch((e) => e);
    expect(error).toBeInstanceOf(NaceAPIResponseValidationError);
    expect(error.message.length).toBeLessThan(300);
    expect(error.message).not.toContain("xxxxxxxxxx");
    expect(error.message).toBe(`Drex's answer to question "q" has the wrong type (expected noul).`);
  });

  it("bounds a question id and quotes it", async () => {
    const long = "q".repeat(500);
    const noul = { type: "noul", instructions: "?" };
    const tooLong = await classifier({ [long]: noul }, {})
      .invoke("x")
      .catch((e) => e);
    expect(tooLong.message).toBe(
      `Drex returned no answer for question ${JSON.stringify(`${"q".repeat(64)}…`)}.`,
    );
    const odd = await classifier({ 'a"\nb': noul }, {})
      .invoke("x")
      .catch((e) => e);
    expect(odd.message).toBe('Drex returned no answer for question "a\\"\\nb".');
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
    expect(error.message).not.toContain("level 5");
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

describe("conditional edge", () => {
  const question = choice({ billing: null, technical: null });
  const edge = (label: string) => {
    /** A classifier that skips response checking, so only the edge's own check is tested. */
    class Unchecked extends NaceClassifier {
      override async invoke(): Promise<ClassificationResponse> {
        return parseResponse({ model: "drex-max", answers: { route: picked(label) } });
      }
    }
    return naceConditionalEdge({
      question,
      state: (s: string) => s,
      minConfidence: 0.5,
      classifier: new Unchecked(offline(fakeDrex().fetch)),
    });
  };

  it("returns an offered label", async () => {
    await expect(edge("billing")("hi")).resolves.toBe("billing");
  });

  it("refuses a label that was not offered, such as the graph's end marker", async () => {
    const error = await edge("__end__")("hi").catch((e) => e);
    expect(error).toBeInstanceOf(NaceAPIResponseValidationError);
    expect(error.message).not.toContain("__end__");
  });
});

describe("error size", () => {
  const MAX = 500;
  const GUIDANCE = 350;

  /** A fetch that answers every request with `status` and a raw body. */
  const answering =
    (status: number, text: string, headers: Record<string, string> = {}) =>
    async () =>
      new Response(text, { status, headers: { "x-request-id": "req_big", ...headers } });
  const fail = (fetch: typeof globalThis.fetch, extra: Record<string, unknown> = {}) =>
    new NaceClassifier(
      offline(fetch, {
        questions: { q: { type: "noul", instructions: "?" } },
        maxRetries: 0,
        ...extra,
      }),
    )
      .invoke("x")
      .catch((e) => e);

  it("summarises an HTML error page and keeps the page on the error", async () => {
    const html = `<!doctype html><html><body>${"<p>Bad gateway</p>".repeat(6_500)}</body></html>`;
    expect(html.length).toBeGreaterThan(114_000);
    const error = await fail(answering(502, html, { "content-type": "text/html" }));
    expect(error.status).toBe(502);
    expect(error.message).toContain("Drex returned an HTML error page (HTTP 502)");
    expect(error.message.length).toBeLessThanOrEqual(MAX + 1 + GUIDANCE);
    expect(error.message).toContain("(request_id=req_big)");
    expect(String(error.body).length).toBe(html.length);
  });

  it("keeps an oversized upstream error out of what inspect() prints", async () => {
    const html = `<!doctype html><html><body>SENSITIVE-MARKER ${"<p>Bad gateway</p>".repeat(7_600)}SENSITIVE-MARKER</body></html>`;
    expect(html.length).toBeGreaterThan(136_000);
    const error = await fail(answering(502, html, { "content-type": "text/html" }));
    const shown = inspect(error, { depth: 5 });
    expect(shown.length).toBeLessThan(5_000);
    expect(shown).not.toContain("SENSITIVE-MARKER");
    expect(String(error.body)).toContain("SENSITIVE-MARKER");
    expect(error.status).toBe(502);
    expect(error.requestId).toBe("req_big");
  });

  it("keeps a large JSON body out of what inspect() prints", async () => {
    const body = JSON.stringify({ error: { message: "x".repeat(5_000) + "SENSITIVE-MARKER" } });
    const error = await fail(answering(500, body, { "content-type": "application/json" }));
    const shown = inspect(error, { depth: 5 });
    expect(shown.length).toBeLessThan(5_000);
    expect(shown).not.toContain("SENSITIVE-MARKER");
    expect(error.body.error.message).toContain("SENSITIVE-MARKER");
  });

  it("names OpenRouter for its HTML error page", async () => {
    const error = await fail(answering(502, "<html>nope</html>", { "content-type": "text/html" }), {
      baseUrl: "https://openrouter.ai/api/v1",
      apiKey: "sk-or-test",
    });
    expect(error.message).toContain("OpenRouter returned an HTML error page (HTTP 502)");
  });

  it("clips a long message and a long list of issues", async () => {
    const issues = Array.from({ length: 2_000 }, (_, i) => ({
      path: `questions.${i}`,
      message: `question ${i} is not valid`,
    }));
    const body = JSON.stringify({ error: { message: "Invalid request", issues } });
    const error = await fail(answering(422, body, { "content-type": "application/json" }));
    expect(error.issues).toHaveLength(2_000);
    expect(error.message).toMatch(/^Drex returned 422: Invalid request\n {2}questions\.0:/);
    expect(error.message.length).toBeLessThanOrEqual(MAX + 1 + GUIDANCE);
    expect(error.message).toContain("(request_id=req_big)");
  });

  it("keeps the billing guidance and the request id after clipping a 402", async () => {
    const body = JSON.stringify({ error: { message: "x".repeat(5_000) } });
    const error = await fail(answering(402, body, { "content-type": "application/json" }));
    expect(error).toBeInstanceOf(NaceInsufficientCreditError);
    expect(error.message).toContain(`Add funds at ${BILLING_URL}`);
    expect(error.message).toContain("(request_id=req_big)");
    expect(error.message.length).toBeLessThanOrEqual(MAX + 1 + GUIDANCE);
    expect(error.body.error.message).toHaveLength(5_000);
  });

  it("keeps the billing guidance when the request id is oversized", async () => {
    const body = JSON.stringify({ error: { message: "Insufficient funds" } });
    const error = await fail(
      answering(402, body, {
        "content-type": "application/json",
        "x-request-id": "r".repeat(5_000),
      }),
    );
    expect(error.message).toContain(`Add funds at ${BILLING_URL}`);
    expect(error.message.length).toBeLessThanOrEqual(MAX + 1 + GUIDANCE);
  });
});

describe("what an error shows", () => {
  const questions = { q: { type: "noul" as const, instructions: "?" } };
  const respond = (status: number, body: unknown, headers: Record<string, string> = {}) =>
    (async () =>
      new Response(JSON.stringify(body), {
        status,
        headers: { "content-type": "application/json", "x-request-id": "req_ok", ...headers },
      })) as typeof globalThis.fetch;
  const fail = (fetch: typeof globalThis.fetch, extra: Record<string, unknown> = {}) =>
    new NaceClassifier(offline(fetch, { questions, maxRetries: 0, ...extra }))
      .invoke("x")
      .catch((e) => e);

  it("never adds anything from the request to the message or the cause", async () => {
    const body = { error: { type: "invalid_request_error", message: "Invalid request" } };
    const error = await new NaceClassifier(
      offline(respond(400, body), {
        apiKey: "nace_sk_REQUEST-SECRET-KEY",
        defaultHeaders: { "x-tenant": "REQUEST-SECRET-HEADER" },
        questions: { q: { type: "noul", instructions: "REQUEST-SECRET-QUESTION" } },
        maxRetries: 0,
      }),
    )
      .invoke({ note: "REQUEST-SECRET-STATE" })
      .catch((e) => e);
    expect(error).toBeInstanceOf(NaceBadRequestError);
    expect(error.message).toContain("Invalid request");
    expect(error.message).not.toContain("REQUEST-SECRET");
    expect(String(error.cause?.message)).not.toContain("REQUEST-SECRET");
    expect(inspect(error, { depth: 5 })).not.toContain("REQUEST-SECRET");
    expect(JSON.stringify(error)).not.toContain("REQUEST-SECRET");
  });

  const hugeIssue = `${"f".repeat(1_000)}ISSUE-SECRET${"g".repeat(99_000)}`;
  const providers: [string, Record<string, unknown>, unknown, RegExp][] = [
    [
      "Nace",
      {},
      {
        error: {
          type: "invalid_request_error",
          message: "Invalid request",
          issues: [{ path: "questions.q", message: hugeIssue }],
        },
      },
      /^Drex returned 422: Invalid request\n {2}questions\.q: f{100}/,
    ],
    [
      "OpenRouter",
      { baseUrl: "https://openrouter.ai/api/v1", apiKey: "sk-or-test" },
      {
        error: {
          code: 422,
          message: `HTTP 422: ${JSON.stringify({ detail: [{ loc: ["body", "questions", "q"], msg: hugeIssue }] })}`,
        },
      },
      /^OpenRouter returned 422: the request is invalid\n {2}questions\.q: f{100}/,
    ],
  ];

  it.each(providers)(
    "keeps a long validation issue out of logs (%s)",
    async (_name, extra, body, format) => {
      const error = await fail(respond(422, body), extra);
      expect(error).toBeInstanceOf(NaceUnprocessableEntityError);
      expect(error.message).toMatch(format);
      expect(error.message).not.toContain("ISSUE-SECRET");
      const shown = inspect(error, { depth: 5 });
      expect(shown.length).toBeLessThan(5_000);
      expect(shown).not.toContain("ISSUE-SECRET");
      expect(JSON.stringify(error)).not.toContain("ISSUE-SECRET");
      expect(Object.keys(error)).not.toContain("issues");
      expect(error.issues).toHaveLength(1);
      expect(error.issues[0].path).toBe("questions.q");
      expect(error.issues[0].message).toBe(hugeIssue);
    },
  );

  it("bounds the error type and drops a request id that is not an id", async () => {
    const body = { error: { type: `${"t".repeat(5_000)}TYPE-SECRET`, message: "bad" } };
    const error = await fail(
      respond(400, body, { "x-request-id": "not an id: <b>RID-SECRET</b>" }),
    );
    expect(error.type).toHaveLength(64);
    expect(error.type).toMatch(/^t{63}…$/);
    expect(error.requestId).toBeUndefined();
    expect(error.message).not.toContain("request_id");
    const shown = inspect(error, { depth: 5 });
    expect(shown).not.toContain("TYPE-SECRET");
    expect(shown).not.toContain("RID-SECRET");
    expect(JSON.stringify(error)).not.toContain("SECRET");
    expect(error.body.error.type).toContain("TYPE-SECRET");
  });

  it("keeps a request id of 128 allowed characters and drops 129", async () => {
    const body = { error: { message: "bad" } };
    const kept = await fail(respond(400, body, { "x-request-id": `a.B_9:-${"c".repeat(121)}` }));
    expect(kept.requestId).toHaveLength(128);
    expect(kept.message).toContain(`(request_id=${kept.requestId})`);
    const dropped = await fail(respond(400, body, { "x-request-id": "c".repeat(129) }));
    expect(dropped.requestId).toBeUndefined();
    expect(dropped.message).not.toContain("request_id");
  });
});

describe("402 guidance", () => {
  const noFunds = async () =>
    new Response(JSON.stringify({ error: { message: "No available funds" } }), {
      status: 402,
      headers: { "content-type": "application/json" },
    });
  const q = { q: { type: "noul" as const, instructions: "?" } };

  it("sends an OpenRouter user to OpenRouter's credits page", async () => {
    const error = await new NaceClassifier({
      apiKey: "sk-or-test",
      baseUrl: "https://openrouter.ai/api/v1",
      questions: q,
      fetch: noFunds as typeof globalThis.fetch,
    })
      .invoke("x")
      .catch((e) => e);
    expect(OPENROUTER_CREDITS_URL).toBe("https://openrouter.ai/settings/credits");
    expect(error.message).toContain(`Add credits at ${OPENROUTER_CREDITS_URL}`);
    expect(error.message).not.toContain(BILLING_URL);
  });

  it("sends a Drex user to the Drex billing page", async () => {
    const error = await new NaceClassifier(
      offline(noFunds as typeof globalThis.fetch, { questions: q }),
    )
      .invoke("x")
      .catch((e) => e);
    expect(error.message).toContain(`Add funds at ${BILLING_URL}`);
    expect(error.message).not.toContain("openrouter.ai");
  });
});

describe("modelKwargs", () => {
  const q = { q: { type: "noul" as const, instructions: "?" } };

  it.each(["model", "state", "questions", "images"])("refuses the reserved key %s", (key) => {
    const server = fakeDrex();
    expect(
      () =>
        new NaceClassifier(offline(server.fetch, { questions: q, modelKwargs: { [key]: "x" } })),
    ).toThrow(new RegExp(`modelKwargs.*"${key}"`));
    expect(server.requests).toHaveLength(0);
  });

  it("does not let modelKwargs.images reach a text-only model", () => {
    expect(
      () =>
        new NaceClassifier(
          offline(fakeDrex().fetch, {
            model: "drex-v2.0",
            questions: q,
            modelKwargs: { images: [PNG_1x1] },
          }),
        ),
    ).toThrow(NaceError);
  });

  it("still sends other fields", async () => {
    const server = fakeDrex();
    await new NaceClassifier(
      offline(server.fetch, { questions: q, modelKwargs: { seed: 7 } }),
    ).invoke("x");
    expect(server.requests[0]?.body.seed).toBe(7);
  });
});
