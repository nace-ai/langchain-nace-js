import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { inspect } from "node:util";

import { AIMessage, HumanMessage, SystemMessage, ToolMessage } from "@langchain/core/messages";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  NaceAuthenticationError,
  NaceClassifier,
  NaceError,
  NaceImageError,
  NaceImageModelError,
  NaceInsufficientCreditError,
  NaceMediaError,
  NaceOverloadedError,
  NaceRateLimitError,
  NaceUnprocessableEntityError,
  VERSION,
} from "../../src/index.js";
import { fakeDrex, offline, okBody, PNG_1x1 } from "./helpers.js";

const URGENT = { urgent: { type: "noul" as const, instructions: "Is this urgent?" } };

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("request", () => {
  it("posts state, questions and the default model drex-max", async () => {
    const drex = fakeDrex();
    const classifier = new NaceClassifier(offline(drex.fetch, { questions: URGENT }));
    await classifier.invoke("My payouts are failing");
    expect(drex.requests).toHaveLength(1);
    const [request] = drex.requests;
    expect(request!.url).toBe("https://drex.test/v1/systemone");
    expect(request!.body).toEqual({
      model: "drex-max",
      state: "My payouts are failing",
      questions: { urgent: { type: "noul", instructions: "Is this urgent?" } },
    });
    expect(request!.headers.get("authorization")).toBe("Bearer nace_sk_test");
  });

  it("sends the attribution headers", async () => {
    const drex = fakeDrex();
    await new NaceClassifier(offline(drex.fetch, { questions: URGENT })).invoke("x");
    const headers = drex.requests[0]!.headers;
    expect(headers.get("user-agent")).toBe(`langchain-nace-js/${VERSION}`);
    expect(headers.get("x-nace-client")).toBe(`langchain-nace-js/${VERSION}`);
  });

  it("keeps VERSION equal to package.json", () => {
    const pkg = JSON.parse(readFileSync(new URL("../../package.json", import.meta.url), "utf8"));
    expect(VERSION).toBe(pkg.version);
  });

  it("always sends instructions as a string, in type/instructions/criteria order", async () => {
    const drex = fakeDrex();
    const classifier = new NaceClassifier(
      offline(drex.fetch, {
        questions: {
          a: { type: "noul", criteria: { true: "Yes, refund", false: "No" } },
          b: { type: "choice", criteria: { billing: null, tech: "Bugs" } },
          c: { type: "score", criteria: ["low", "high"], instructions: { rubric: "strict" } },
        },
      }),
    );
    await classifier.invoke("x");
    const q = drex.requests[0]!.body.questions;
    expect(q.a).toEqual({
      type: "noul",
      instructions: "",
      criteria: { true: "Yes, refund", false: "No" },
    });
    expect(Object.keys(q.a)).toEqual(["type", "instructions", "criteria"]);
    expect(q.b).toEqual({
      type: "choice",
      instructions: "",
      criteria: { billing: null, tech: "Bugs" },
    });
    expect(q.c.instructions).toBe('{"rubric":"strict"}');
  });

  it("passes unknown question types through", async () => {
    const drex = fakeDrex();
    const classifier = new NaceClassifier(
      offline(drex.fetch, { questions: { future: { type: "rank", items: ["a", "b"] } } }),
    );
    const response = await classifier.invoke("x");
    expect(drex.requests[0]!.body.questions.future).toEqual({
      type: "rank",
      instructions: "",
      items: ["a", "b"],
    });
    expect(response.answers.future).toEqual({ type: "rank", value: 42 });
  });

  it("uses call-option questions in place of the constructor's", async () => {
    const drex = fakeDrex();
    const classifier = new NaceClassifier(offline(drex.fetch, { questions: URGENT }));
    await classifier.invoke("x", { questions: { other: { type: "noul", instructions: "?" } } });
    expect(Object.keys(drex.requests[0]!.body.questions)).toEqual(["other"]);
  });

  it("refuses a call with no questions before sending", async () => {
    const drex = fakeDrex();
    const classifier = new NaceClassifier(offline(drex.fetch));
    await expect(classifier.invoke("x")).rejects.toThrow(/No questions to ask/);
    expect(drex.requests).toHaveLength(0);
  });

  it("validates questions, naming the id and never the value", () => {
    expect(
      () =>
        new NaceClassifier(
          offline(fakeDrex().fetch, { questions: { secret_q: { type: "choice", criteria: {} } } }),
        ),
    ).toThrow(/Invalid question "secret_q": .*at least 1 option/);
    expect(
      () => new NaceClassifier(offline(fakeDrex().fetch, { questions: { q: { type: "noul" } } })),
    ).toThrow(/criteria or instructions/);
  });

  it("sends modelKwargs as top-level body fields", async () => {
    const drex = fakeDrex();
    await new NaceClassifier(
      offline(drex.fetch, { questions: URGENT, modelKwargs: { seed: 7 } }),
    ).invoke("x");
    expect(drex.requests[0]!.body.seed).toBe(7);
  });

  it("encodes messages with tool calls as role/content JSON", async () => {
    const drex = fakeDrex();
    await new NaceClassifier(offline(drex.fetch, { questions: URGENT })).invoke([
      new SystemMessage("be careful"),
      new HumanMessage("delete ticket 7"),
      new AIMessage({
        content: "",
        tool_calls: [{ id: "c1", name: "delete_ticket", args: { id: 7 } }],
      }),
      new ToolMessage({ content: "deleted", tool_call_id: "c1", name: "delete_ticket" }),
    ]);
    expect(drex.requests[0]!.body.state).toEqual([
      { role: "system", content: "be careful" },
      { role: "user", content: "delete ticket 7" },
      {
        role: "assistant",
        content: "",
        tool_calls: [
          {
            id: "c1",
            type: "function",
            function: { name: "delete_ticket", arguments: '{"id":7}' },
          },
        ],
      },
      { role: "tool", content: "deleted", tool_call_id: "c1", name: "delete_ticket" },
    ]);
  });

  it("turns Dates into ISO strings and refuses cycles and bare scalars", async () => {
    const drex = fakeDrex();
    const classifier = new NaceClassifier(offline(drex.fetch, { questions: URGENT }));
    await classifier.invoke({ at: new Date("2026-01-02T03:04:05Z") });
    expect(drex.requests[0]!.body.state).toEqual({ at: "2026-01-02T03:04:05.000Z" });
    const cycle: Record<string, unknown> = { a: 1 };
    cycle.self = cycle;
    await expect(classifier.invoke(cycle as never)).rejects.toThrow(/Circular/);
    await expect(classifier.invoke(5 as never)).rejects.toThrow(NaceError);
    expect(drex.requests).toHaveLength(1);
  });
});

describe("images", () => {
  it("sends explicit images first, then images lifted from messages", async () => {
    const drex = fakeDrex();
    const classifier = new NaceClassifier(offline(drex.fetch, { questions: URGENT }));
    await classifier.invoke(
      [
        new HumanMessage({
          content: [
            { type: "text", text: "Is this screen dangerous?" },
            { type: "image", source_type: "base64", data: PNG_1x1, mime_type: "image/png" },
          ],
        }),
      ],
      { images: [`data:image/png;base64,${PNG_1x1}`] },
    );
    const body = drex.requests[0]!.body;
    expect(body.images).toEqual([
      `data:image/png;base64,${PNG_1x1}`,
      `data:image/png;base64,${PNG_1x1}`,
    ]);
    expect(body.state[0].content).toEqual([
      { type: "text", text: "Is this screen dangerous?" },
      { type: "text", text: "[Image 2]" },
    ]);
    expect(JSON.stringify(body.state)).not.toContain(PNG_1x1);
  });

  it("lifts images from plain { role, content } objects too (the 0.1.0 Python bug)", async () => {
    const drex = fakeDrex();
    await new NaceClassifier(offline(drex.fetch, { questions: URGENT })).invoke([
      {
        role: "user",
        content: [
          { type: "text", text: "Which colour?" },
          { type: "image_url", image_url: { url: `data:image/png;base64,${PNG_1x1}` } },
        ],
        extra: "kept",
      },
    ]);
    const body = drex.requests[0]!.body;
    expect(body.images).toEqual([`data:image/png;base64,${PNG_1x1}`]);
    expect(body.state).toEqual([
      {
        role: "user",
        content: [
          { type: "text", text: "Which colour?" },
          { type: "text", text: "[Image 1]" },
        ],
        extra: "kept",
      },
    ]);
  });

  it("accepts v1 image blocks, bytes, base64 and file paths", async () => {
    const drex = fakeDrex();
    const dir = mkdtempSync(join(tmpdir(), "nace-"));
    const path = join(dir, "red.png");
    writeFileSync(path, Buffer.from(PNG_1x1, "base64"));
    await new NaceClassifier(offline(drex.fetch, { questions: URGENT })).invoke(
      [new HumanMessage({ content: [{ type: "image", data: PNG_1x1, mimeType: "image/png" }] })],
      { images: [Buffer.from(PNG_1x1, "base64"), PNG_1x1, path] },
    );
    const images = drex.requests[0]!.body.images;
    expect(images).toHaveLength(4);
    expect(images[0]).toBe(`data:image/png;base64,${PNG_1x1}`);
    expect(images[1]).toBe(PNG_1x1);
    expect(images[2]).toBe(`data:image/png;base64,${PNG_1x1}`);
    expect(images[3]).toBe(`data:image/png;base64,${PNG_1x1}`);
  });

  it("passes workspace_file references through", async () => {
    const drex = fakeDrex();
    const ref = { type: "workspace_file", workspace_id: "ws_1", file_id: "f_1" };
    await new NaceClassifier(offline(drex.fetch, { questions: URGENT })).invoke("x", {
      images: [ref],
    });
    expect(drex.requests[0]!.body.images).toEqual([ref]);
  });

  it("refuses remote URLs: Drex never fetches them", async () => {
    const classifier = new NaceClassifier(offline(fakeDrex().fetch, { questions: URGENT }));
    await expect(
      classifier.invoke("x", { images: ["https://example.com/cat.png"] }),
    ).rejects.toThrow(NaceImageError);
  });

  it("refuses images on text-only models before sending", async () => {
    const drex = fakeDrex();
    const classifier = new NaceClassifier(
      offline(drex.fetch, { questions: URGENT, model: "drex-v2.0" }),
    );
    await expect(classifier.invoke("x", { images: [PNG_1x1] })).rejects.toThrow(
      NaceImageModelError,
    );
    await expect(
      classifier.invoke([
        {
          role: "user",
          content: [{ type: "image_url", image_url: { url: `data:image/png;base64,${PNG_1x1}` } }],
        },
      ]),
    ).rejects.toThrow(NaceImageModelError);
    const prefixed = new NaceClassifier(
      offline(drex.fetch, { questions: URGENT, model: "nace-ai/drex-v2.0" }),
    );
    await expect(prefixed.invoke("x", { images: [PNG_1x1] })).rejects.toThrow(NaceImageModelError);
    expect(drex.requests).toHaveLength(0);
  });

  it("refuses audio and files in state rather than dropping them", async () => {
    const classifier = new NaceClassifier(offline(fakeDrex().fetch, { questions: URGENT }));
    await expect(
      classifier.invoke([
        new HumanMessage({ content: [{ type: "audio", data: "AAAA", mimeType: "audio/wav" }] }),
      ]),
    ).rejects.toThrow(NaceMediaError);
  });
});

describe("response", () => {
  it("returns typed answers with full probabilities and TypeSafe's fields", async () => {
    const drex = fakeDrex();
    const classifier = new NaceClassifier(
      offline(drex.fetch, {
        questions: {
          urgent: { type: "noul", instructions: "?" },
          team: { type: "choice", criteria: { billing: null, tech: null } },
          risk: { type: "score", criteria: ["low", "mid", "high"] },
        },
      }),
    );
    const r = await classifier.invoke("x");
    expect(r.model).toBe("drex-max");
    expect(r.nouls.urgent!.noul).toBe(0.9);
    expect(r.choices.team!.probabilities).toEqual({ billing: 0.8, tech: 0.2 });
    expect(r.scores.risk!.legend).toEqual({ "0": "low", "1": "mid", "2": "high" });
    expect(r.scores.risk!.legend[0]).toBe("low");
    expect(r.usage).toEqual({ inputTokens: 120, outputTokens: 0, imageTokens: 0 });
    expect(r.requestId).toBe("req_header");
    expect(r.evaluationTimeMs).toBe(87);
    expect(r.raw.request_id).toBe("req_body");
    expect(Object.keys(r).sort()).toEqual(
      ["answers", "evaluationTimeMs", "model", "requestId", "usage"].sort(),
    );
  });
});

describe("configuration", () => {
  it("reads NACE_API_KEY, NACE_BASE_URL and NACE_DEFAULT_DECISION_MODEL", async () => {
    vi.stubEnv("NACE_API_KEY", "nace_sk_env");
    vi.stubEnv("NACE_BASE_URL", "https://env.test/");
    vi.stubEnv("NACE_DEFAULT_DECISION_MODEL", "drex-v2.0");
    const drex = fakeDrex();
    await new NaceClassifier({ questions: URGENT, fetch: drex.fetch }).invoke("x");
    expect(drex.requests[0]!.url).toBe("https://env.test/v1/systemone");
    expect(drex.requests[0]!.body.model).toBe("drex-v2.0");
    expect(drex.requests[0]!.headers.get("authorization")).toBe("Bearer nace_sk_env");
  });

  it("treats empty env vars as unset (a real Python bug)", async () => {
    vi.stubEnv("NACE_API_KEY", "nace_sk_env");
    vi.stubEnv("NACE_BASE_URL", "");
    vi.stubEnv("NACE_DEFAULT_DECISION_MODEL", "  ");
    const drex = fakeDrex();
    await new NaceClassifier({ questions: URGENT, fetch: drex.fetch }).invoke("x");
    expect(drex.requests[0]!.url).toBe("https://console.nace.ai/v1/systemone");
    expect(drex.requests[0]!.body.model).toBe("drex-max");
  });

  it("requires an API key", () => {
    vi.stubEnv("NACE_API_KEY", "");
    expect(() => new NaceClassifier({ questions: URGENT })).toThrow(/API key is required/);
  });

  it("never exposes the API key", () => {
    const classifier = new NaceClassifier(offline(fakeDrex().fetch, { questions: URGENT }));
    expect(inspect(classifier, { depth: 5 })).not.toContain("nace_sk_test");
    expect(JSON.stringify(classifier.lc_kwargs)).not.toContain("nace_sk_test");
    expect(JSON.stringify(classifier.toJSON())).not.toContain("nace_sk_test");
  });

  it("speaks OpenRouter's format on openrouter.ai", async () => {
    const drex = fakeDrex();
    const classifier = new NaceClassifier({
      apiKey: "sk-or-test",
      baseUrl: "https://openrouter.ai/api/v1",
      model: "nace-ai/drex-max",
      questions: URGENT,
      fetch: drex.fetch,
    });
    await classifier.invoke({ ticket: "x" }, { images: [PNG_1x1] });
    const request = drex.requests[0]!;
    expect(request.url).toBe("https://openrouter.ai/api/v1/systemone");
    expect(request.body.images).toBeUndefined();
    expect(request.body.state).toEqual([
      { type: "text", text: '{"ticket":"x"}' },
      { type: "image_url", image_url: { url: `data:image/png;base64,${PNG_1x1}` } },
    ]);
    expect(request.headers.get("http-referer")).toBe("https://nace.ai");
    expect(request.headers.get("x-openrouter-title")).toBe("Drex for LangChain");
  });
});

describe("errors and retries", () => {
  it("shows a 422 as path: message per issue", async () => {
    const drex = fakeDrex([
      {
        status: 422,
        body: {
          error: {
            type: "invalid_request_error",
            message: "Invalid request",
            issues: [
              { path: "questions", message: "Drex Max allows at most 16 questions." },
              { path: "images.0", message: "not an image" },
            ],
          },
        },
      },
    ]);
    const error = await new NaceClassifier(offline(drex.fetch, { questions: URGENT }))
      .invoke("x")
      .catch((e) => e);
    expect(error).toBeInstanceOf(NaceUnprocessableEntityError);
    expect(NaceUnprocessableEntityError.isInstance(error)).toBe(true);
    expect(error.message).toBe(
      "Drex returned 422: Invalid request\n" +
        "  questions: Drex Max allows at most 16 questions.\n" +
        "  images.0: not an image\n" +
        "(request_id=req_header)",
    );
    expect(error.issues).toHaveLength(2);
    expect(error.statusCode).toBe(422);
    expect(JSON.stringify(error)).not.toContain('"body"');
  });

  it("points a 402 to billing", async () => {
    const drex = fakeDrex([{ status: 402, body: { error: { message: "No available funds" } } }]);
    const error = await new NaceClassifier(offline(drex.fetch, { questions: URGENT }))
      .invoke("x")
      .catch((e) => e);
    expect(error).toBeInstanceOf(NaceInsufficientCreditError);
    expect(error.message).toContain("Add funds at https://console.nace.ai/dashboard/billing");
  });

  it("makes a 401 a NaceAuthenticationError without retrying", async () => {
    const drex = fakeDrex([{ status: 401, body: { error: { message: "Invalid API key" } } }]);
    const error = await new NaceClassifier(offline(drex.fetch, { questions: URGENT }))
      .invoke("x")
      .catch((e) => e);
    expect(error).toBeInstanceOf(NaceAuthenticationError);
    expect(drex.requests).toHaveLength(1);
  });

  it("retries 429 and 529 after retry-after-ms", async () => {
    const drex = fakeDrex([
      {
        status: 429,
        headers: { "retry-after-ms": "30" },
        body: { error: { message: "slow down" } },
      },
      { status: 529, headers: { "retry-after-ms": "30" }, body: { error: { message: "busy" } } },
    ]);
    const started = Date.now();
    const r = await new NaceClassifier(offline(drex.fetch, { questions: URGENT })).invoke("x");
    expect(r.nouls.urgent!.noul).toBe(0.9);
    expect(drex.requests).toHaveLength(3);
    expect(Date.now() - started).toBeGreaterThanOrEqual(55);
  });

  it("throws the last error once retries run out", async () => {
    const busy = {
      status: 529,
      headers: { "retry-after-ms": "1" },
      body: { error: { message: "busy" } },
    };
    const drex = fakeDrex([busy, busy, busy]);
    const error = await new NaceClassifier(offline(drex.fetch, { questions: URGENT }))
      .invoke("x")
      .catch((e) => e);
    expect(error).toBeInstanceOf(NaceOverloadedError);
    expect(drex.requests).toHaveLength(3);
  });

  it("has no total retry budget: a slow failure is still retried", async () => {
    // With nace-sdk's default 30 s budget, the retry after a slow failure is skipped.
    // Simulate a slow failure by advancing time instead of waiting.
    let now = Date.now();
    const spy = vi.spyOn(Date, "now").mockImplementation(() => now);
    const drex = fakeDrex([
      () => {
        now += 45_000;
        return {
          status: 529,
          headers: { "retry-after-ms": "1" },
          body: { error: { message: "busy" } },
        };
      },
    ]);
    const r = await new NaceClassifier(offline(drex.fetch, { questions: URGENT })).invoke("x");
    spy.mockRestore();
    expect(r.nouls.urgent).toBeDefined();
    expect(drex.requests).toHaveLength(2);
  });

  it("keeps the rate limit delay on the error", async () => {
    const limited = {
      status: 429,
      headers: { "retry-after-ms": "1" },
      body: { error: { message: "x" } },
    };
    const drex = fakeDrex([limited]);
    const error = await new NaceClassifier(
      offline(drex.fetch, { questions: URGENT, maxRetries: 0 }),
    )
      .invoke("x")
      .catch((e) => e);
    expect(error).toBeInstanceOf(NaceRateLimitError);
    expect(error.retryAfterMs).toBe(1);
  });

  it("says OpenRouter, and unwraps its raw error text", async () => {
    const drex = fakeDrex([
      {
        status: 422,
        body: {
          error: {
            message:
              'HTTP 422: {"detail":[{"loc":["body","questions"],"msg":"too many questions"}]}',
          },
        },
      },
    ]);
    const error = await new NaceClassifier({
      apiKey: "sk-or",
      baseUrl: "https://openrouter.ai/api/v1",
      questions: URGENT,
      fetch: drex.fetch,
    })
      .invoke("x")
      .catch((e) => e);
    expect(error.message).toMatch(
      /^OpenRouter returned 422: the request is invalid\n {2}questions: too many questions/,
    );
  });
});

describe("batch", () => {
  it("answers every input, at most 8 at a time by default", async () => {
    let inFlight = 0;
    let peak = 0;
    const drex = fakeDrex(
      Array.from({ length: 20 }, () => (body: any) => {
        inFlight += 1;
        peak = Math.max(peak, inFlight);
        return { delayMs: 10, body: okBody(body) };
      }),
    );
    const wrapped = (async (...args: Parameters<typeof fetch>) => {
      try {
        return await drex.fetch(...args);
      } finally {
        inFlight -= 1;
      }
    }) as typeof fetch;
    const classifier = new NaceClassifier(offline(wrapped, { questions: URGENT }));
    const results = await classifier.batch(Array.from({ length: 20 }, (_, i) => `ticket ${i}`));
    expect(results).toHaveLength(20);
    expect(peak).toBeLessThanOrEqual(8);
  });
});
