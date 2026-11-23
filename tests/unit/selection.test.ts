import { AIMessage, HumanMessage, ToolMessage } from "@langchain/core/messages";
import { BaseTracer, type Run } from "@langchain/core/tracers/base";
import fc from "fast-check";
import { NaceClient } from "nace-sdk";
import { describe, expect, it, vi } from "vitest";
import { VERSION } from "../../src/index.js";
import {
  estimateRequestTokens,
  exchangeRoute,
  filterEligibleCandidates,
  NaceModelSelector,
  projectRouteMessages,
  selectionFromOutcome,
  validateRouteResponse,
  wireCandidates,
} from "../../src/selection.js";
import { fakeDrex, offline } from "./helpers.js";

const CANARY = "CANARY-selection-9f3c";

class Collector extends BaseTracer {
  name = "collector";
  runs: Run[] = [];

  protected async persistRun(run: Run): Promise<void> {
    this.runs.push(run);
  }
}

function probs(ids: string[], selected: string, top = 0.8): Record<string, number> {
  const rest = ids.length === 1 ? 0 : (1 - top) / (ids.length - 1);
  return Object.fromEntries(ids.map((id) => [id, id === selected ? top : rest]));
}

function routeReply(selected: string, ids: string[], top = 0.8) {
  return {
    selected,
    probabilities: probs(ids, selected, top),
    request_id: "req_route",
  };
}

function selector(fetch: typeof globalThis.fetch, extra: Record<string, unknown> = {}) {
  return new NaceModelSelector(offline(fetch, extra));
}

const turns = [{ role: "user" as const, content: "Hello" }];

describe("projectRouteMessages", () => {
  it("case 10: keeps a placeholder and drops the tool call and the tool result", () => {
    const projected = projectRouteMessages(
      {
        systemPrompt: "be brief",
        messages: [
          new HumanMessage({
            content: [
              { type: "text", text: "look" },
              { type: "image_url", image_url: { url: `data:image/png;base64,${CANARY}` } },
            ],
          }),
          new AIMessage({
            content: "checking",
            tool_calls: [{ id: "call_1", name: "lookup", args: { q: CANARY } }],
          }),
          new ToolMessage({ content: CANARY, tool_call_id: "call_1" }),
        ],
      },
      "conversation",
    );
    expect(projected).toEqual([
      { role: "system", content: "be brief" },
      { role: "user", content: "look\n[image omitted]" },
      { role: "assistant", content: "checking" },
    ]);
    expect(JSON.stringify(projected)).not.toContain(CANARY);
  });

  it("latest human with no real text is no_input in both context modes", () => {
    const blank = [new HumanMessage("old text"), new HumanMessage("   ")];
    expect(projectRouteMessages({ messages: blank }, "latest_human")).toBeNull();
    expect(projectRouteMessages({ messages: blank }, "conversation")).toBeNull();
    const imageOnly = [
      new HumanMessage("old text"),
      new HumanMessage({
        content: [{ type: "image_url", image_url: { url: "data:image/png;base64,aa" } }],
      }),
    ];
    expect(projectRouteMessages({ messages: imageOnly }, "conversation")).toBeNull();
    expect(projectRouteMessages({ messages: imageOnly }, "latest_human")).toBeNull();
  });

  it("sends the system text and only the latest human turn", () => {
    const projected = projectRouteMessages(
      {
        systemPrompt: "rules",
        messages: [new HumanMessage("first"), new AIMessage("answer"), new HumanMessage("second")],
      },
      "latest_human",
    );
    expect(projected).toEqual([
      { role: "system", content: "rules" },
      { role: "user", content: "second" },
    ]);
  });

  it("returns null when no human text remains", () => {
    expect(projectRouteMessages({ messages: [new AIMessage("only")] }, "conversation")).toBeNull();
    expect(projectRouteMessages({ messages: [new HumanMessage("  ")] }, "conversation")).toBeNull();
  });

  it("keeps the newest turns inside 24000 characters and always keeps the latest human", () => {
    const old = "o".repeat(100);
    const latest = "L".repeat(30_000);
    const projected = projectRouteMessages(
      {
        messages: [new HumanMessage(old), new AIMessage("a".repeat(100)), new HumanMessage(latest)],
      },
      "conversation",
    );
    expect(projected).toEqual([{ role: "user", content: "L".repeat(24_000) }]);
  });

  it("fills the budget with the newest other turns and skips one that does not fit", () => {
    const older = `O${"a".repeat(11_999)}`;
    const newer = `N${"b".repeat(11_999)}`;
    const projected = projectRouteMessages(
      {
        messages: [new HumanMessage("hi"), new AIMessage(older), new AIMessage(newer)],
      },
      "conversation",
    );
    expect(projected?.map((message) => message.content[0])).toEqual(["h", "N"]);
    const huge = "H".repeat(24_000);
    const skipped = projectRouteMessages(
      {
        messages: [new AIMessage("keep-me"), new AIMessage(huge), new HumanMessage("hi")],
      },
      "conversation",
    );
    expect(skipped?.map((message) => message.content)).toEqual(["keep-me", "hi"]);
  });

  it("sends at most 1024 messages", () => {
    const messages = Array.from({ length: 1100 }, (_, index) => new HumanMessage(`m${index}`));
    const projected = projectRouteMessages({ messages }, "conversation");
    expect(projected?.length).toBe(1024);
    expect(projected?.at(-1)?.content).toBe("m1099");
  });

  it("holds for generated conversations", () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.record({
            role: fc.constantFrom("system", "user", "assistant", "tool"),
            text: fc.string({ maxLength: 30 }),
            secret: fc.constant(CANARY),
          }),
          { maxLength: 20 },
        ),
        (rows) => {
          const messages = rows.map((row) => ({
            role: row.role,
            content: row.role === "tool" ? `${row.text}${CANARY}` : row.text,
          }));
          const projected = projectRouteMessages({ messages }, "conversation");
          let latestUser = "";
          for (const row of rows) {
            if (row.role === "user") latestUser = row.text;
          }
          const human = latestUser.split("\n").some((line) => {
            const trimmed = line.trim();
            return trimmed !== "" && !/^\[[^\]\n]+ omitted\]$/.test(trimmed);
          });
          if (!human) {
            expect(projected).toBeNull();
            return;
          }
          expect(projected).not.toBeNull();
          expect(projected!.length).toBeLessThanOrEqual(1024);
          const chars = projected!.reduce((sum, message) => sum + message.content.length, 0);
          expect(chars).toBeLessThanOrEqual(24_000);
          expect(projected!.some((message) => message.role === "user")).toBe(true);
          expect(JSON.stringify(projected)).not.toContain(CANARY);
          for (const message of projected!) {
            expect(["system", "user", "assistant"]).toContain(message.role);
            expect(typeof message.content).toBe("string");
          }
        },
      ),
      { numRuns: 50 },
    );
  });
});

describe("filterEligibleCandidates", () => {
  const candidates = [
    { id: "both", images: true, tools: true, context: 1000 },
    { id: "no-images", images: false, tools: true, context: 1000 },
    { id: "no-tools", images: true, tools: false, context: 1000 },
    { id: "tiny", images: true, tools: true, context: 1 },
  ];

  it("case 13: drops candidates that refuse the request's images or tools", () => {
    const messages = [
      {
        role: "user",
        content: [
          { type: "text", text: "see" },
          { type: "image_url", image_url: "data:image/png,xx" },
        ],
      },
    ];
    const eligible = filterEligibleCandidates(candidates, {
      messages,
      tools: [{ name: "lookup", description: "find" }],
    });
    expect(eligible).toEqual(["both"]);
  });

  it("treats file and media image blocks as images, and structured output as tools", () => {
    const candidates = [
      { id: "blind", images: false, tools: true },
      { id: "sighted", images: true, tools: true },
      { id: "notool", images: true, tools: false },
    ];
    const fileImage = {
      content: [{ type: "file", source: { media_type: "image/png" } }],
    };
    expect(
      filterEligibleCandidates(candidates, { messages: [fileImage], tools: [] }).includes("blind"),
    ).toBe(false);
    const mediaImage = {
      content: [{ type: "media", mime_type: "image/jpeg" }],
    };
    expect(
      filterEligibleCandidates(candidates, { messages: [mediaImage], tools: [] }),
    ).not.toContain("blind");
    const native = { content: [{ image: { format: "png", source: { bytes: "aa" } } }] };
    expect(filterEligibleCandidates(candidates, { messages: [native], tools: [] })).not.toContain(
      "blind",
    );
    const inputImage = { content: [{ type: "input_image", image_url: "data:image/png,aa" }] };
    expect(
      filterEligibleCandidates(candidates, { messages: [inputImage], tools: [] }),
    ).not.toContain("blind");
    const pdf = { content: [{ type: "file", mime_type: "application/pdf" }] };
    expect(filterEligibleCandidates(candidates, { messages: [pdf], tools: [] })).toContain("blind");
    expect(
      filterEligibleCandidates(
        [
          { id: "notool", images: true, tools: false },
          { id: "tool", images: true, tools: true },
        ],
        { messages: [{ content: "hi" }], tools: [], responseFormat: { type: "json" } },
      ),
    ).toEqual(["tool"]);
  });

  it("drops a profile whose context window is smaller than the estimate", () => {
    const text = "x".repeat(30);
    expect(estimateRequestTokens("", [{ role: "user", content: text }], [])).toBe(10);
    const eligible = filterEligibleCandidates(
      [
        { id: "wide", images: true, tools: true, context: 10 },
        { id: "narrow", images: true, tools: true, context: 9 },
        { id: "unbounded", images: true, tools: true },
      ],
      { messages: [{ role: "user", content: text }], tools: [] },
    );
    expect(eligible).toEqual(["wide", "unbounded"]);
  });

  it("keeps binding order for generated flags", () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.record({
            id: fc.string({ minLength: 1, maxLength: 6, unit: "grapheme" }),
            images: fc.boolean(),
            tools: fc.boolean(),
          }),
          { maxLength: 8 },
        ),
        fc.boolean(),
        fc.boolean(),
        (rows, image, withTools) => {
          const unique: typeof rows = [];
          const seen = new Set<string>();
          for (const row of rows) {
            if (row.id.trim() === "" || seen.has(row.id)) continue;
            seen.add(row.id);
            unique.push(row);
          }
          const messages = image
            ? [{ content: [{ type: "image_url", image_url: "data:image/png,aa" }] }]
            : [{ content: "hi" }];
          const eligible = filterEligibleCandidates(unique, {
            messages,
            tools: withTools ? [{ name: "lookup", description: "x" }] : [],
          });
          let last = -1;
          for (const id of eligible) {
            const index = unique.findIndex((row) => row.id === id);
            expect(index).toBeGreaterThan(last);
            last = index;
            const row = unique[index]!;
            if (image) expect(row.images).toBe(true);
            if (withTools) expect(row.tools).toBe(true);
          }
        },
      ),
      { numRuns: 50 },
    );
  });
});

describe("validateRouteResponse", () => {
  const ids = ["fast", "strong"];

  it("accepts a complete probability map", () => {
    const result = validateRouteResponse(
      { selected: "strong", probabilities: probs(ids, "strong"), request_id: "req_abc12345" },
      ids,
    );
    expect(result).toMatchObject({ ok: true, selected: "strong", requestId: "req_abc12345" });
  });

  it("drops a request id that is not req_ plus 8 to 64 letters", () => {
    const body = {
      selected: "strong",
      probabilities: probs(ids, "strong"),
      request_id: "medical-canary",
    };
    expect(validateRouteResponse(body, ids).requestId).toBeNull();
    expect(validateRouteResponse({ ...body, request_id: "req_header" }, ids).requestId).toBeNull();
    expect(validateRouteResponse({ ...body, request_id: "req_abc12345" }, ids).requestId).toBe(
      "req_abc12345",
    );
  });

  it("case 3: rejects a selected id that was not sent", () => {
    expect(
      validateRouteResponse({ selected: "other", probabilities: probs(ids, "fast") }, ids).ok,
    ).toBe(false);
  });

  it("case 4: rejects probability keys that differ from the ids sent", () => {
    expect(
      validateRouteResponse({ selected: "fast", probabilities: { fast: 0.4, other: 0.6 } }, ids).ok,
    ).toBe(false);
    expect(validateRouteResponse({ selected: "fast", probabilities: { fast: 1 } }, ids).ok).toBe(
      false,
    );
  });

  it("case 5: rejects NaN, negative and above 1", () => {
    for (const value of [Number.NaN, -0.1, 1.1, Number.POSITIVE_INFINITY]) {
      const probabilities = { fast: value, strong: 0 };
      expect(validateRouteResponse({ selected: "fast", probabilities }, ids).ok).toBe(false);
    }
  });

  it("case 6: rejects __proto__ and constructor unless that id was sent", () => {
    for (const selected of ["__proto__", "constructor"]) {
      const probabilities = JSON.parse(`{"${selected}":1,"fast":0}`) as Record<string, number>;
      expect(validateRouteResponse({ selected, probabilities }, ids).ok).toBe(false);
    }
    const sent = ["__proto__", "constructor"];
    const probabilities = JSON.parse('{"__proto__":0.4,"constructor":0.6}') as Record<
      string,
      number
    >;
    const accepted = validateRouteResponse({ selected: "__proto__", probabilities }, sent);
    expect(accepted.ok).toBe(true);
    if (accepted.ok) {
      expect(Object.getOwnPropertyDescriptor(accepted.probabilities, "__proto__")?.value).toBe(0.4);
    }
  });

  it("accepts generated valid maps and rejects a broken probability", () => {
    fc.assert(
      fc.property(
        fc.uniqueArray(
          fc
            .string({ minLength: 1, maxLength: 8, unit: "grapheme" })
            .filter((id) => id.trim() === id && !id.includes("__proto__")),
          { minLength: 1, maxLength: 4 },
        ),
        (ids) => {
          const share = 1 / ids.length;
          const probabilities = Object.fromEntries(ids.map((id) => [id, share]));
          const valid = validateRouteResponse(
            { selected: ids[0], probabilities, request_id: "req_ok" },
            ids,
          );
          expect(valid.ok).toBe(true);
          const broken = { ...probabilities, [ids[0]!]: 2 };
          expect(validateRouteResponse({ selected: ids[0], probabilities: broken }, ids).ok).toBe(
            false,
          );
        },
      ),
      { numRuns: 40 },
    );
  });
});

describe("selectionFromOutcome", () => {
  const context = {
    priority: "balanced" as const,
    sentIds: ["fast", "strong"],
    fallbackId: "fast",
    minProbability: 0.5,
  };

  it("case 2: maps each outcome to one reason and one model", () => {
    const rows = [
      {
        outcome: {
          type: "response" as const,
          selected: "strong",
          probabilities: probs(["fast", "strong"], "strong", 0.9),
          requestId: "req_route",
          latencyMs: 4,
        },
        reason: "selected",
        model: "strong",
      },
      {
        outcome: {
          type: "response" as const,
          selected: "strong",
          probabilities: probs(["fast", "strong"], "strong", 0.2),
          requestId: "req_route",
          latencyMs: 4,
        },
        reason: "low_probability",
        model: "fast",
      },
      {
        outcome: { type: "single_candidate" as const, id: "strong" },
        reason: "single_candidate",
        model: "strong",
      },
      { outcome: { type: "no_input" as const }, reason: "no_input", model: null },
      {
        outcome: { type: "no_eligible_candidate" as const },
        reason: "no_eligible_candidate",
        model: "fast",
      },
      {
        outcome: { type: "timeout" as const, requestId: null, latencyMs: 3 },
        reason: "timeout",
        model: "fast",
      },
      {
        outcome: { type: "connection_error" as const, latencyMs: 1 },
        reason: "connection_error",
        model: "fast",
      },
      {
        outcome: { type: "status" as const, status: 429, requestId: "req_route", latencyMs: 2 },
        reason: "rate_limited",
        model: "fast",
      },
      {
        outcome: { type: "status" as const, status: 503, requestId: null, latencyMs: 2 },
        reason: "unavailable",
        model: "fast",
      },
      {
        outcome: { type: "status" as const, status: 504, requestId: null, latencyMs: 2 },
        reason: "unavailable",
        model: "fast",
      },
      {
        outcome: { type: "status" as const, status: 529, requestId: null, latencyMs: 2 },
        reason: "unavailable",
        model: "fast",
      },
      {
        outcome: { type: "status" as const, status: 401, requestId: null, latencyMs: 2 },
        reason: "auth_error",
        model: "fast",
      },
      {
        outcome: { type: "status" as const, status: 403, requestId: null, latencyMs: 2 },
        reason: "auth_error",
        model: "fast",
      },
      {
        outcome: { type: "status" as const, status: 402, requestId: null, latencyMs: 2 },
        reason: "no_funds",
        model: "fast",
      },
      {
        outcome: { type: "status" as const, status: 413, requestId: null, latencyMs: 2 },
        reason: "invalid_request",
        model: "fast",
      },
      {
        outcome: { type: "status" as const, status: 422, requestId: null, latencyMs: 2 },
        reason: "invalid_request",
        model: "fast",
      },
      {
        outcome: { type: "invalid_response" as const, requestId: "req_route", latencyMs: 2 },
        reason: "invalid_response",
        model: "fast",
      },
    ];
    for (const row of rows) {
      const selection = selectionFromOutcome(row.outcome, context);
      expect(selection.reason, row.reason).toBe(row.reason);
      expect(selection.model, row.reason).toBe(row.model);
      expect(JSON.stringify(selection)).not.toContain(CANARY);
    }
    const own = selectionFromOutcome(
      { type: "no_eligible_candidate" },
      { ...context, fallbackId: null },
    );
    expect(own).toMatchObject({ reason: "no_eligible_candidate", model: null });
  });
});

describe("NaceModelSelector", () => {
  it("case 18: refuses an OpenRouter base URL at construction", () => {
    expect(
      () =>
        new NaceModelSelector({ baseUrl: "https://openrouter.ai/api/v1", apiKey: "sk-or-test" }),
    ).toThrow(/OpenRouter/);
  });

  it("case 20: refuses an empty list, a long id and a bad profile", async () => {
    const fetch = fakeDrex().fetch;
    await expect(selector(fetch).select(turns, { candidates: [] })).rejects.toThrow(/at least one/);
    expect(() =>
      wireCandidates([
        { id: " ", provider: "openai", priceIn: 1, priceOut: 1, context: 8, tags: [] },
      ]),
    ).toThrow(/blank/);
    expect(() => wireCandidates(["x".repeat(257)])).toThrow(/256/);
    expect(() => wireCandidates(Array.from({ length: 256 }, (_, index) => `m${index}`))).toThrow(
      /255/,
    );
    expect(() =>
      wireCandidates([
        { id: "fast", provider: "", priceIn: -1, priceOut: 1, context: 0, tags: ["ok"] },
      ]),
    ).toThrow(/invalid profile/);
    const provider = "👍".repeat(128);
    expect(() =>
      wireCandidates([{ id: "fast", provider, priceIn: 1, priceOut: 1, context: 8, tags: ["ok"] }]),
    ).not.toThrow();
    expect(() =>
      wireCandidates([
        {
          id: "fast",
          provider: `${provider}👍`,
          priceIn: 1,
          priceOut: 1,
          context: 8,
          tags: ["ok"],
        },
      ]),
    ).toThrow(/invalid profile/);
    const extra = {
      id: "fast",
      provider: "openai",
      priceIn: 1,
      priceOut: 1,
      context: 8,
      tags: ["ok"],
      note: "extra",
    };
    expect(() => wireCandidates([extra as never])).toThrow(/invalid profile/);
    const tag = "t".repeat(128);
    const fat = (id: string) => ({
      id,
      provider: "openai",
      priceIn: 1,
      priceOut: 1,
      context: 8,
      tags: Array.from({ length: 64 }, () => tag),
    });
    expect(() =>
      wireCandidates(Array.from({ length: 12 }, (_, index) => fat(`m${index}`))),
    ).not.toThrow();
    expect(() =>
      wireCandidates(Array.from({ length: 13 }, (_, index) => fat(`m${index}`))),
    ).toThrow(/100000/);
    expect(() =>
      wireCandidates([
        {
          id: "fast",
          provider: "openai",
          priceIn: 1,
          priceOut: 1,
          context: 2 ** 53,
          tags: ["ok"],
        },
      ]),
    ).toThrow(/invalid profile/);
    expect(() =>
      wireCandidates([
        {
          id: "fast",
          provider: "\uFEFF",
          priceIn: 1,
          priceOut: 1,
          context: 8,
          tags: ["\uFEFF"],
        },
      ]),
    ).toThrow(/invalid profile/);
    expect(() =>
      wireCandidates([
        {
          id: "fast",
          provider: "openai",
          priceIn: 1,
          priceOut: 1,
          context: Number.MAX_SAFE_INTEGER,
          tags: ["ok"],
        },
      ]),
    ).not.toThrow();
  });

  it("case 9: sends only messages, candidates and selection_priority", async () => {
    const drex = fakeDrex([{ body: routeReply("strong", ["fast", "strong"]) }]);
    const selection = await selector(drex.fetch).select(
      [
        { role: "system", content: "rules" },
        { role: "user", content: "Hello" },
      ],
      { candidates: ["fast", "strong"], priority: "quality" },
    );
    expect(drex.requests).toHaveLength(1);
    const request = drex.requests[0]!;
    expect(request.url).toBe("https://drex.test/v1/route");
    expect(Object.keys(request.body).sort()).toEqual([
      "candidates",
      "messages",
      "selection_priority",
    ]);
    expect(request.body.selection_priority).toBe("quality");
    expect(request.body.candidates).toEqual(["fast", "strong"]);
    for (const message of request.body.messages) {
      expect(["system", "user", "assistant"]).toContain(message.role);
      expect(typeof message.content).toBe("string");
    }
    expect(selection).toMatchObject({ reason: "selected", model: "strong", selected: "strong" });
  });

  it("case 19: sends the attribution headers", async () => {
    const drex = fakeDrex([{ body: routeReply("fast", ["fast", "strong"]) }]);
    await selector(drex.fetch).select(turns, { candidates: ["fast", "strong"] });
    const headers = drex.requests[0]!.headers;
    expect(headers.get("user-agent")).toBe(`langchain-nace-js/${VERSION}`);
    expect(headers.get("x-nace-client")).toBe(`langchain-nace-js/${VERSION}`);
  });

  it("case 2: each HTTP failure uses the fallback", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const rows: [number, string][] = [
      [429, "rate_limited"],
      [500, "unavailable"],
      [503, "unavailable"],
      [504, "unavailable"],
      [529, "unavailable"],
      [401, "auth_error"],
      [403, "auth_error"],
      [402, "no_funds"],
      [413, "invalid_request"],
      [422, "invalid_request"],
    ];
    for (const [status, reason] of rows) {
      const drex = fakeDrex([{ status, body: { error: { message: CANARY } } }]);
      const selection = await selector(drex.fetch).select(turns, {
        candidates: ["fast", "strong"],
        fallback: "fast",
      });
      expect(selection.reason, String(status)).toBe(reason);
      expect(selection.model).toBe("fast");
      expect(JSON.stringify(selection)).not.toContain(CANARY);
      expect(warn.mock.calls.at(-1)?.join(" ")).not.toContain(CANARY);
      expect(drex.requests).toHaveLength(1);
    }
    warn.mockRestore();
  });

  it("case 3 to 6: an invalid body uses the fallback", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const bodies = [
      { selected: "other", probabilities: probs(["fast", "strong"], "fast") },
      { selected: "fast", probabilities: { fast: 1, other: 0 } },
      { selected: "fast", probabilities: { fast: Number.NaN, strong: 0.2 } },
      { selected: "__proto__", probabilities: { __proto__: 1, constructor: 0 } },
      { selected: "constructor", probabilities: { constructor: 1, fast: 0 } },
    ];
    for (const body of bodies) {
      const drex = fakeDrex([{ body }]);
      const selection = await selector(drex.fetch).select(turns, {
        candidates: ["fast", "strong"],
        fallback: "fast",
      });
      expect(selection.reason).toBe("invalid_response");
      expect(selection.model).toBe("fast");
    }
    warn.mockRestore();
  });

  it("case 7: a slow route uses the fallback and ignores the late body", async () => {
    let late = false;
    const fetch = async () => {
      await new Promise((resolve) => setTimeout(resolve, 80));
      late = true;
      return new Response(JSON.stringify(routeReply("strong", ["fast", "strong"])), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    };
    const started = Date.now();
    const selection = await selector(fetch as typeof fetch, { timeout: 20 }).select(turns, {
      candidates: ["fast", "strong"],
      fallback: "fast",
      timeout: 20,
    });
    expect(Date.now() - started).toBeLessThan(70);
    expect(selection).toMatchObject({ reason: "timeout", model: "fast", selected: null });
    await new Promise((resolve) => setTimeout(resolve, 100));
    expect(late).toBe(true);
    expect(selection.selected).toBeNull();
    expect(selection.model).toBe("fast");
  });

  it("case 8: cancellation is raised and the request is closed", async () => {
    const controller = new AbortController();
    let seen: AbortSignal | undefined;
    let started: () => void = () => {};
    const gate = new Promise<void>((resolve) => {
      started = resolve;
    });
    const fetch: typeof globalThis.fetch = async (_input, init) => {
      seen = init?.signal ?? undefined;
      started();
      await new Promise((_resolve, reject) => {
        const timer = setTimeout(() => reject(new Error("hung")), 5_000);
        init?.signal?.addEventListener("abort", () => {
          clearTimeout(timer);
          reject(Object.assign(new Error("aborted"), { name: "AbortError" }));
        });
      });
      return new Response("{}", { status: 200 });
    };
    const pending = selector(fetch).select(turns, {
      candidates: ["fast", "strong"],
      signal: controller.signal,
    });
    await gate;
    controller.abort();
    await expect(pending).rejects.toThrow();
    expect(seen?.aborted).toBe(true);
  });

  it("a connection error uses the fallback and makes one attempt", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    let calls = 0;
    const fetch = async () => {
      calls += 1;
      throw new TypeError("fetch failed");
    };
    const selection = await selector(fetch as typeof fetch).select(turns, {
      candidates: ["fast", "strong"],
      fallback: "fast",
    });
    warn.mockRestore();
    expect(calls).toBe(1);
    expect(selection).toMatchObject({ reason: "connection_error", model: "fast" });
  });

  it("case 17: the canary is absent from the log, the selection and the trace inputs", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const info = vi.spyOn(console, "info").mockImplementation(() => {});
    const log = vi.spyOn(console, "log").mockImplementation(() => {});
    const error = vi.spyOn(console, "error").mockImplementation(() => {});
    const tracer = new Collector();
    const drex = fakeDrex([
      { status: 422, body: { error: { message: CANARY }, request_id: "req_route" } },
    ]);
    const selection = await selector(drex.fetch).select(
      [
        { role: "system", content: CANARY },
        { role: "user", content: `hello ${CANARY}` },
      ],
      { candidates: ["fast", "strong"], fallback: "fast", callbacks: [tracer] },
    );
    const logged = [warn, info, log, error].flatMap((spy) => spy.mock.calls).join(" ");
    expect(logged).not.toContain(CANARY);
    expect(JSON.stringify(selection)).not.toContain(CANARY);
    const run = tracer.runs.find((item) => item.name === "NaceModelSelector");
    expect(run?.inputs).toMatchObject({
      candidates: ["fast", "strong"],
      priority: "balanced",
      messageCount: 2,
    });
    expect(JSON.stringify(run?.inputs)).not.toContain(CANARY);
    expect(run?.outputs).toMatchObject({ reason: "invalid_request", model: "fast" });
    expect(JSON.stringify(run?.outputs)).not.toContain(CANARY);
    warn.mockRestore();
    info.mockRestore();
    log.mockRestore();
    error.mockRestore();
  });

  it("drops a request id that is the API key", async () => {
    const key = "nace_sk_test";
    const drex = fakeDrex([
      {
        body: { selected: "strong", probabilities: probs(["fast", "strong"], "strong") },
        headers: { "x-request-id": key },
      },
    ]);
    const selection = await selector(drex.fetch).select(turns, {
      candidates: ["fast", "strong"],
    });
    expect(selection.requestId).toBeNull();
    expect(JSON.stringify(selection)).not.toContain(key);
    const shaped = "req_nacekey1";
    const shapedDrex = fakeDrex([
      {
        body: { selected: "strong", probabilities: probs(["fast", "strong"], "strong") },
        headers: { "x-request-id": shaped },
      },
    ]);
    const shapedSelection = await new NaceModelSelector(
      offline(shapedDrex.fetch, { apiKey: shaped }),
    ).select(turns, { candidates: ["fast", "strong"] });
    expect(shapedSelection.requestId).toBeNull();
    expect(JSON.stringify(shapedSelection)).not.toContain(shaped);
  });

  it("an error that arrives after the deadline is a timeout", async () => {
    const now = vi.spyOn(Date, "now");
    let tick = 1_000;
    now.mockImplementation(() => tick);
    const fetch: typeof globalThis.fetch = async () => {
      tick = 1_050;
      throw new TypeError("fetch failed");
    };
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const selection = await selector(fetch, { timeout: 30 }).select(turns, {
        candidates: ["fast", "strong"],
        fallback: "fast",
        timeout: 30,
      });
      expect(selection.reason).toBe("timeout");
      expect(warn.mock.calls.at(-1)?.join(" ")).not.toContain("fetch failed");
    } finally {
      warn.mockRestore();
      now.mockRestore();
    }
  });

  it("a thrown non-status error becomes connection_error", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const fetch: typeof globalThis.fetch = async () => {
      throw "boom";
    };
    try {
      const selection = await selector(fetch).select(turns, {
        candidates: ["fast", "strong"],
        fallback: "fast",
      });
      expect(selection.reason).toBe("connection_error");
      expect(warn.mock.calls.join(" ")).not.toContain("boom");
    } finally {
      warn.mockRestore();
    }
  });

  it("leaves no timer and no abort listener after each outcome", async () => {
    vi.useFakeTimers();
    try {
      const outcomes: Array<{ status?: number; hang?: boolean; throw?: boolean }> = [
        {},
        { status: 429 },
        { throw: true },
        { hang: true },
      ];
      for (const outcome of outcomes) {
        const controller = new AbortController();
        const add = vi.spyOn(controller.signal, "addEventListener");
        const remove = vi.spyOn(controller.signal, "removeEventListener");
        const fetch: typeof globalThis.fetch = async (_input, init) => {
          if (outcome.throw) throw new TypeError("fetch failed");
          if (outcome.hang) {
            return new Promise((_resolve, reject) => {
              init?.signal?.addEventListener("abort", () => {
                reject(new DOMException("aborted", "AbortError"));
              });
            });
          }
          return new Response(JSON.stringify(routeReply("fast", ["fast", "strong"])), {
            status: outcome.status ?? 200,
            headers: { "content-type": "application/json", "x-request-id": "req_ok" },
          });
        };
        const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
        const pending = selector(fetch, { timeout: 20 }).select(turns, {
          candidates: ["fast", "strong"],
          fallback: "fast",
          timeout: 20,
          signal: controller.signal,
        });
        if (outcome.hang) await vi.advanceTimersByTimeAsync(20);
        await pending;
        const adds = add.mock.calls.filter((call) => call[0] === "abort").length;
        const removes = remove.mock.calls.filter((call) => call[0] === "abort").length;
        expect(adds).toBeGreaterThan(0);
        expect(removes).toBe(adds);
        expect(vi.getTimerCount()).toBe(0);
        warn.mockRestore();
        add.mockRestore();
        remove.mockRestore();
      }
    } finally {
      vi.useRealTimers();
    }
  });

  it("discards a valid response that finishes after the deadline", async () => {
    const now = vi.spyOn(Date, "now");
    let tick = 1_000;
    now.mockImplementation(() => tick);
    const fetch: typeof globalThis.fetch = async () => {
      tick = 1_050;
      return new Response(JSON.stringify(routeReply("strong", ["fast", "strong"])), {
        status: 200,
        headers: { "content-type": "application/json", "x-request-id": "req_abc12345" },
      });
    };
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const selection = await selector(fetch, { timeout: 30 }).select(turns, {
        candidates: ["fast", "strong"],
        fallback: "fast",
        timeout: 30,
      });
      expect(selection.reason).toBe("timeout");
      expect(selection.model).toBe("fast");
      expect(selection.selected).toBeNull();
    } finally {
      warn.mockRestore();
      now.mockRestore();
    }
  });

  it("rethrows the abort reason when the exchange throws a different error", async () => {
    const reason = new Error("stop-the-run");
    const controller = new AbortController();
    let cleanup = false;
    const remove = vi
      .spyOn(controller.signal, "removeEventListener")
      .mockImplementation(function mockRemove(this: AbortSignal, ...args) {
        if (cleanup) throw new Error("listener cleanup failed");
        return AbortSignal.prototype.removeEventListener.apply(this, args as never);
      });
    const fetch: typeof globalThis.fetch = async () => {
      controller.abort(reason);
      cleanup = true;
      throw new TypeError("sdk cleanup");
    };
    const client = new NaceClient({
      apiKey: "nace_sk_test",
      baseURL: "https://drex.test",
      fetch,
      retry: { maxRetries: 0 },
    });
    try {
      await expect(
        exchangeRoute(
          client,
          {
            messages: [{ role: "user", content: "hi" }],
            candidates: ["fast", "strong"],
            selection_priority: "balanced",
          },
          { signal: controller.signal, timeoutMs: 1_000 },
        ),
      ).rejects.toBe(reason);
    } finally {
      remove.mockRestore();
    }
  });

  it("rethrows the abort reason when the route call throws after the abort", async () => {
    const reason = new Error("stop-the-run");
    const controller = new AbortController();
    const fetch: typeof globalThis.fetch = async () => {
      controller.abort(reason);
      throw new TypeError("sdk cleanup");
    };
    await expect(
      selector(fetch).select(turns, {
        candidates: ["fast", "strong"],
        fallback: "fast",
        signal: controller.signal,
      }),
    ).rejects.toBe(reason);
  });

  it("records the request id on the trace", async () => {
    const tracer = new Collector();
    const drex = fakeDrex([
      {
        body: routeReply("strong", ["fast", "strong"]),
        headers: { "x-request-id": "req_abc12345" },
      },
    ]);
    await selector(drex.fetch).select(turns, {
      candidates: ["fast", "strong"],
      callbacks: [tracer],
    });
    expect(tracer.runs[0]?.extra?.metadata).toMatchObject({ nace_request_id: "req_abc12345" });
  });

  it("does not call the route when one candidate is eligible", async () => {
    const drex = fakeDrex();
    const selection = await selector(drex.fetch).select(turns, { candidates: ["only"] });
    expect(drex.requests).toHaveLength(0);
    expect(selection).toMatchObject({ reason: "single_candidate", model: "only", fallback: true });
  });

  it("does not send when the run is already aborted", async () => {
    const controller = new AbortController();
    controller.abort();
    const drex = fakeDrex();
    await expect(
      selector(drex.fetch).select(turns, {
        candidates: ["fast", "strong"],
        signal: controller.signal,
      }),
    ).rejects.toThrow();
    expect(drex.requests).toHaveLength(0);
  });
});
