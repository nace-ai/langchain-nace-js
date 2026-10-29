/** Regression tests for issues found in the pre-release review. */
import { inspect } from "node:util";

import { AIMessage, HumanMessage, ToolMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { MemorySaver } from "@langchain/langgraph";
import { createAgent } from "langchain";
import { describe, expect, it } from "vitest";
import * as z from "zod/v4";

import {
  NaceClassifier,
  NaceError,
  NaceImageError,
  NaceMediaError,
  NaceRouter,
} from "../../src/index.js";
import { naceAutoModeMiddleware } from "../../src/middleware/index.js";
import { fakeDrex, offline, okBody, PNG_1x1, type Reply, ScriptedChatModel } from "./helpers.js";

const Q = { q: { type: "noul" as const, instructions: "?" } };

function gate(level: number, destructive: number, matches: number) {
  return (body: any): Reply => ({
    body: okBody(body, {
      risk: {
        type: "score",
        score: level,
        legend: ["a", "b", "c", "d", "e"],
        confidence: 0.9,
        probabilities: { "0": 1 },
      },
      destructive: { type: "noul", noul: destructive },
      matches_request: { type: "noul", noul: matches },
    }),
  });
}

describe("secrets", () => {
  it("keeps the API key out of a NaceRouter's printed form", () => {
    const router = new NaceRouter({
      question: { type: "choice", criteria: { a: null } },
      classifierOptions: offline(fakeDrex().fetch, { apiKey: "nace_sk_SECRET123" }),
    });
    expect(inspect(router, { depth: 8 })).not.toContain("nace_sk_SECRET123");
    expect(JSON.stringify(router)).not.toContain("nace_sk_SECRET123");
  });

  it("leaves a remote image URL out of the error message", async () => {
    const error = await new NaceClassifier(offline(fakeDrex().fetch, { questions: Q }))
      .invoke("x", { images: ["https://files.example.com/a.png?token=secret"] })
      .catch((e) => e);
    expect(error).toBeInstanceOf(NaceImageError);
    expect(error.message).not.toContain("secret");
    expect(error.message).not.toContain("example.com");
  });
});

describe("tool gate", () => {
  it("checks a different call that reuses a tool call id", async () => {
    const drex = fakeDrex([gate(0, 0.01, 0.99), gate(4, 0.99, 0.0)]);
    const ran: string[] = [];
    const list = tool(
      async () => {
        ran.push("list");
        return "a.txt";
      },
      { name: "list_files", description: "List files.", schema: z.object({}) },
    );
    const wipe = tool(
      async () => {
        ran.push("wipe");
        return "gone";
      },
      {
        name: "delete_all_files",
        description: "Delete every file.",
        schema: z.object({ force: z.boolean() }),
      },
    );
    const agent = createAgent({
      model: new ScriptedChatModel([
        new AIMessage({
          content: "",
          tool_calls: [{ id: "call_0", name: "list_files", args: {} }],
        }),
        new AIMessage({
          content: "",
          tool_calls: [{ id: "call_0", name: "delete_all_files", args: { force: true } }],
        }),
        new AIMessage("done"),
      ]),
      tools: [list, wipe],
      middleware: [naceAutoModeMiddleware({ classifierOptions: offline(drex.fetch) })],
      checkpointer: new MemorySaver(),
    });
    await agent.invoke(
      { messages: [new HumanMessage("List my files.")] },
      { configurable: { thread_id: "reused-id" } },
    );
    expect(drex.requests).toHaveLength(2);
    expect(ran).toEqual(["list"]);
  });

  it("checks without the screenshot when the package can't send it", async () => {
    const drex = fakeDrex([gate(0, 0.01, 0.99)]);
    const ran: string[] = [];
    const read = tool(
      async () => {
        ran.push("read");
        return "ok";
      },
      { name: "read_ticket", description: "Read a ticket.", schema: z.object({}) },
    );
    const agent = createAgent({
      model: new ScriptedChatModel([
        new AIMessage({ content: "", tool_calls: [{ id: "c1", name: "read_ticket", args: {} }] }),
        new AIMessage("done"),
      ]),
      tools: [read],
      middleware: [
        naceAutoModeMiddleware({
          classifierOptions: offline(drex.fetch),
          // GIF bytes: not a format Drex Max reads, so the package refuses them.
          screenshots: () => [new Uint8Array([0x47, 0x49, 0x46, 0x38, 0x39, 0x61])],
        }),
      ],
    });
    await agent.invoke({ messages: [new HumanMessage("Read the ticket")] });
    expect(drex.requests).toHaveLength(1);
    expect(drex.requests[0]!.body.images).toBeUndefined();
    expect(ran).toEqual(["read"]);
  });
});

describe("cancellation", () => {
  it("aborts the HTTP request when the call times out", async () => {
    let seen: AbortSignal | undefined;
    const slow = (async (_input: string, init?: RequestInit) => {
      seen = init?.signal ?? undefined;
      await new Promise((resolve) => setTimeout(resolve, 300));
      return new Response("{}", { status: 200 });
    }) as unknown as typeof fetch;
    const classifier = new NaceClassifier(offline(slow, { questions: Q, maxRetries: 0 }));
    await expect(classifier.invoke("x", { timeout: 30 })).rejects.toThrow();
    expect(seen?.aborted).toBe(true);
  });
});

describe("images in provider formats", () => {
  const data = `data:image/png;base64,${PNG_1x1}`;
  it.each([
    ["OpenAI Responses input_image", { type: "input_image", image_url: data }],
    [
      "Anthropic source",
      { type: "image", source: { type: "base64", media_type: "image/png", data: PNG_1x1 } },
    ],
    ["Google media", { type: "media", mimeType: "image/png", data: PNG_1x1 }],
  ])("lifts %s blocks into images", async (_name, block) => {
    const drex = fakeDrex();
    await new NaceClassifier(offline(drex.fetch, { questions: Q })).invoke([
      new HumanMessage({ content: [{ type: "text", text: "see" }, block as never] }),
    ]);
    const body = drex.requests[0]!.body;
    expect(body.images).toEqual([data]);
    expect(JSON.stringify(body.state)).not.toContain(PNG_1x1);
  });

  it("refuses non-image media blocks rather than sending them as text", async () => {
    await expect(
      new NaceClassifier(offline(fakeDrex().fetch, { questions: Q })).invoke([
        new HumanMessage({
          content: [{ type: "media", mimeType: "audio/wav", data: "AAAA" } as never],
        }),
      ]),
    ).rejects.toThrow(NaceMediaError);
  });
});

describe("keys and question validation", () => {
  it("keeps __proto__ answer ids, choice labels and message keys", async () => {
    const drex = fakeDrex([
      (body) => ({
        body: okBody(
          body,
          JSON.parse(
            '{"__proto__":{"type":"noul","noul":0.4},"c":{"type":"choice","choice":"__proto__","confidence":0.9,"probabilities":{"__proto__":0.9,"b":0.1}}}',
          ),
        ),
      }),
    ]);
    const r = await new NaceClassifier(
      offline(drex.fetch, {
        questions: JSON.parse(
          '{"__proto__":{"type":"noul","instructions":"?"},"c":{"type":"choice","criteria":{"__proto__":null,"b":null}}}',
        ),
      }),
    ).invoke([
      JSON.parse('{"role":"user","content":[{"type":"text","text":"hi"}],"__proto__":"kept"}'),
    ]);
    expect(Object.keys(r.answers)).toEqual(["__proto__", "c"]);
    expect(Object.keys(r.choices.c!.probabilities)).toEqual(["__proto__", "b"]);
    expect(Object.hasOwn(drex.requests[0]!.body.state[0], "__proto__")).toBe(true);
  });

  it("sends a question type named like an Object method as a raw question", async () => {
    const drex = fakeDrex();
    await new NaceClassifier(
      offline(drex.fetch, { questions: { q: { type: "toString", items: [1] } } }),
    ).invoke("x");
    expect(drex.requests[0]!.body.questions.q.type).toBe("toString");
  });

  it("refuses noul criteria keys other than true and false", () => {
    expect(
      () =>
        new NaceClassifier(
          offline(fakeDrex().fetch, {
            questions: {
              q: { type: "noul", instructions: "?", criteria: { True: "yes" } as never },
            },
          }),
        ),
    ).toThrow(NaceError);
  });
});

it("still answers through ToolMessage content that holds an image", async () => {
  const drex = fakeDrex();
  await new NaceClassifier(offline(drex.fetch, { questions: Q })).invoke([
    new ToolMessage({
      content: [{ type: "image", data: PNG_1x1, mimeType: "image/png" }],
      tool_call_id: "t1",
    }),
  ]);
  expect(drex.requests[0]!.body.images).toHaveLength(1);
});
