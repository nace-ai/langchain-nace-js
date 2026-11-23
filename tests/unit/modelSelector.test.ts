import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import {
  AIMessage,
  AIMessageChunk,
  type BaseMessage,
  HumanMessage,
  ToolMessage,
} from "@langchain/core/messages";
import { ChatGenerationChunk, type ChatResult } from "@langchain/core/outputs";
import { tool } from "@langchain/core/tools";
import { Command, interrupt, MemorySaver } from "@langchain/langgraph";
import { createAgent, createMiddleware, providerStrategy } from "langchain";
import { describe, expect, it, vi } from "vitest";
import * as z from "zod/v4";

import { NaceModelSelector } from "../../src/index.js";
import { naceModelSelectorMiddleware } from "../../src/middleware/index.js";
import { selectionAnchor } from "../../src/selection.js";
import { fakeDrex, offline, ScriptedChatModel } from "./helpers.js";

const CANARY = "CANARY-middleware-9f3c";

class TokenModel extends BaseChatModel {
  constructor(private readonly tokens: string[]) {
    super({});
  }

  _llmType(): string {
    return "tokens";
  }

  bindTools(): this {
    return this;
  }

  async *_streamResponseChunks(
    _messages: BaseMessage[],
    _options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun,
  ): AsyncGenerator<ChatGenerationChunk> {
    let completion = 0;
    for (const token of this.tokens) {
      const chunk = new ChatGenerationChunk({
        message: new AIMessageChunk({ content: token }),
        text: token,
      });
      await runManager?.handleLLMNewToken(
        token,
        { prompt: 0, completion },
        undefined,
        undefined,
        undefined,
        { chunk },
      );
      completion += 1;
      yield chunk;
    }
  }

  async _generate(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun,
  ): Promise<ChatResult> {
    let text = "";
    for await (const chunk of this._streamResponseChunks(messages, options, runManager)) {
      text += chunk.text;
    }
    return { generations: [{ message: new AIMessageChunk({ content: text }), text }] };
  }
}

function probs(ids: string[], selected: string, top = 0.8): Record<string, number> {
  const rest = ids.length === 1 ? 0 : (1 - top) / (ids.length - 1);
  return Object.fromEntries(ids.map((id) => [id, id === selected ? top : rest]));
}

function routeReply(selected: string, ids: string[], top = 0.8) {
  return { selected, probabilities: probs(ids, selected, top), request_id: "req_route" };
}

function pair(content: string) {
  return new ScriptedChatModel([new AIMessage(content)]);
}

/** A chat model that answers a response schema, either as a tool call or as JSON text. */
class StructuredReply extends BaseChatModel {
  private toolName: string | undefined;

  constructor(
    private readonly answer: string,
    private readonly mode: "tool" | "json",
  ) {
    super({});
  }

  _llmType(): string {
    return "structured-reply";
  }

  bindTools(tools: { name?: string; function?: { name?: string } }[]): this {
    const first = tools[0];
    this.toolName = first?.name ?? first?.function?.name;
    return this;
  }

  async _generate(): Promise<ChatResult> {
    if (this.mode === "json") {
      const message = new AIMessage(JSON.stringify({ answer: this.answer }));
      return { generations: [{ message, text: message.text }] };
    }
    const message = new AIMessage({
      content: "",
      tool_calls: [
        {
          id: "call_1",
          name: this.toolName ?? "extract",
          args: { answer: this.answer },
          type: "tool_call",
        },
      ],
    });
    return { generations: [{ message, text: "" }] };
  }
}

function agentOf(options: {
  script?: { status?: number; body?: unknown; delayMs?: number }[];
  fetch?: typeof globalThis.fetch;
  fallback?: string | null;
  minProbability?: number;
  selectOn?: "run" | "model_call";
  timeout?: number;
  tools?: readonly unknown[];
  systemPrompt?: string;
  checkpointer?: MemorySaver;
  fast?: BaseChatModel;
  strong?: BaseChatModel;
  base?: BaseChatModel;
  fastFlags?: { images?: boolean; tools?: boolean };
  strongFlags?: { images?: boolean; tools?: boolean };
  extra?: Record<string, { model: BaseChatModel; images?: boolean; tools?: boolean }>;
  middleware?: ReturnType<typeof createMiddleware>[];
  responseFormat?: unknown;
}) {
  const drex = fakeDrex(options.script ?? []);
  const base = options.base ?? pair("base");
  const fast = options.fast ?? pair("fast");
  const strong = options.strong ?? pair("strong");
  const agent = createAgent({
    model: base,
    tools: (options.tools ?? []) as never,
    systemPrompt: options.systemPrompt,
    checkpointer: options.checkpointer,
    ...(options.responseFormat !== undefined ? { responseFormat: options.responseFormat } : {}),
    middleware: [
      naceModelSelectorMiddleware({
        candidates: {
          fast: { model: fast, ...options.fastFlags },
          strong: { model: strong, ...options.strongFlags },
          ...options.extra,
        },
        fallback: options.fallback === undefined ? "fast" : options.fallback,
        minProbability: options.minProbability,
        selectOn: options.selectOn,
        timeout: options.timeout,
        selector: new NaceModelSelector(
          offline(options.fetch ?? drex.fetch, { timeout: options.timeout ?? 3000 }),
        ),
      }),
      ...(options.middleware ?? []),
    ],
  });
  return { agent, drex, base, fast, strong };
}

function aiContents(result: { messages: unknown[] }): string[] {
  return result.messages
    .filter((message): message is AIMessage => AIMessage.isInstance(message))
    .map((message) => String(message.content));
}

describe("naceModelSelectorMiddleware", () => {
  it("case 1: a valid response selects the bound model and the reply reaches state", async () => {
    const { agent, drex } = agentOf({
      script: [{ body: routeReply("strong", ["fast", "strong"]) }],
    });
    const result = await agent.invoke({ messages: [new HumanMessage("hello")] });
    expect(aiContents(result)).toEqual(["strong"]);
    expect(result.modelSelection).toMatchObject({
      reason: "selected",
      model: "strong",
      selected: "strong",
      fallback: false,
    });
    expect((result as { modelRoute?: unknown }).modelRoute).toBeUndefined();
    expect(drex.requests).toHaveLength(1);
  });

  it("case 2: each failure reason runs the fallback model", async () => {
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
      const { agent } = agentOf({ script: [{ status, body: { error: { message: CANARY } } }] });
      const result = await agent.invoke({ messages: [new HumanMessage("hello")] });
      expect(aiContents(result), reason).toEqual(["fast"]);
      expect(result.modelSelection).toMatchObject({ reason, model: "fast" });
      expect(JSON.stringify(result.modelSelection)).not.toContain(CANARY);
    }
    warn.mockRestore();
  });

  it("case 2: low probability, one candidate, no input and no eligible candidate", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const low = agentOf({
      script: [{ body: routeReply("strong", ["fast", "strong"], 0.2) }],
      minProbability: 0.5,
    });
    const lowResult = await low.agent.invoke({ messages: [new HumanMessage("hello")] });
    expect(lowResult.modelSelection).toMatchObject({ reason: "low_probability", model: "fast" });
    expect(aiContents(lowResult)).toEqual(["fast"]);

    const image = new HumanMessage({
      content: [
        { type: "text", text: "see" },
        { type: "image_url", image_url: { url: "data:image/png;base64,aa" } },
      ],
    });
    const single = agentOf({ fastFlags: { images: false } });
    const singleResult = await single.agent.invoke({ messages: [image] });
    expect(single.drex.requests).toHaveLength(0);
    expect(singleResult.modelSelection).toMatchObject({
      reason: "single_candidate",
      model: "strong",
    });
    expect(aiContents(singleResult)).toEqual(["strong"]);

    const none = agentOf({ fastFlags: { images: false }, strongFlags: { images: false } });
    const noneResult = await none.agent.invoke({ messages: [image] });
    expect(none.drex.requests).toHaveLength(0);
    expect(noneResult.modelSelection).toMatchObject({
      reason: "no_eligible_candidate",
      model: null,
    });
    expect(aiContents(noneResult)).toEqual(["base"]);

    const empty = agentOf({});
    const emptyResult = await empty.agent.invoke({ messages: [new HumanMessage("")] });
    expect(empty.drex.requests).toHaveLength(0);
    expect(emptyResult.modelSelection).toMatchObject({ reason: "no_input", model: null });
    expect(aiContents(emptyResult)).toEqual(["base"]);
    warn.mockRestore();
  });

  it("case 7: a slow route runs the fallback and ignores the late body", async () => {
    let late = false;
    const fetch = async () => {
      await new Promise((resolve) => setTimeout(resolve, 150));
      late = true;
      return new Response(JSON.stringify(routeReply("strong", ["fast", "strong"])), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    };
    const { agent } = agentOf({ fetch: fetch as typeof fetch, timeout: 30 });
    const started = Date.now();
    const result = await agent.invoke({ messages: [new HumanMessage("hello")] });
    expect(Date.now() - started).toBeLessThan(120);
    expect(aiContents(result)).toEqual(["fast"]);
    expect(result.modelSelection).toMatchObject({
      reason: "timeout",
      model: "fast",
      selected: null,
    });
    await new Promise((resolve) => setTimeout(resolve, 180));
    expect(late).toBe(true);
    expect(result.modelSelection?.selected).toBeNull();
    expect(aiContents(result)).toEqual(["fast"]);
  });

  it("case 8: cancellation does not call the handler and closes the request", async () => {
    const controller = new AbortController();
    let seen: AbortSignal | undefined;
    let open: () => void = () => {};
    const gate = new Promise<void>((resolve) => {
      open = resolve;
    });
    let calls = 0;
    const counter = createMiddleware({
      name: "CountCalls",
      wrapModelCall: async (request, handler) => {
        calls += 1;
        return handler(request);
      },
    });
    const fetch: typeof globalThis.fetch = async (_input, init) => {
      seen = init?.signal ?? undefined;
      open();
      await new Promise((_resolve, reject) => {
        const timer = setTimeout(() => reject(new Error("hung")), 5_000);
        init?.signal?.addEventListener("abort", () => {
          clearTimeout(timer);
          reject(Object.assign(new Error("aborted"), { name: "AbortError" }));
        });
      });
      return new Response("{}", { status: 200 });
    };
    const { agent } = agentOf({
      fetch,
      middleware: [counter],
    });
    const pending = agent.invoke(
      { messages: [new HumanMessage("hello")] },
      { signal: controller.signal },
    );
    await gate;
    controller.abort();
    await expect(pending).rejects.toThrow();
    expect(calls).toBe(0);
    expect(seen?.aborted).toBe(true);
  });

  it("case 11: the handler sees the same request with only the model replaced", async () => {
    const seen: {
      outer?: { model: unknown; messages: unknown[]; tools: unknown; systemMessage: unknown };
      inner?: { model: unknown; messages: unknown[]; tools: unknown; systemMessage: unknown };
    } = {};
    const watch = (slot: "outer" | "inner") =>
      createMiddleware({
        name: slot === "outer" ? "WatchOuter" : "WatchInner",
        wrapModelCall: async (request, handler) => {
          seen[slot] = {
            model: request.model,
            messages: request.messages,
            tools: request.tools,
            systemMessage: request.systemMessage,
          };
          return handler(request);
        },
      });
    const lookup = tool(async () => "found", {
      name: "lookup",
      description: "Look up a record.",
      schema: z.object({}),
    });
    const selected = agentOf({
      script: [{ body: routeReply("strong", ["fast", "strong"]) }],
      tools: [lookup],
      systemPrompt: "be brief",
      middleware: [watch("inner")],
    });
    const outerMw = watch("outer");
    const agent = createAgent({
      model: selected.base,
      tools: [lookup],
      systemPrompt: "be brief",
      middleware: [
        outerMw,
        naceModelSelectorMiddleware({
          candidates: { fast: { model: selected.fast }, strong: { model: selected.strong } },
          fallback: "fast",
          selector: new NaceModelSelector(offline(selected.drex.fetch)),
        }),
        watch("inner"),
      ],
    });
    await agent.invoke({ messages: [new HumanMessage("hello")] });
    expect(seen.inner?.model).toBe(selected.strong);
    expect(seen.inner?.model).not.toBe(seen.outer?.model);
    expect(seen.inner?.messages).toBe(seen.outer?.messages);
    expect(seen.inner?.messages[0]).toBe(seen.outer?.messages[0]);
    expect(seen.inner?.tools).toBe(seen.outer?.tools);
    expect(seen.inner?.systemMessage).toBe(seen.outer?.systemMessage);

    seen.outer = undefined;
    seen.inner = undefined;
    const kept = createAgent({
      model: selected.base,
      middleware: [
        watch("outer"),
        naceModelSelectorMiddleware({
          candidates: { fast: { model: selected.fast }, strong: { model: selected.strong } },
          fallback: "fast",
          selector: new NaceModelSelector(offline(fakeDrex().fetch)),
        }),
        watch("inner"),
      ],
    });
    await kept.invoke({ messages: [new HumanMessage("")] });
    const keptSeen = seen as {
      outer?: { model: unknown; messages: unknown[]; tools: unknown; systemMessage: unknown };
      inner?: { model: unknown; messages: unknown[]; tools: unknown; systemMessage: unknown };
    };
    expect(keptSeen.inner?.model).toBe(keptSeen.outer?.model);
    expect(keptSeen.inner?.messages).toBe(keptSeen.outer?.messages);
    expect(keptSeen.inner?.messages[0]).toBe(keptSeen.outer?.messages[0]);
    expect(keptSeen.inner?.tools).toBe(keptSeen.outer?.tools);
    expect(keptSeen.inner?.systemMessage).toBe(keptSeen.outer?.systemMessage);
  });

  it("case 12: tokens from the selected model arrive in order", async () => {
    const strong = new TokenModel(["one", "two", "three"]);
    const { agent } = agentOf({
      script: [{ body: routeReply("strong", ["fast", "strong"]) }],
      strong,
    });
    const contents: string[] = [];
    const stream = await agent.stream(
      { messages: [new HumanMessage("hello")] },
      { streamMode: "messages" },
    );
    for await (const event of stream) {
      const message = Array.isArray(event) ? event[0] : event;
      if (message && typeof message === "object" && "content" in message) {
        const content = (message as { content: unknown }).content;
        if (typeof content === "string" && content !== "") contents.push(content);
      }
    }
    const wanted = ["one", "two", "three"];
    let index = 0;
    for (const content of contents) {
      if (content === wanted[index]) index += 1;
    }
    expect(contents.slice(0, wanted.length)).toEqual(wanted);
    expect(index).toBe(wanted.length);
  });

  it("case 13: candidates that refuse images or tools are not sent", async () => {
    const plain = pair("plain");
    const { agent, drex } = agentOf({
      script: [{ body: routeReply("strong", ["fast", "strong"]) }],
      tools: [
        tool(async () => "found", {
          name: "lookup",
          description: "Look up a record.",
          schema: z.object({}),
        }),
      ],
      extra: {
        blind: { model: plain, images: false },
        quiet: { model: plain, tools: false },
      },
    });
    await agent.invoke({
      messages: [
        new HumanMessage({
          content: [
            { type: "text", text: "see" },
            { type: "image_url", image_url: { url: "data:image/png;base64,aa" } },
          ],
        }),
      ],
    });
    expect(drex.requests[0]?.body.candidates).toEqual(["fast", "strong"]);
  });

  it("case 10: an image, a tool call and a tool result stay out of the route body", async () => {
    const lookup = tool(async () => CANARY, {
      name: "lookup",
      description: "Look up a record.",
      schema: z.object({}),
    });
    const { agent, drex } = agentOf({
      script: [{ body: routeReply("strong", ["fast", "strong"]) }],
      tools: [lookup],
      systemPrompt: "be brief",
      strong: new ScriptedChatModel([new AIMessage("done")]),
    });
    await agent.invoke({
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
        new HumanMessage("What next?"),
      ],
    });
    const body = JSON.stringify(drex.requests[0]?.body);
    expect(body).toContain("[image omitted]");
    expect(body).toContain("checking");
    expect(body).toContain("be brief");
    expect(body).not.toContain(CANARY);
    expect(body).not.toContain("tool_call");
  });

  it("case 14: one run routes once, and a resumed run does not route again", async () => {
    let first = true;
    const lookup = tool(
      async () => {
        if (first) {
          first = false;
          interrupt({ pause: true });
        }
        return "found";
      },
      { name: "lookup", description: "Look up a record.", schema: z.object({}) },
    );
    let calls = 0;
    const counter = createMiddleware({
      name: "CountCalls",
      wrapModelCall: async (request, handler) => {
        calls += 1;
        return handler(request);
      },
    });
    const strong = new ScriptedChatModel([
      new AIMessageChunk({ content: "", tool_calls: [{ id: "c1", name: "lookup", args: {} }] }),
      new AIMessageChunk({ content: "", tool_calls: [{ id: "c2", name: "lookup", args: {} }] }),
      new AIMessage("done"),
    ]);
    const { agent, drex } = agentOf({
      script: [{ body: routeReply("strong", ["fast", "strong"]) }],
      tools: [lookup],
      checkpointer: new MemorySaver(),
      strong,
      middleware: [counter],
    });
    const thread = { configurable: { thread_id: "resume" } };
    const paused = await agent.invoke({ messages: [new HumanMessage("go")] }, thread);
    expect(drex.requests).toHaveLength(1);
    expect((paused as { __interrupt__?: unknown[] }).__interrupt__?.length).toBeGreaterThan(0);
    const finished = await agent.invoke(new Command({ resume: true }), thread);
    expect(drex.requests).toHaveLength(1);
    expect(aiContents(finished).at(-1)).toBe("done");
    expect(calls).toBe(3);
  });

  it("case 14: three model calls in one run make one route call", async () => {
    const lookup = tool(async () => "found", {
      name: "lookup",
      description: "Look up a record.",
      schema: z.object({}),
    });
    const strong = new ScriptedChatModel([
      new AIMessageChunk({ content: "", tool_calls: [{ id: "c1", name: "lookup", args: {} }] }),
      new AIMessageChunk({ content: "", tool_calls: [{ id: "c2", name: "lookup", args: {} }] }),
      new AIMessage("done"),
    ]);
    const { agent, drex } = agentOf({
      script: [{ body: routeReply("strong", ["fast", "strong"]) }],
      tools: [lookup],
      strong,
    });
    const result = await agent.invoke({ messages: [new HumanMessage("go")] });
    expect(drex.requests).toHaveLength(1);
    expect(aiContents(result).at(-1)).toBe("done");
    expect(result.messages.filter((message) => ToolMessage.isInstance(message))).toHaveLength(2);
  });

  it("case 15: model_call routes on every model call", async () => {
    const lookup = tool(async () => "found", {
      name: "lookup",
      description: "Look up a record.",
      schema: z.object({}),
    });
    const strong = new ScriptedChatModel([
      new AIMessageChunk({ content: "", tool_calls: [{ id: "c1", name: "lookup", args: {} }] }),
      new AIMessageChunk({ content: "", tool_calls: [{ id: "c2", name: "lookup", args: {} }] }),
      new AIMessage("done"),
    ]);
    const { agent, drex } = agentOf({
      script: [
        { body: routeReply("strong", ["fast", "strong"]) },
        { body: routeReply("strong", ["fast", "strong"]) },
        { body: routeReply("strong", ["fast", "strong"]) },
      ],
      tools: [lookup],
      strong,
      selectOn: "model_call",
    });
    await agent.invoke({ messages: [new HumanMessage("go")] });
    expect(drex.requests).toHaveLength(3);
  });

  it("case 16: a second run on the same thread selects again", async () => {
    const { agent, drex } = agentOf({
      script: [
        { body: routeReply("strong", ["fast", "strong"]) },
        { body: routeReply("fast", ["fast", "strong"]) },
      ],
      checkpointer: new MemorySaver(),
    });
    const thread = { configurable: { thread_id: "again" } };
    const first = await agent.invoke({ messages: [new HumanMessage("one")] }, thread);
    const second = await agent.invoke({ messages: [new HumanMessage("two")] }, thread);
    expect(first.modelSelection).toMatchObject({ model: "strong", reason: "selected" });
    expect(second.modelSelection).toMatchObject({ model: "fast", reason: "selected" });
    expect(drex.requests).toHaveLength(2);
    expect(aiContents(second).at(-1)).toBe("fast");
  });

  it("selects again when a tool result makes the stored model ineligible", async () => {
    const lookup = tool(
      async () =>
        new ToolMessage({
          content: [{ type: "image_url", image_url: { url: "data:image/png;base64,aa" } }],
          tool_call_id: "c1",
          name: "lookup",
        }),
      { name: "lookup", description: "Looks up an image.", schema: z.object({}) },
    );
    const blind = new ScriptedChatModel([
      new AIMessage({
        content: "",
        tool_calls: [{ id: "c1", name: "lookup", args: {}, type: "tool_call" }],
      }),
      new AIMessage("from-blind"),
    ]);
    const sighted = new ScriptedChatModel([new AIMessage("from-sighted")]);
    const { agent, drex } = agentOf({
      script: [{ body: routeReply("fast", ["fast", "strong"]) }],
      fast: blind,
      strong: sighted,
      tools: [lookup],
      fastFlags: { images: false },
    });
    const result = await agent.invoke({ messages: [new HumanMessage("describe")] });
    expect(aiContents(result)).toContain("from-sighted");
    expect(aiContents(result)).not.toContain("from-blind");
    expect(drex.requests).toHaveLength(1);
  });

  it("selects again when the same message id has different text", async () => {
    const { agent, drex } = agentOf({
      script: [
        { body: routeReply("strong", ["fast", "strong"]) },
        { body: routeReply("fast", ["fast", "strong"]) },
      ],
      checkpointer: new MemorySaver(),
    });
    const thread = { configurable: { thread_id: "edited" } };
    const first = await agent.invoke(
      { messages: [new HumanMessage({ content: "hello", id: "human_1" })] },
      thread,
    );
    const again = await agent.invoke(
      { messages: [new HumanMessage({ content: "hello", id: "human_1" })] },
      thread,
    );
    expect(drex.requests).toHaveLength(1);
    expect(again.modelSelection).toMatchObject({ model: "strong" });
    const edited = await agent.invoke(
      { messages: [new HumanMessage({ content: "edited", id: "human_1" })] },
      thread,
    );
    expect(drex.requests).toHaveLength(2);
    expect(edited.modelSelection).toMatchObject({ model: "fast" });
    expect(edited.modelSelection?.anchor).not.toBe(first.modelSelection?.anchor);
    expect(JSON.stringify(edited.modelSelection)).not.toContain("edited");
  });

  it("stores an anchor with no message text and selects again for a new human message", async () => {
    const { agent, drex } = agentOf({
      script: [
        { body: routeReply("strong", ["fast", "strong"]) },
        { body: routeReply("fast", ["fast", "strong"]) },
      ],
      checkpointer: new MemorySaver(),
    });
    const thread = { configurable: { thread_id: "anchor" } };
    const first = await agent.invoke(
      { messages: [new HumanMessage({ content: CANARY, id: "human_1" })] },
      thread,
    );
    expect(first.modelSelection).toMatchObject({
      anchor: selectionAnchor([new HumanMessage({ content: CANARY, id: "human_1" })]),
      model: "strong",
    });
    expect(JSON.stringify(first.modelSelection)).not.toContain(CANARY);
    const second = await agent.invoke(
      { messages: [new HumanMessage({ content: "next", id: "human_2" })] },
      thread,
    );
    expect(second.modelSelection).toMatchObject({
      anchor: selectionAnchor([new HumanMessage({ content: "next", id: "human_2" })]),
      model: "fast",
    });
    expect(drex.requests).toHaveLength(2);
  });

  it("does not share a selection between concurrent threads", async () => {
    let calls = 0;
    const fetch: typeof globalThis.fetch = async (_input, init) => {
      calls += 1;
      const selected = String(init?.body).includes("alpha") ? "strong" : "fast";
      await new Promise((resolve) => setTimeout(resolve, 15));
      return new Response(JSON.stringify(routeReply(selected, ["fast", "strong"])), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    };
    const { agent } = agentOf({ fetch, checkpointer: new MemorySaver() });
    const [alpha, beta] = await Promise.all([
      agent.invoke(
        { messages: [new HumanMessage("alpha")] },
        { configurable: { thread_id: "alpha" } },
      ),
      agent.invoke(
        { messages: [new HumanMessage("beta")] },
        { configurable: { thread_id: "beta" } },
      ),
    ]);
    expect(aiContents(alpha).at(-1)).toBe("strong");
    expect(aiContents(beta).at(-1)).toBe("fast");
    expect(calls).toBe(2);
    const { agent: sequential, drex } = agentOf({
      script: [
        { body: routeReply("strong", ["fast", "strong"]) },
        { body: routeReply("fast", ["fast", "strong"]) },
      ],
      checkpointer: new MemorySaver(),
    });
    const one = await sequential.invoke(
      { messages: [new HumanMessage("one")] },
      { configurable: { thread_id: "one" } },
    );
    const two = await sequential.invoke(
      { messages: [new HumanMessage("two")] },
      { configurable: { thread_id: "two" } },
    );
    expect(aiContents(one).at(-1)).toBe("strong");
    expect(aiContents(two).at(-1)).toBe("fast");
    expect(drex.requests).toHaveLength(2);
  });

  it("propagates a handler error and calls the handler once", async () => {
    let calls = 0;
    const inner = createMiddleware({
      name: "Boom",
      wrapModelCall: async () => {
        calls += 1;
        throw new Error("boom");
      },
    });
    const { agent } = agentOf({ middleware: [inner] });
    await expect(agent.invoke({ messages: [new HumanMessage("hello")] })).rejects.toThrow(/boom/);
    expect(calls).toBe(1);
  });

  it("propagates an abort that lands in the same tick as the deadline", async () => {
    vi.useFakeTimers();
    let calls = 0;
    const counter = createMiddleware({
      name: "Counter",
      wrapModelCall: async (request, handler) => {
        calls += 1;
        return handler(request);
      },
    });
    const controller = new AbortController();
    let opened: () => void = () => {};
    const gate = new Promise<void>((resolve) => {
      opened = resolve;
    });
    const fetch: typeof globalThis.fetch = async (_input, init) => {
      opened();
      return new Promise((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          reject(new DOMException("aborted", "AbortError"));
        });
      });
    };
    const { agent } = agentOf({ fetch, timeout: 30, middleware: [counter] });
    const pending = agent
      .invoke({ messages: [new HumanMessage("hello")] }, { signal: controller.signal })
      .then(
        () => ({ failed: false as const }),
        (error: unknown) => ({ failed: true as const, error }),
      );
    try {
      await gate;
      controller.abort();
      await vi.advanceTimersByTimeAsync(30);
      const outcome = await pending;
      expect(outcome.failed).toBe(true);
      if (outcome.failed) expect(String(outcome.error)).toMatch(/abort/i);
      expect(calls).toBe(0);
      expect(controller.signal.aborted).toBe(true);
    } finally {
      vi.useRealTimers();
    }
  });

  it("sends no route request when the latest human turn is only an image", async () => {
    const { agent, drex } = agentOf({
      script: [{ body: routeReply("strong", ["fast", "strong"]) }],
    });
    const result = await agent.invoke({
      messages: [
        new HumanMessage("earlier text"),
        new HumanMessage({
          content: [{ type: "image_url", image_url: { url: "data:image/png;base64,aa" } }],
        }),
      ],
    });
    expect(drex.requests).toHaveLength(0);
    expect(result.modelSelection).toMatchObject({ reason: "no_input", model: null });
  });

  it("accepts a route response for the candidates that were sent", async () => {
    const { agent, drex } = agentOf({
      script: [{ body: routeReply("strong", ["fast", "strong"]) }],
      extra: { blind: { model: pair("blind"), images: false } },
      fallback: "fast",
    });
    const result = await agent.invoke({
      messages: [
        new HumanMessage({
          content: [
            { type: "text", text: "look" },
            { type: "image_url", image_url: { url: "data:image/png;base64,aa" } },
          ],
        }),
      ],
    });
    expect(drex.requests[0]?.body.candidates).toEqual(["fast", "strong"]);
    expect(result.modelSelection).toMatchObject({ model: "strong", reason: "selected" });
    expect(aiContents(result)).toEqual(["strong"]);
  });

  it("does not use a tools-false fallback for structured output after a 429", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const { agent, drex } = agentOf({
      script: [{ status: 429, body: { error: { message: "slow down" } } }],
      extra: { plain: { model: pair('{"answer":"from-plain"}'), tools: false } },
      fallback: "plain",
      base: new StructuredReply("from-base", "tool"),
      fast: pair('{"answer":"from-fast"}'),
      strong: pair('{"answer":"from-strong"}'),
      responseFormat: z.object({ answer: z.string() }),
    });
    try {
      const result = await agent.invoke({ messages: [new HumanMessage("hello")] });
      expect(drex.requests).toHaveLength(1);
      expect(result.modelSelection).toMatchObject({ reason: "rate_limited", model: null });
      expect((result as { structuredResponse?: unknown }).structuredResponse).toEqual({
        answer: "from-base",
      });
      expect(result.messages.filter((message) => AIMessage.isInstance(message))).toHaveLength(2);
      expect(JSON.stringify(result)).not.toContain("from-plain");
    } finally {
      warn.mockRestore();
    }
  });

  it("refuses a profile id that differs from the candidate key", () => {
    const profile = {
      id: "other",
      provider: "openai",
      priceIn: 1,
      priceOut: 1,
      context: 8,
      tags: ["chat"],
    };
    expect(() =>
      naceModelSelectorMiddleware({
        candidates: {
          fast: {
            model: pair("fast"),
            profile: profile as unknown as {
              provider: string;
              priceIn: number;
              priceOut: number;
              context: number;
              tags: string[];
            },
          },
        },
      }),
    ).toThrow(/profile id/);
  });

  it("does not call the handler when the run aborts after the model is initialized", async () => {
    const reason = new Error("stop-before-handler");
    const controller = new AbortController();
    let calls = 0;
    const counter = createMiddleware({
      name: "Counter",
      wrapModelCall: async (request, handler) => {
        calls += 1;
        return handler(request);
      },
    });
    class GateModel extends ScriptedChatModel {
      constructor() {
        super([new AIMessage("nope")]);
        // Promise.resolve awaits this, which is the model-init await in the middleware.
        // biome-ignore lint/suspicious/noThenProperty: the await must abort before the handler
        Object.defineProperty(this, "then", {
          configurable: true,
          value: (resolve: (model: BaseChatModel) => void) => {
            delete (this as { then?: unknown }).then;
            controller.abort(reason);
            resolve(this);
          },
        });
      }
    }
    const { agent } = agentOf({
      script: [{ body: routeReply("strong", ["fast", "strong"]) }],
      strong: new GateModel(),
      middleware: [counter],
    });
    await expect(
      agent.invoke({ messages: [new HumanMessage("hello")] }, { signal: controller.signal }),
    ).rejects.toBe(reason);
    expect(calls).toBe(0);
  });

  it("throws the abort reason when the handler settles after the run is aborted", async () => {
    const reason = new Error("stop-during-handler");
    const controller = new AbortController();
    let release: () => void = () => {};
    let started: () => void = () => {};
    const gate = new Promise<void>((resolve) => {
      started = resolve;
    });
    class SlowModel extends ScriptedChatModel {
      override async _generate(): Promise<ChatResult> {
        started();
        await new Promise<void>((resolve) => {
          release = resolve;
        });
        throw new Error("from-model");
      }
    }
    const { agent } = agentOf({
      script: [{ body: routeReply("strong", ["fast", "strong"]) }],
      strong: new SlowModel([new AIMessage("nope")]),
    });
    const pending = agent
      .invoke({ messages: [new HumanMessage("hello")] }, { signal: controller.signal })
      .then(
        () => ({ failed: false as const }),
        (error: unknown) => ({ failed: true as const, error }),
      );
    await gate;
    controller.abort(reason);
    release();
    const outcome = await pending;
    expect(outcome.failed).toBe(true);
    if (outcome.failed) {
      expect(outcome.error).toBe(reason);
      expect(String(outcome.error)).not.toContain("from-model");
    }
  });

  it("keeps a provider-strategy structured response and the selection", async () => {
    const { agent } = agentOf({
      script: [{ body: routeReply("strong", ["fast", "strong"]) }],
      strong: new StructuredReply("from-strong", "json"),
      responseFormat: providerStrategy(z.object({ answer: z.string() })),
    });
    const result = await agent.invoke({ messages: [new HumanMessage("hello")] });
    expect((result as { structuredResponse?: unknown }).structuredResponse).toEqual({
      answer: "from-strong",
    });
    expect(result.modelSelection).toMatchObject({ model: "strong", reason: "selected" });
    expect(result.messages.filter((message) => AIMessage.isInstance(message))).toHaveLength(1);
  });

  it("reuses a well formed stored selection and ignores a malformed one", async () => {
    const valid = {
      model: "fast",
      selected: "fast",
      probabilities: { fast: 0.9, strong: 0.1 },
      fallback: false,
      reason: "selected",
      priority: "balanced",
      candidates: ["fast", "strong"],
      requestId: "req_abc12345",
      latencyMs: 4,
      anchor: selectionAnchor([new HumanMessage({ content: "hello", id: "human_1" })]),
    };
    const malformed = [
      { ...valid, reason: "nope" },
      { ...valid, requestId: "raw-body" },
      { ...valid, selected: "not-a-candidate" },
      { ...valid, probabilities: { fast: 2 } },
      { ...valid, fallback: "yes" },
      { ...valid, candidates: ["fast", 1] },
      { ...valid, model: null, reason: "nope", selected: null, fallback: true },
      { ...valid, reason: "selected", model: "fast", selected: "strong", fallback: false },
      { ...valid, reason: "selected", model: "fast", selected: "fast", fallback: true },
    ];
    const { agent, drex } = agentOf({
      script: [valid, ...malformed].map(() => ({
        body: routeReply("strong", ["fast", "strong"]),
      })),
      checkpointer: new MemorySaver(),
    });
    const thread = { configurable: { thread_id: "stored-record" } };
    const message = new HumanMessage({ content: "hello", id: "human_1" });
    await agent.invoke({ messages: [message] }, thread);
    expect(drex.requests).toHaveLength(1);
    await agent.updateState(thread, { modelSelection: valid });
    const reused = await agent.invoke({ messages: [message] }, thread);
    expect(drex.requests).toHaveLength(1);
    expect(reused.modelSelection).toMatchObject({ model: "fast", reason: "selected" });
    for (const stored of malformed) {
      const before = drex.requests.length;
      await agent.updateState(thread, { modelSelection: stored });
      const result = await agent.invoke({ messages: [message] }, thread);
      expect(drex.requests).toHaveLength(before + 1);
      expect(result.modelSelection).toMatchObject({ model: "strong", reason: "selected" });
      const body = JSON.stringify(result.modelSelection);
      expect(body).not.toContain("nope");
      expect(body).not.toContain("not-a-candidate");
      expect(body).not.toContain("raw-body");
    }
  });

  it("ignores a stored selection that is not a plain eligible record", async () => {
    class Box {
      model = "fast";
      anchor = "count:1";
    }
    const boom = {};
    Object.defineProperty(boom, "model", {
      get() {
        throw new TypeError("boom");
      },
    });
    Object.defineProperty(boom, "anchor", {
      get() {
        throw new TypeError("boom");
      },
    });
    for (const stored of [new Box(), boom]) {
      let planted = false;
      const poison = createMiddleware({
        name: "Poison",
        wrapModelCall: async (request, handler) => {
          if (!planted) {
            planted = true;
            (request.state as { modelSelection?: unknown }).modelSelection = stored;
          }
          return handler(request);
        },
      });
      const drex = fakeDrex([{ body: routeReply("strong", ["fast", "strong"]) }]);
      const agent = createAgent({
        model: pair("base"),
        tools: [],
        middleware: [
          poison,
          naceModelSelectorMiddleware({
            candidates: { fast: { model: pair("fast") }, strong: { model: pair("strong") } },
            fallback: "fast",
            selector: new NaceModelSelector(offline(drex.fetch)),
          }),
        ],
      });
      const result = await agent.invoke({ messages: [new HumanMessage("hello")] });
      expect(drex.requests).toHaveLength(1);
      expect(result.modelSelection).toMatchObject({ model: "strong", reason: "selected" });
    }
  });

  it("refuses a profile the server would refuse", () => {
    const model = pair("x");
    const provider = "👍".repeat(129);
    expect(() =>
      naceModelSelectorMiddleware({
        candidates: {
          fast: {
            model,
            profile: { provider, priceIn: 1, priceOut: 1, context: 8, tags: ["ok"] },
          },
        },
      }),
    ).toThrow(/profile/);
    const extra = {
      provider: "openai",
      priceIn: 1,
      priceOut: 1,
      context: 8,
      tags: ["ok"],
      note: "nope",
    };
    expect(() =>
      naceModelSelectorMiddleware({
        candidates: { fast: { model, profile: extra as never } },
      }),
    ).toThrow(/profile/);
  });

  it("case 20: construction refuses an empty list, a long id, a bad fallback and a bad profile", () => {
    const model = pair("x");
    expect(() => naceModelSelectorMiddleware({ candidates: {} })).toThrow(/at least one/);
    const many = Object.fromEntries(
      Array.from({ length: 256 }, (_, index) => [`m${index}`, { model }]),
    );
    expect(() => naceModelSelectorMiddleware({ candidates: many })).toThrow(/255/);
    expect(() => naceModelSelectorMiddleware({ candidates: { " ": { model } } })).toThrow(/blank/);
    expect(() =>
      naceModelSelectorMiddleware({ candidates: { ["x".repeat(257)]: { model } } }),
    ).toThrow(/256/);
    expect(() =>
      naceModelSelectorMiddleware({ candidates: { fast: { model } }, fallback: "missing" }),
    ).toThrow(/fallback/);
    expect(() =>
      naceModelSelectorMiddleware({
        candidates: {
          fast: {
            model,
            profile: { provider: "", priceIn: -1, priceOut: 1, context: 0, tags: [] },
          },
        },
      }),
    ).toThrow(/profile/);
    expect(
      () =>
        new NaceModelSelector({ baseUrl: "https://openrouter.ai/api/v1", apiKey: "sk-or-test" }),
    ).toThrow(/OpenRouter/);
  });
});
