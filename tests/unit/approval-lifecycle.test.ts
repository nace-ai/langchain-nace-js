/** Approval lifecycle across interrupts, checkpoints and agent rebuilds; all traffic is fake. */
import { readFileSync } from "node:fs";
import { AIMessage, HumanMessage, ToolMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { Command, interrupt, isGraphBubbleUp, MemorySaver, task } from "@langchain/langgraph";
import { createAgent, createMiddleware } from "langchain";
import { expect, it, vi } from "vitest";
import * as z from "zod/v4";
import { NaceAPIError } from "../../src/errors.js";
import {
  AskHuman,
  GateDecision,
  type GateVerdict,
  naceAutoModeMiddleware,
  thresholdPolicy,
} from "../../src/middleware/index.js";
import { fakeDrex, offline, okBody, type Reply, ScriptedChatModel } from "./helpers.js";

// Lets a test switch off the gate's look at the task's scratchpad (see the last test).
const scratchpad = vi.hoisted(() => ({ hidden: false }));
vi.mock("@langchain/langgraph", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@langchain/langgraph")>();
  return { ...actual, getConfig: () => (scratchpad.hidden ? undefined : actual.getConfig()) };
});

const RED = readFileSync(new URL("../fixtures/red.png", import.meta.url));
const BLUE = readFileSync(new URL("../fixtures/blue.png", import.meta.url));

/** A gate verdict: risk level 0..4 and P(destructive); the call was asked for. */
function verdict(risk: number, destructive: number) {
  return (body: any): Reply => ({
    body: okBody(body, {
      risk: {
        type: "score",
        score: risk,
        legend: ["a", "b", "c", "d", "e"],
        confidence: 1,
        probabilities: { "0": 1 },
      },
      destructive: { type: "noul", noul: destructive },
      matches_request: { type: "noul", noul: 1 },
    }),
  });
}
const ASK = verdict(2, 1);
const ALLOW = verdict(0, 0);

function sendTool(ran: string[]) {
  return tool(
    async ({ target }: { target?: string }) => {
      ran.push(target ?? "send");
      return "sent";
    },
    { name: "send_report", schema: z.object({ target: z.string().optional() }) },
  );
}

const sendCall = () =>
  new AIMessage({ content: "", tool_calls: [{ id: "send", name: "send_report", args: {} }] });

const toolMessage = (result: { messages: unknown[] }) =>
  result.messages.find((m): m is ToolMessage => ToolMessage.isInstance(m));

const OTHER_ANSWER = "an answer was waiting that was given for a different decision";

/** Pause on one gated call with a first agent, then build a second agent (another worker). */
async function pauseThenRebuild(options: {
  script: (Reply | ((body: any) => Reply))[];
  gate?: Parameters<typeof naceAutoModeMiddleware>[0];
  // biome-ignore lint/suspicious/noExplicitAny: any LangChain tool
  tools?: (ran: string[]) => any[];
  thread: string;
}) {
  const server = fakeDrex(options.script);
  const saver = new MemorySaver();
  const ran: string[] = [];
  const make = (replies: AIMessage[]) =>
    createAgent({
      model: new ScriptedChatModel(replies),
      tools: options.tools?.(ran) ?? [sendTool(ran)],
      checkpointer: saver,
      middleware: [
        naceAutoModeMiddleware({ classifierOptions: offline(server.fetch), ...options.gate }),
      ],
    });
  const config = { configurable: { thread_id: options.thread } };
  const paused = await make([sendCall(), new AIMessage("done")]).invoke(
    { messages: [new HumanMessage("Send the report")] },
    config,
  );
  const resume = (value: unknown) =>
    make([new AIMessage("done")]).invoke(new Command({ resume: value }), config);
  return { server, saver, ran, paused, resume };
}

it.each([
  { changed: false, second: ASK, answer: "approve", runs: true },
  { changed: true, second: ASK, answer: "approve", runs: false },
  { changed: true, second: ALLOW, answer: "approve", runs: false },
  { changed: false, second: ALLOW, answer: "reject", runs: false },
])("binds a pending approval to its context across a rebuild: %j", async (c) => {
  let screen = RED;
  const { server, ran, paused, resume } = await pauseThenRebuild({
    script: [ASK, c.second],
    gate: { screenshots: () => [screen] },
    thread: `bind-${c.changed}-${c.answer}`,
  });
  expect((paused as any).__interrupt__).toHaveLength(1);
  if (c.changed) screen = BLUE;
  await resume({ type: c.answer });
  expect(ran).toEqual(c.runs ? ["send"] : []);
  // A resume replays the checkpointed decision; it never asks Drex again.
  expect(server.requests).toHaveLength(1);
});

class Audited extends GateDecision {
  async apply(request: any, handler: any) {
    return handler(request);
  }
}

class TwoPersonRule extends GateDecision {
  async apply(request: any, handler: any, _verdict: unknown, context: any) {
    const answer = interrupt({ type: "two_person", name: request.toolCall.name }) as any;
    if (answer?.type === "approve") return handler(request);
    return context.blocked("rejected by the two-person rule");
  }
}

it.each([
  [
    "a custom decision that runs the tool",
    (d: GateDecision) => (d instanceof AskHuman ? d : new Audited()),
  ],
  [
    "a custom decision that asks itself",
    (d: GateDecision) => (d instanceof AskHuman ? new TwoPersonRule() : d),
  ],
])("%s cannot bypass a pending rejection after a rebuild", async (_name, wrap) => {
  const base = thresholdPolicy();
  const { ran, paused, resume } = await pauseThenRebuild({
    script: [ASK, ALLOW],
    gate: { policy: (v: GateVerdict) => wrap(base(v)) },
    thread: `custom-${_name}`,
  });
  expect((paused as any).__interrupt__).toHaveLength(1);
  await resume({ type: "reject" });
  expect(ran).toEqual([]);
});

it("answers a tool's own interrupt with the tool's resume value after a rebuild", async () => {
  const answers: unknown[] = [];
  const { server, paused, resume } = await pauseThenRebuild({
    // The first check allows; a second check (if any) would ask a person.
    script: [ALLOW, ASK],
    tools: () => [
      tool(
        async () => {
          const answer = interrupt({ question: "Which channel?" });
          answers.push(answer);
          return `sent to ${String(answer)}`;
        },
        { name: "send_report", schema: z.object({}) },
      ),
    ],
    thread: "tool-interrupt",
  });
  expect((paused as any).__interrupt__[0].value).toEqual({ question: "Which channel?" });
  const result = await resume("#general");
  expect(answers).toEqual(["#general"]);
  expect(toolMessage(result)?.content).toBe("sent to #general");
  expect(server.requests).toHaveLength(1);
});

it("blocks a pending approval when the policy changed while it was paused", async () => {
  const server = fakeDrex([ASK]);
  const saver = new MemorySaver();
  const ran: string[] = [];
  const make = (policy: ReturnType<typeof thresholdPolicy>, replies: AIMessage[]) =>
    createAgent({
      model: new ScriptedChatModel(replies),
      tools: [sendTool(ran)],
      checkpointer: saver,
      middleware: [naceAutoModeMiddleware({ classifierOptions: offline(server.fetch), policy })],
    });
  const config = { configurable: { thread_id: "policy" } };
  await make(thresholdPolicy(), [sendCall(), new AIMessage("done")]).invoke(
    { messages: [new HumanMessage("Send the report")] },
    config,
  );
  const result = await make(thresholdPolicy({ askAt: 0.9, destructiveAt: 1.1 }), [
    new AIMessage("done"),
  ]).invoke(new Command({ resume: { type: "reject" } }), config);
  expect(ran).toEqual([]);
  expect(toolMessage(result)?.content).toContain("policy changed while the call was paused");
});

it("reports a failed check, not a context change, when preparation fails", async () => {
  const server = fakeDrex([ALLOW]);
  const agent = createAgent({
    model: new ScriptedChatModel([sendCall(), new AIMessage("done")]),
    tools: [sendTool([])],
    checkpointer: new MemorySaver(),
    middleware: [
      naceAutoModeMiddleware({
        classifierOptions: offline(server.fetch),
        questions: { extra: { bad: { type: "noul", instructions: 5 } as any } },
      }),
    ],
  });
  const result = await agent.invoke(
    { messages: [new HumanMessage("Send the report")] },
    { configurable: { thread_id: "prepare" } },
  );
  expect(toolMessage(result)?.content).toContain("its risk could not be checked (NaceError)");
  expect(server.requests).toHaveLength(0);
});

it("checks again after a failed check with onError: 'throw', then replays on a rebuild", async () => {
  const server = fakeDrex([{ status: 500, body: {} }, ASK]);
  const saver = new MemorySaver();
  const ran: string[] = [];
  const make = (replies: AIMessage[]) =>
    createAgent({
      model: new ScriptedChatModel(replies),
      tools: [sendTool(ran)],
      checkpointer: saver,
      middleware: [
        naceAutoModeMiddleware({
          classifierOptions: offline(server.fetch, { maxRetries: 0 }),
          onError: "throw",
        }),
      ],
    });
  const config = { configurable: { thread_id: "retry" } };
  const agent = make([sendCall(), new AIMessage("done")]);
  // The failed check fails the run, and callers see Drex's own error.
  const failure = await agent
    .invoke({ messages: [new HumanMessage("Send the report")] }, config)
    .catch((error: unknown) => error);
  expect(NaceAPIError.isInstance(failure)).toBe(true);
  expect(failure).toMatchObject({ message: expect.stringMatching(/Drex returned 500/) });
  const paused = await agent.invoke(null as any, config);
  expect((paused as any).__interrupt__).toHaveLength(1);
  await make([new AIMessage("done")]).invoke(new Command({ resume: { type: "reject" } }), config);
  expect(ran).toEqual([]);
  expect(server.requests).toHaveLength(2);
});

it("decides directly when called outside a LangGraph run", async () => {
  const server = fakeDrex([ALLOW]);
  const gate = naceAutoModeMiddleware({ classifierOptions: offline(server.fetch) });
  let ran = false;
  const result = await (gate as any).wrapToolCall(
    {
      toolCall: { id: "c1", name: "send_report", args: {} },
      state: { messages: [new HumanMessage("Send the report")] },
      runtime: { configurable: { thread_id: "direct" } },
    },
    async () => {
      ran = true;
      return new ToolMessage({ content: "sent", tool_call_id: "c1" });
    },
  );
  expect(ran).toBe(true);
  expect(result.content).toBe("sent");
});

it("checkpoints JSON only: no screenshot bytes and no key", async () => {
  const { saver } = await pauseThenRebuild({
    script: [ASK],
    gate: { screenshots: () => [RED] },
    thread: "record",
  });
  const dump = JSON.stringify((saver as any).writes, (_key, value) =>
    value instanceof Uint8Array ? Buffer.from(value).toString("utf8") : value,
  );
  expect(dump).toContain('\\"decision\\":\\"ask\\"');
  expect(dump).not.toContain(RED.toString("base64").slice(16, 48));
  expect(dump).not.toContain("nace_sk_test");
});

it.each(["sync", "async", "exit"] as const)(
  "keeps the protection with durability %s",
  async (durability) => {
    for (const c of [
      { change: false, second: ASK, answer: "approve", runs: ["send"] },
      { change: true, second: ASK, answer: "approve", runs: [] },
      { change: false, second: ALLOW, answer: "reject", runs: [] },
    ]) {
      let screen = RED;
      const server = fakeDrex([ASK, c.second]);
      const saver = new MemorySaver();
      const ran: string[] = [];
      const make = (replies: AIMessage[]) =>
        createAgent({
          model: new ScriptedChatModel(replies),
          tools: [sendTool(ran)],
          checkpointer: saver,
          middleware: [
            naceAutoModeMiddleware({
              classifierOptions: offline(server.fetch),
              screenshots: () => [screen],
            }),
          ],
        });
      const config = {
        configurable: { thread_id: `durability-${durability}-${c.change}-${c.answer}` },
        durability,
      } as any;
      await make([sendCall(), new AIMessage("done")]).invoke(
        { messages: [new HumanMessage("Send the report")] },
        config,
      );
      if (c.change) screen = BLUE;
      await make([new AIMessage("done")]).invoke(
        new Command({ resume: { type: c.answer } }),
        config,
      );
      expect(ran).toEqual(c.runs);
    }
  },
);

it.each([0, 2])("preserves text-only fallback in a checkpointed call (risk %i)", async (risk) => {
  const server = fakeDrex([verdict(risk, risk === 0 ? 0 : 1)]);
  let ran = false;
  const send = tool(
    async () => {
      ran = true;
      return "sent";
    },
    { name: "send_report", schema: z.object({}) },
  );
  const agent = createAgent({
    model: new ScriptedChatModel([
      new AIMessage({ content: "", tool_calls: [{ id: "send", name: "send_report", args: {} }] }),
      new AIMessage("done"),
    ]),
    tools: [send],
    checkpointer: new MemorySaver(),
    middleware: [
      naceAutoModeMiddleware({
        classifierOptions: offline(server.fetch),
        screenshots: () => ["https://example.com/screenshot.png"],
      }),
    ],
  });
  const config = { configurable: { thread_id: "fallback" } };
  const result = await agent.invoke({ messages: [new HumanMessage("Send the report")] }, config);
  if (risk === 2) {
    expect((result as any).__interrupt__).toHaveLength(1);
    await agent.invoke(new Command({ resume: { type: "approve" } }), config);
  }
  expect(ran).toBe(true);
  expect(server.requests).toHaveLength(1);
  expect(server.requests[0]!.body.images).toBeUndefined();
});

it("keeps parallel approvals separate across agent reconstruction", async () => {
  const server = fakeDrex(Array.from({ length: 4 }, () => verdict(2, 1)));
  const saver = new MemorySaver();
  const ran: string[] = [];
  const send = tool(
    async ({ target }) => {
      ran.push(target);
      return "sent";
    },
    { name: "send_report", schema: z.object({ target: z.string() }) },
  );
  const make = (replies: AIMessage[]) =>
    createAgent({
      model: new ScriptedChatModel(replies),
      tools: [send],
      checkpointer: saver,
      middleware: [naceAutoModeMiddleware({ classifierOptions: offline(server.fetch) })],
    });
  const config = { configurable: { thread_id: "parallel" } };
  const original = make([
    new AIMessage({
      content: "",
      tool_calls: [
        { id: "a", name: "send_report", args: { target: "a" } },
        { id: "b", name: "send_report", args: { target: "b" } },
      ],
    }),
  ]);
  const paused = await original.invoke(
    { messages: [new HumanMessage("Send the reports")] },
    config,
  );
  const interrupts = (paused as any).__interrupt__;
  expect(interrupts).toHaveLength(2);
  const resume = Object.fromEntries(
    interrupts.map((item: any) => [
      item.id,
      {
        type: item.value.args.target === "a" ? "approve" : "reject",
      },
    ]),
  );
  await make([new AIMessage("done")]).invoke(new Command({ resume }), config);
  expect(ran).toEqual(["a"]);
  expect(server.requests).toHaveLength(2);
});

it.each([false, true])(
  "preserves the decision wire input with checkpoints (OpenRouter %s)",
  async (openrouter) => {
    const bodies: unknown[] = [];
    for (const checkpointed of [false, true]) {
      const server = fakeDrex([verdict(0, 0)]);
      const read = tool(async () => "ok", { name: "read", schema: z.object({}) });
      const agent = createAgent({
        model: new ScriptedChatModel([
          new AIMessage({ content: "", tool_calls: [{ id: "read", name: "read", args: {} }] }),
          new AIMessage("done"),
        ]),
        tools: [read],
        ...(checkpointed ? { checkpointer: new MemorySaver() } : {}),
        middleware: [
          naceAutoModeMiddleware({
            classifierOptions: offline(
              server.fetch,
              openrouter ? { baseUrl: "https://openrouter.ai/api/v1" } : {},
            ),
            screenshots: () => [readFileSync(new URL("../fixtures/red.png", import.meta.url))],
          }),
        ],
      });
      await agent.invoke(
        { messages: [new HumanMessage("Read the screen")] },
        checkpointed ? { configurable: { thread_id: "wire" } } : {},
      );
      expect(server.requests).toHaveLength(1);
      bodies.push(server.requests[0]!.body);
    }
    expect(bodies[0]).toEqual(bodies[1]);
  },
);

it("does not execute a resumed tool after cancellation during screenshot capture", async () => {
  const server = fakeDrex([verdict(2, 1)]);
  const controller = new AbortController();
  let capturingResume = false;
  let ran = false;
  const send = tool(
    async () => {
      ran = true;
      return "sent";
    },
    { name: "send_report", schema: z.object({}) },
  );
  const agent = createAgent({
    model: new ScriptedChatModel([
      new AIMessage({ content: "", tool_calls: [{ id: "send", name: "send_report", args: {} }] }),
      new AIMessage("done"),
    ]),
    tools: [send],
    checkpointer: new MemorySaver(),
    middleware: [
      naceAutoModeMiddleware({
        classifierOptions: offline(server.fetch),
        screenshots: async () => {
          if (capturingResume) controller.abort(new Error("review cancellation"));
          return [readFileSync(new URL("../fixtures/red.png", import.meta.url))];
        },
      }),
    ],
  });
  const config = { configurable: { thread_id: "cancel" } };
  await agent.invoke({ messages: [new HumanMessage("Send report")] }, config);
  capturingResume = true;
  await expect(
    agent.invoke(new Command({ resume: { type: "approve" } }), {
      ...config,
      signal: controller.signal,
    }),
  ).rejects.toThrow();
  expect(ran).toBe(false);
  expect(server.requests).toHaveLength(1);
});

// A person's answer belongs to the decision it was given for. These cases end up with an
// answer waiting for a decision that is not the one the person saw.

const sendCallWith = (args: Record<string, unknown>) =>
  new AIMessage({ content: "", tool_calls: [{ id: "send", name: "send_report", args }] });

/** One agent over a shared saver, so state can be edited and old checkpoints resumed. */
function agentOver(
  server: ReturnType<typeof fakeDrex>,
  saver: MemorySaver,
  ran: string[],
  replies: AIMessage[],
  gate: Parameters<typeof naceAutoModeMiddleware>[0] = {},
) {
  return createAgent({
    model: new ScriptedChatModel(replies),
    tools: [sendTool(ran)],
    checkpointer: saver,
    middleware: [naceAutoModeMiddleware({ classifierOptions: offline(server.fetch), ...gate })],
  });
}

/** Replace the paused AI message (same id) through `updateState`. */
async function replacePausedCall(agent: any, config: unknown, args: Record<string, unknown>) {
  const state = await agent.getState(config);
  const paused = state.values.messages.find(
    (m: unknown) => AIMessage.isInstance(m) && (m.tool_calls?.length ?? 0) > 0,
  );
  await agent.updateState(config, {
    messages: [
      new AIMessage({
        id: paused.id,
        content: "",
        tool_calls: [{ id: "send", name: "send_report", args }],
      }),
    ],
  });
}

it.each([
  { name: "changed call, approve, fresh Ask", edited: "secret", second: ASK, answer: "approve" },
  {
    name: "unchanged call, reject, fresh Allow",
    edited: "public",
    second: ALLOW,
    answer: "reject",
  },
])("does not apply an answer to a call edited with updateState: $name", async (c) => {
  const server = fakeDrex([ASK, c.second]);
  const ran: string[] = [];
  const agent = agentOver(server, new MemorySaver(), ran, [
    sendCallWith({ target: "public" }),
    new AIMessage("done"),
  ]);
  const config = { configurable: { thread_id: `update-${c.name}` } };
  const paused = await agent.invoke({ messages: [new HumanMessage("Send the report")] }, config);
  expect((paused as any).__interrupt__).toHaveLength(1);
  await replacePausedCall(agent, config, { target: c.edited });
  const result = await agent.invoke(new Command({ resume: { type: c.answer } }), config);
  expect(ran).toEqual([]);
  expect(toolMessage(result)?.content).toContain(OTHER_ANSWER);
});

it("does not apply a reject to a fresh decision on a superseded checkpoint", async () => {
  const server = fakeDrex([ASK, ALLOW]);
  const ran: string[] = [];
  const agent = agentOver(server, new MemorySaver(), ran, [sendCall(), new AIMessage("done")]);
  const thread_id = "fork";
  const config = { configurable: { thread_id } };
  const paused = await agent.invoke({ messages: [new HumanMessage("Send the report")] }, config);
  expect((paused as any).__interrupt__).toHaveLength(1);
  const checkpoint_id = (await (agent as any).getState(config)).config.configurable.checkpoint_id;
  await agent.invoke(new Command({ resume: { type: "approve" } }), config);
  expect(ran).toEqual(["send"]);
  const forked = await agent.invoke(new Command({ resume: { type: "reject" } }), {
    configurable: { thread_id, checkpoint_id },
  });
  expect(ran).toEqual(["send"]);
  expect(toolMessage(forked)?.content).toContain(OTHER_ANSWER);
});

class Configurable extends GateDecision {
  constructor(private readonly ask: boolean) {
    super();
  }

  async apply(request: any, handler: any, verdict: any, context: any) {
    if (this.ask) return new AskHuman().apply(request, handler, verdict, context);
    return handler(request);
  }
}

class Sneaky extends AskHuman {
  async apply(request: any, handler: any) {
    return handler(request);
  }
}

it.each([
  [
    "a custom decision configured differently on resume",
    () => () => new Configurable(true),
    () => () => new Configurable(false),
  ],
  [
    "an AskHuman subclass that runs the tool",
    () => thresholdPolicy(),
    () => {
      const base = thresholdPolicy();
      return (v: GateVerdict) => {
        const d = base(v);
        return d instanceof AskHuman ? new Sneaky() : d;
      };
    },
  ],
])("%s cannot bypass a pending rejection", async (name, first, rebuilt) => {
  const server = fakeDrex([ASK]);
  const saver = new MemorySaver();
  const ran: string[] = [];
  const config = { configurable: { thread_id: `identity-${name}` } };
  const paused = await agentOver(server, saver, ran, [sendCall(), new AIMessage("done")], {
    policy: first(),
  }).invoke({ messages: [new HumanMessage("Send the report")] }, config);
  expect((paused as any).__interrupt__).toHaveLength(1);
  const result = await agentOver(server, saver, ran, [new AIMessage("done")], {
    policy: rebuilt(),
  }).invoke(new Command({ resume: { type: "reject" } }), config);
  expect(ran).toEqual([]);
  expect(toolMessage(result)?.content).toContain(
    "a custom decision without an id can't be resumed",
  );
});

it("blocks through AskHuman alone when the gate can't see the waiting answer", async () => {
  const server = fakeDrex([ASK, ASK]);
  const ran: string[] = [];
  const agent = agentOver(server, new MemorySaver(), ran, [
    sendCallWith({ target: "public" }),
    new AIMessage("done"),
  ]);
  const config = { configurable: { thread_id: "no-probe" } };
  await agent.invoke({ messages: [new HumanMessage("Send the report")] }, config);
  await replacePausedCall(agent, config, { target: "secret" });
  scratchpad.hidden = true;
  try {
    const result = await agent.invoke(new Command({ resume: { type: "approve" } }), config);
    expect(ran).toEqual([]);
    expect(toolMessage(result)?.content).toContain(OTHER_ANSWER);
  } finally {
    scratchpad.hidden = false;
  }
});

class IdentifiedAsk extends GateDecision {
  constructor(readonly id: string) {
    super();
  }

  async apply(request: any, handler: any, verdict: any, context: any) {
    return new AskHuman().apply(request, handler, verdict, context);
  }
}

it.each([
  { rebuiltId: "review-v1", ran: ["send"], text: undefined },
  { rebuiltId: "review-v2", ran: [], text: "policy changed while the call was paused" },
])("resumes a custom decision only under its own id: rebuilt as $rebuiltId", async (c) => {
  const server = fakeDrex([ASK]);
  const saver = new MemorySaver();
  const ran: string[] = [];
  const base = thresholdPolicy();
  const policy = (id: string) => (v: GateVerdict) => {
    const d = base(v);
    return d instanceof AskHuman ? new IdentifiedAsk(id) : d;
  };
  const config = { configurable: { thread_id: `identified-${c.rebuiltId}` } };
  const paused = await agentOver(server, saver, ran, [sendCall(), new AIMessage("done")], {
    policy: policy("review-v1"),
  }).invoke({ messages: [new HumanMessage("Send the report")] }, config);
  expect((paused as any).__interrupt__).toHaveLength(1);
  const result = await agentOver(server, saver, ran, [new AIMessage("done")], {
    policy: policy(c.rebuiltId),
  }).invoke(new Command({ resume: { type: "approve" } }), config);
  expect(ran).toEqual(c.ran);
  expect(server.requests).toHaveLength(1);
  if (c.text) expect(toolMessage(result)?.content).toContain(c.text);
});

// A failed check ends the run (with `onError: "throw"`). Retrying the run asks Drex again;
// nothing records the failure, so nothing shifts the task indices other middleware relies on.

/**
 * An outer retry middleware that records its own task between two attempts. It lets graph
 * control flow (an interrupt) pass, as any retry middleware must.
 */
const auditRetry = createMiddleware({
  name: "AuditRetry",
  wrapToolCall: async (request, handler) => {
    try {
      return await handler(request);
    } catch (error) {
      if (isGraphBubbleUp(error)) throw error;
      await task("AuditRetry", async () => "logged")();
      return handler(request);
    }
  },
});

/** A custom decision that asks inside a nested task, and runs the tool only on "approve". */
class Review extends GateDecision {
  readonly id = "review-v1";

  async apply(request: any, handler: any, _verdict: unknown, context: any) {
    const answer = (await task("ReviewTool", async () => interrupt({ kind: "review" }))()) as any;
    if (answer?.type === "approve") return handler(request);
    return context.blocked("rejected in review");
  }
}

const retrying = (
  server: ReturnType<typeof fakeDrex>,
  saver: MemorySaver,
  ran: string[],
  replies: AIMessage[],
  gate: Parameters<typeof naceAutoModeMiddleware>[0] = {},
) =>
  createAgent({
    model: new ScriptedChatModel(replies),
    tools: [sendTool(ran)],
    checkpointer: saver,
    middleware: [
      auditRetry,
      naceAutoModeMiddleware({
        classifierOptions: offline(server.fetch, { maxRetries: 0 }),
        onError: "throw",
        ...gate,
      }),
    ],
  });

it("keeps an id-keyed rejection of a nested approval after a failed check and an outer retry", async () => {
  const server = fakeDrex([{ status: 500, body: {} }, ALLOW]);
  const saver = new MemorySaver();
  const ran: string[] = [];
  const config = { configurable: { thread_id: "nested-review" } };
  const gate = { policy: () => new Review() };
  const agent = retrying(server, saver, ran, [sendCall(), new AIMessage("done")], gate);
  // The failed gate task fails the run, even though the outer middleware catches its error.
  await expect(
    agent.invoke({ messages: [new HumanMessage("Send the report")] }, config),
  ).rejects.toThrow();
  expect(ran).toEqual([]);
  const paused = await agent.invoke(null as any, config);
  expect((paused as any).__interrupt__).toHaveLength(1);
  expect((paused as any).__interrupt__[0].value).toEqual({ kind: "review" });
  const id = (paused as any).__interrupt__[0].id;
  await retrying(server, saver, ran, [new AIMessage("done")], gate).invoke(
    new Command({ resume: { [id]: { type: "reject" } } }),
    config,
  );
  expect(ran).toEqual([]);
  expect(server.requests).toHaveLength(2);
});

it.each([
  { answer: "approve", runs: ["send"] },
  { answer: "reject", runs: [] },
])("honours a person's $answer after a failed check and an outer retry", async (c) => {
  const server = fakeDrex([{ status: 500, body: {} }, ASK]);
  const saver = new MemorySaver();
  const ran: string[] = [];
  const config = { configurable: { thread_id: `plain-ask-${c.answer}` } };
  const agent = retrying(server, saver, ran, [sendCall(), new AIMessage("done")]);
  await expect(
    agent.invoke({ messages: [new HumanMessage("Send the report")] }, config),
  ).rejects.toThrow();
  expect(ran).toEqual([]);
  const paused = await agent.invoke(null as any, config);
  expect((paused as any).__interrupt__).toHaveLength(1);
  const result = await retrying(server, saver, ran, [new AIMessage("done")]).invoke(
    new Command({ resume: { type: c.answer } }),
    config,
  );
  expect(ran).toEqual(c.runs);
  expect(toolMessage(result)?.content).not.toContain(OTHER_ANSWER);
  expect(server.requests).toHaveLength(2);
});

it.each([
  { name: "straight through", outer: false },
  { name: "replayed after an outer middleware pauses", outer: true },
])("with onError: 'block' a failed check is a block, $name", async (c) => {
  const server = fakeDrex([{ status: 500, body: {} }]);
  const ran: string[] = [];
  const pauseAfter = createMiddleware({
    name: "PauseAfter",
    wrapToolCall: async (request, handler) => {
      const result = await handler(request);
      interrupt({ kind: "audit" });
      return result;
    },
  });
  const gate = naceAutoModeMiddleware({
    classifierOptions: offline(server.fetch, { maxRetries: 0 }),
    onError: "block",
  });
  const make = (replies: AIMessage[]) =>
    createAgent({
      model: new ScriptedChatModel(replies),
      tools: [sendTool(ran)],
      checkpointer: shared,
      middleware: c.outer ? [pauseAfter, gate] : [gate],
    });
  const shared = new MemorySaver();
  const config = { configurable: { thread_id: `block-${c.name}` } };
  let result: any = await make([sendCall(), new AIMessage("done")]).invoke(
    { messages: [new HumanMessage("Send the report")] },
    config,
  );
  if (c.outer) {
    // The block is replayed from the checkpoint; Drex is not asked again.
    expect(result.__interrupt__).toHaveLength(1);
    expect(result.__interrupt__[0].value).toEqual({ kind: "audit" });
    result = await make([new AIMessage("done")]).invoke(
      new Command({ resume: { type: "approve" } }),
      config,
    );
  }
  expect(result.__interrupt__).toBeUndefined();
  expect(ran).toEqual([]);
  const message = toolMessage(result);
  expect(message?.status).toBe("error");
  expect(message?.content).toContain("its risk could not be checked");
  // The agent went on to its next model turn.
  expect(
    result.messages.some((m: unknown) => AIMessage.isInstance(m) && m.content === "done"),
  ).toBe(true);
  expect(server.requests).toHaveLength(1);
});

it("does not carry an id-keyed approval over to a call edited with updateState", async () => {
  const server = fakeDrex([ASK, ASK]);
  const ran: string[] = [];
  const agent = agentOver(server, new MemorySaver(), ran, [
    sendCallWith({ target: "public" }),
    new AIMessage("done"),
  ]);
  const config = { configurable: { thread_id: "update-id-keyed" } };
  const paused: any = await agent.invoke(
    { messages: [new HumanMessage("Send the report")] },
    config,
  );
  expect(paused.__interrupt__).toHaveLength(1);
  const original = paused.__interrupt__[0].id;
  await replacePausedCall(agent, config, { target: "secret" });
  const again: any = await agent.invoke(
    new Command({ resume: { [original]: { type: "approve" } } }),
    config,
  );
  // The old approval doesn't apply: "secret" didn't run and is waiting for its own answer.
  expect(ran).toEqual([]);
  expect(again.__interrupt__).toHaveLength(1);
  expect(again.__interrupt__[0].value.args).toEqual({ target: "secret" });
  expect(again.__interrupt__[0].id).not.toBe(original);
  expect(server.requests).toHaveLength(2);
});
