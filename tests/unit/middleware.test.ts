import { AIMessage, HumanMessage, ToolMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { Command, MemorySaver } from "@langchain/langgraph";
import { createAgent } from "langchain";
import { describe, expect, it, vi } from "vitest";
import * as z from "zod/v4";

import { NaceClassifier } from "../../src/index.js";
import {
  naceAutoModeMiddleware,
  naceModelRouterMiddleware,
  thresholdPolicy,
} from "../../src/middleware/index.js";
import { fakeDrex, offline, okBody, PNG_1x1, type Reply, ScriptedChatModel } from "./helpers.js";

/** A Drex reply for the gate: risk level 0..4, P(destructive), P(asked for). */
function gate(
  level: number,
  destructive: number,
  matches: number,
  extra: Record<string, unknown> = {},
) {
  return (body: any): Reply => ({
    body: okBody(body, {
      risk: {
        type: "score",
        score: level,
        legend: ["harmless", "low", "medium", "high", "severe"],
        confidence: 0.9,
        probabilities: { "0": 0.2, "1": 0.2, "2": 0.2, "3": 0.2, "4": 0.2 },
      },
      destructive: { type: "noul", noul: destructive },
      matches_request: { type: "noul", noul: matches },
      ...extra,
    }),
  });
}

function ticketTools() {
  const deleted: number[] = [];
  const deleteTicket = tool(
    async ({ id }: { id: number }) => {
      deleted.push(id);
      return `deleted ${id}`;
    },
    {
      name: "delete_ticket",
      description: "Delete a support ticket permanently.",
      schema: z.object({ id: z.number() }),
    },
  );
  const readTicket = tool(async ({ id }: { id: number }) => `ticket ${id}: printer on fire`, {
    name: "read_ticket",
    description: "Read a support ticket.",
    schema: z.object({ id: z.number() }),
  });
  return { deleted, tools: [deleteTicket, readTicket] };
}

function calling(name: string, args: Record<string, unknown>, id = "call_1") {
  return [new AIMessage({ content: "", tool_calls: [{ id, name, args }] }), new AIMessage("Done.")];
}

function toolMessages(result: { messages: unknown[] }): ToolMessage[] {
  return result.messages.filter((m): m is ToolMessage => ToolMessage.isInstance(m));
}

const config = (thread: string) => ({ configurable: { thread_id: thread } });

describe("naceAutoModeMiddleware", () => {
  it("allows a low-risk call and sends the original request, the call and its description", async () => {
    const drex = fakeDrex([gate(0, 0.05, 0.95)]);
    const { tools } = ticketTools();
    const agent = createAgent({
      model: new ScriptedChatModel(calling("read_ticket", { id: 7 })),
      tools,
      middleware: [naceAutoModeMiddleware({ classifierOptions: offline(drex.fetch) })],
    });
    const result = await agent.invoke({ messages: [new HumanMessage("Read ticket 7")] });
    expect(toolMessages(result)[0]!.content).toBe("ticket 7: printer on fire");
    const state = drex.requests[0]!.body.state;
    expect(state.original_request).toBe("Read ticket 7");
    expect(state.latest_user_message).toBe("Read ticket 7");
    expect(state.tool_call).toEqual({ id: "call_1", name: "read_ticket", args: { id: 7 } });
    expect(state.tool_description).toBe("Read a support ticket.");
    expect(Object.keys(drex.requests[0]!.body.questions)).toEqual([
      "risk",
      "destructive",
      "matches_request",
    ]);
  });

  it("blocks a high-risk call with the verdict in the message", async () => {
    const drex = fakeDrex([gate(4, 0.99, 0.9)]);
    const { deleted, tools } = ticketTools();
    const agent = createAgent({
      model: new ScriptedChatModel(calling("delete_ticket", { id: 7 })),
      tools,
      middleware: [naceAutoModeMiddleware({ classifierOptions: offline(drex.fetch) })],
    });
    const result = await agent.invoke({ messages: [new HumanMessage("Delete ticket 7")] });
    const message = toolMessages(result)[0]!;
    expect(deleted).toEqual([]);
    expect(message.status).toBe("error");
    expect(message.content).toBe(
      "The tool call `delete_ticket` was blocked because its risk is high " +
        "(risk 1.00, destructive 0.99, asked for 0.90). It was not run.",
    );
    expect(message.artifact).toEqual({
      nace_verdict: { risk: 1, p_destructive: 0.99, p_matches_request: 0.9 },
    });
  });

  it("blocks a destructive call the user didn't ask for", async () => {
    const drex = fakeDrex([gate(1, 0.95, 0.1)]);
    const { deleted, tools } = ticketTools();
    const agent = createAgent({
      model: new ScriptedChatModel(calling("delete_ticket", { id: 9 })),
      tools,
      middleware: [naceAutoModeMiddleware({ classifierOptions: offline(drex.fetch) })],
    });
    const result = await agent.invoke({ messages: [new HumanMessage("Summarize ticket 7")] });
    expect(deleted).toEqual([]);
    expect(toolMessages(result)[0]!.content).toContain("the user did not ask for");
  });

  it("asks a person, then runs the call on approve", async () => {
    const drex = fakeDrex([gate(2, 0.9, 0.95)]);
    const { deleted, tools } = ticketTools();
    const agent = createAgent({
      model: new ScriptedChatModel(calling("delete_ticket", { id: 7 })),
      tools,
      middleware: [naceAutoModeMiddleware({ classifierOptions: offline(drex.fetch) })],
      checkpointer: new MemorySaver(),
    });
    const paused = await agent.invoke(
      { messages: [new HumanMessage("Delete ticket 7")] },
      config("approve"),
    );
    const interrupts = (paused as any).__interrupt__;
    expect(interrupts[0].value).toEqual({
      type: "nace_tool_approval",
      tool_call_id: "call_1",
      name: "delete_ticket",
      args: { id: 7 },
      description: "Delete a support ticket permanently.",
      verdict: { risk: 0.5, p_destructive: 0.9, p_matches_request: 0.95 },
    });
    expect(deleted).toEqual([]);
    await agent.invoke(new Command({ resume: { type: "approve" } }), config("approve"));
    expect(deleted).toEqual([7]);
    // The verdict is reused on resume: Drex was asked once.
    expect(drex.requests).toHaveLength(1);
  });

  it("runs the edited call on edit, in either edit shape", async () => {
    for (const resume of [
      { type: "edit", args: { id: 19 } },
      { type: "edit", edited_action: { name: "delete_ticket", args: { id: 19 } } },
    ]) {
      const drex = fakeDrex([gate(2, 0.9, 0.95)]);
      const { deleted, tools } = ticketTools();
      const agent = createAgent({
        model: new ScriptedChatModel(calling("delete_ticket", { id: 7 })),
        tools,
        middleware: [naceAutoModeMiddleware({ classifierOptions: offline(drex.fetch) })],
        checkpointer: new MemorySaver(),
      });
      await agent.invoke({ messages: [new HumanMessage("Delete ticket 7")] }, config("edit"));
      await agent.invoke(new Command({ resume }), config("edit"));
      expect(deleted).toEqual([19]);
    }
  });

  it("returns the person's reason on reject", async () => {
    const drex = fakeDrex([gate(2, 0.9, 0.95)]);
    const { deleted, tools } = ticketTools();
    const agent = createAgent({
      model: new ScriptedChatModel(calling("delete_ticket", { id: 7 })),
      tools,
      middleware: [naceAutoModeMiddleware({ classifierOptions: offline(drex.fetch) })],
      checkpointer: new MemorySaver(),
    });
    await agent.invoke({ messages: [new HumanMessage("Delete ticket 7")] }, config("reject"));
    const result = await agent.invoke(
      new Command({ resume: { type: "reject", message: "legal hold" } }),
      config("reject"),
    );
    expect(deleted).toEqual([]);
    expect(toolMessages(result)[0]!.content).toBe(
      "The user rejected the tool call `delete_ticket`. It was not run. Reason: legal hold",
    );
  });

  it("blocks with an explanation when asking needs a checkpointer the agent lacks", async () => {
    const drex = fakeDrex([gate(2, 0.9, 0.95)]);
    const { deleted, tools } = ticketTools();
    const agent = createAgent({
      model: new ScriptedChatModel(calling("delete_ticket", { id: 7 })),
      tools,
      middleware: [naceAutoModeMiddleware({ classifierOptions: offline(drex.fetch) })],
    });
    const result = await agent.invoke({ messages: [new HumanMessage("Delete ticket 7")] });
    expect(deleted).toEqual([]);
    expect(toolMessages(result)[0]!.content).toContain("no checkpointer");
  });

  it("blocks instead of asking with askHuman: false", async () => {
    const drex = fakeDrex([gate(2, 0.9, 0.95)]);
    const { deleted, tools } = ticketTools();
    const agent = createAgent({
      model: new ScriptedChatModel(calling("delete_ticket", { id: 7 })),
      tools,
      middleware: [
        naceAutoModeMiddleware({ classifierOptions: offline(drex.fetch), askHuman: false }),
      ],
    });
    const result = await agent.invoke({ messages: [new HumanMessage("Delete ticket 7")] });
    expect(deleted).toEqual([]);
    expect(toolMessages(result)[0]!.content).toContain("it needs a person's approval");
  });

  it("fails closed when Drex can't answer, or throws with onError: 'throw'", async () => {
    const down = {
      status: 500,
      headers: { "retry-after-ms": "1" },
      body: { error: { message: "down" } },
    };
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const { deleted, tools } = ticketTools();
    const agent = createAgent({
      model: new ScriptedChatModel(calling("delete_ticket", { id: 7 })),
      tools,
      middleware: [
        naceAutoModeMiddleware({ classifierOptions: offline(fakeDrex([down, down, down]).fetch) }),
      ],
    });
    const result = await agent.invoke({ messages: [new HumanMessage("Delete ticket 7")] });
    expect(deleted).toEqual([]);
    expect(toolMessages(result)[0]!.content).toContain(
      "its risk could not be checked (NaceInternalServerError)",
    );
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();

    const throwing = createAgent({
      model: new ScriptedChatModel(calling("delete_ticket", { id: 7 })),
      tools,
      middleware: [
        naceAutoModeMiddleware({
          classifierOptions: offline(fakeDrex([down, down, down]).fetch),
          onError: "throw",
        }),
      ],
    });
    await expect(
      throwing.invoke({ messages: [new HumanMessage("Delete ticket 7")] }),
    ).rejects.toThrow(/Drex returned 500/);
    expect(deleted).toEqual([]);
  });

  it("checks only the listed tools", async () => {
    const drex = fakeDrex();
    const { tools } = ticketTools();
    const agent = createAgent({
      model: new ScriptedChatModel(calling("read_ticket", { id: 1 })),
      tools,
      middleware: [
        naceAutoModeMiddleware({
          tools: ["delete_ticket"],
          classifierOptions: offline(drex.fetch),
        }),
      ],
    });
    await agent.invoke({ messages: [new HumanMessage("Read ticket 1")] });
    expect(drex.requests).toHaveLength(0);
  });

  it("keeps the original request on the 40th tool call", async () => {
    const drex = fakeDrex(Array.from({ length: 40 }, () => gate(0, 0.01, 0.99)));
    const { tools } = ticketTools();
    const replies = Array.from(
      { length: 40 },
      (_, i) =>
        new AIMessage({
          content: "",
          tool_calls: [{ id: `c${i}`, name: "read_ticket", args: { id: i } }],
        }),
    );
    replies.push(new AIMessage("All read."));
    const agent = createAgent({
      model: new ScriptedChatModel(replies),
      tools,
      middleware: [naceAutoModeMiddleware({ classifierOptions: offline(drex.fetch) })],
    });
    await agent.invoke(
      { messages: [new HumanMessage("Read every ticket, but never delete anything.")] },
      { recursionLimit: 200 },
    );
    expect(drex.requests).toHaveLength(40);
    const last = drex.requests[39]!.body.state;
    expect(last.original_request).toBe("Read every ticket, but never delete anything.");
    expect(last.recent_messages).toHaveLength(20);
    expect(last.tool_call.args).toEqual({ id: 39 });
  });

  it("sends screenshots on Drex Max and retries once without them on a 422", async () => {
    const drex = fakeDrex([
      {
        status: 422,
        body: { error: { message: "bad image", issues: [{ path: "images.0", message: "bad" }] } },
      },
      gate(0, 0.01, 0.99),
    ]);
    const { tools } = ticketTools();
    const agent = createAgent({
      model: new ScriptedChatModel(calling("read_ticket", { id: 3 })),
      tools,
      middleware: [
        naceAutoModeMiddleware({ classifierOptions: offline(drex.fetch), screenshots: true }),
      ],
    });
    await agent.invoke({
      messages: [
        new HumanMessage({
          content: [
            { type: "text", text: "Here is my screen" },
            { type: "image", source_type: "base64", data: PNG_1x1, mime_type: "image/png" },
          ],
        }),
      ],
    });
    expect(drex.requests).toHaveLength(2);
    expect(drex.requests[0]!.body.images).toEqual([`data:image/png;base64,${PNG_1x1}`]);
    expect(drex.requests[0]!.body.state.screenshots).toBe("Image 1 to Image 1 show the screen.");
    expect(JSON.stringify(drex.requests[0]!.body.state.recent_messages)).toContain(
      "[image omitted]",
    );
    expect(drex.requests[1]!.body.images).toBeUndefined();
  });

  it("refuses screenshots with a text-only model", () => {
    expect(() =>
      naceAutoModeMiddleware({
        classifierOptions: offline(fakeDrex().fetch, { model: "drex-v2.0" }),
        screenshots: true,
      }),
    ).toThrow(/screenshots need an image model/);
  });

  it("accepts a custom policy and extra questions", async () => {
    const drex = fakeDrex([gate(0, 0.01, 0.99, { pii: { type: "noul", noul: 0.97 } })]);
    const { tools } = ticketTools();
    const agent = createAgent({
      model: new ScriptedChatModel(calling("read_ticket", { id: 7 })),
      tools,
      middleware: [
        naceAutoModeMiddleware({
          classifierOptions: offline(drex.fetch),
          questions: { extra: { pii: { type: "noul", instructions: "Does it expose PII?" } } },
          policy: thresholdPolicy({ askHuman: false }),
        }),
      ],
    });
    const result = await agent.invoke({ messages: [new HumanMessage("Read ticket 7")] });
    expect(drex.requests[0]!.body.questions.pii).toBeDefined();
    expect(toolMessages(result)[0]!.content).toContain("printer on fire");
  });
});

describe("naceModelRouterMiddleware", () => {
  function models() {
    const fast = new ScriptedChatModel([new AIMessage("fast answer")]);
    const strong = new ScriptedChatModel([new AIMessage("strong answer")]);
    const base = new ScriptedChatModel([new AIMessage("base answer")]);
    return { fast, strong, base };
  }

  function routeReply(choice: string, confidence: number) {
    return (body: any): Reply => ({
      body: okBody(body, {
        model_route: {
          type: "choice",
          choice,
          confidence,
          probabilities: { fast: confidence, strong: Math.round((1 - confidence) * 100) / 100 },
        },
      }),
    });
  }

  function routed(drex: ReturnType<typeof fakeDrex>, m: ReturnType<typeof models>) {
    return createAgent({
      model: m.base,
      tools: [],
      middleware: [
        naceModelRouterMiddleware({
          choices: {
            fast: { model: m.fast, criteria: "Simple questions." },
            strong: { model: m.strong, criteria: "Hard, multi-step work." },
          },
          instructions: "Pick the cheapest model that can do this well.",
          classifierOptions: offline(drex.fetch),
        }),
      ],
    });
  }

  it("uses the chosen model and keeps the decision in state", async () => {
    const drex = fakeDrex([routeReply("fast", 0.9)]);
    const m = models();
    const result = await routed(drex, m).invoke({ messages: [new HumanMessage("What is 2+2?")] });
    expect(result.messages.at(-1)!.content).toBe("fast answer");
    expect((result as any).modelRoute).toEqual({
      route: "fast",
      choice: "fast",
      confidence: 0.9,
      probabilities: { fast: 0.9, strong: 0.1 },
      escalated: false,
      reason: "confident",
    });
    const question = drex.requests[0]!.body.questions.model_route;
    expect(question.criteria).toEqual({
      fast: "Simple questions.",
      strong: "Hard, multi-step work.",
    });
    expect(drex.requests[0]!.body.state).toEqual({ role: "user", content: "What is 2+2?" });
  });

  it("escalates to the last choice below minConfidence", async () => {
    const m = models();
    const result = await routed(fakeDrex([routeReply("fast", 0.4)]), m).invoke({
      messages: [new HumanMessage("Plan a migration")],
    });
    expect(result.messages.at(-1)!.content).toBe("strong answer");
    expect((result as any).modelRoute).toMatchObject({
      route: "strong",
      escalated: true,
      reason: "low_confidence",
    });
  });

  it("escalates when Drex fails, instead of throwing", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const m = models();
    const result = await routed(
      fakeDrex([{ status: 401, body: { error: { message: "Invalid API key" } } }]),
      m,
    ).invoke({ messages: [new HumanMessage("hi")] });
    warn.mockRestore();
    expect(result.messages.at(-1)!.content).toBe("strong answer");
    expect((result as any).modelRoute).toMatchObject({
      route: "strong",
      choice: null,
      reason: "error",
    });
  });

  it("validates its config", () => {
    const fetch = fakeDrex().fetch;
    expect(() =>
      naceModelRouterMiddleware({
        choices: {},
        instructions: "x",
        classifierOptions: offline(fetch),
      }),
    ).toThrow(/at least one model/);
    expect(() =>
      naceModelRouterMiddleware({
        choices: { a: { model: "anthropic:x", criteria: "y" } },
        instructions: "x",
        escalateTo: "b",
        classifierOptions: offline(fetch),
      }),
    ).toThrow(/escalateTo "b"/);
  });

  it("shares a classifier passed in", () => {
    const classifier = new NaceClassifier(offline(fakeDrex().fetch));
    expect(() =>
      naceModelRouterMiddleware({
        choices: { a: { model: "anthropic:x", criteria: "y" } },
        instructions: "x",
        classifier,
      }),
    ).not.toThrow();
  });
});
