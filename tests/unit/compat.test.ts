/**
 * Switching from `@langchain/typesafe` (0.0.2): its README samples, with only
 * the imports and the API key variable changed, must type-check and run.
 */
import { AIMessage, HumanMessage, ToolMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { createAgent } from "langchain";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as z from "zod/v4";

// - import { TypeSafeClassifier, TypeSafeRateLimitError } from "@langchain/typesafe";
import {
  type Choice,
  type ClassificationResponse,
  NaceClassifier,
  type NaceClassifierFields,
  NaceRateLimitError,
  type Noul,
  type Score,
} from "../../src/index.js";
// - import { autoModeMiddleware, modelRouterMiddleware } from "@langchain/typesafe/middleware";
import {
  type AutoModeMiddlewareConfig,
  autoModeMiddleware,
  type ModelRouterMiddlewareConfig,
  modelRouterMiddleware,
  naceAutoModeMiddleware,
  naceModelRouterMiddleware,
} from "../../src/middleware/index.js";
import { fakeDrex, okBody, type Reply, ScriptedChatModel } from "./helpers.js";

let drex: ReturnType<typeof fakeDrex>;

beforeEach(() => {
  // - TYPESAFE_API_KEY
  vi.stubEnv("NACE_API_KEY", "nace_sk_test");
  vi.stubEnv("NACE_BASE_URL", "");
  vi.stubEnv("NACE_DEFAULT_DECISION_MODEL", "");
  drex = fakeDrex();
  vi.spyOn(globalThis, "fetch").mockImplementation(drex.fetch);
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
});

describe("README: usage", () => {
  it("runs the classifier sample unchanged", async () => {
    const classifier = new NaceClassifier({
      questions: {
        department: {
          type: "choice",
          criteria: {
            billing: "Payment and payout issues",
            technical: "Bugs and outages",
            sales: null,
          },
          instructions: "Which team should handle this?",
        },
        urgent: { type: "noul", instructions: "Does this convey urgency?" },
        frustration: {
          type: "score",
          criteria: ["calm", "frustrated", "angry"],
          instructions: "How frustrated is the writer?",
        },
      },
    });

    const result = await classifier.invoke({
      message: "Help! My payouts have been failing for 3 days.",
      account_tier: "enterprise",
    });

    expect(result.answers.department).toMatchObject({ type: "choice", choice: "billing" });
    expect(result.answers.urgent).toMatchObject({ type: "noul" });
    expect(result.answers.frustration).toMatchObject({ type: "score", legend: { 0: "calm" } });
    expect(result.choices.department).toBe(result.answers.department);
    expect(result.nouls.urgent).toBe(result.answers.urgent);
    expect(result.scores.frustration).toBe(result.answers.frustration);
    expect(typeof result.model).toBe("string");
    // Non-enumerable helpers stay out of JSON.
    expect(JSON.stringify(result)).not.toContain('"choices"');
  });

  it("accepts messages anywhere in the input", async () => {
    const classifier = new NaceClassifier({
      questions: { urgent: { type: "noul", instructions: "Urgent?" } },
    });
    const messages = [new HumanMessage("..."), new AIMessage("...")];
    await classifier.invoke(messages);
    await classifier.invoke({ conversation: messages, account: { tier: "pro" } });
    expect(drex.requests).toHaveLength(2);
  });

  it("accepts every TypeSafe constructor field", () => {
    const fields: NaceClassifierFields = {
      questions: { q: { type: "noul", instructions: "?" } },
      model: "drex-max",
      apiKey: "nace_sk_test",
      baseUrl: "https://console.nace.ai",
      timeout: 30_000,
      fetch: drex.fetch,
      dangerouslyAllowBrowser: true,
      maxRetries: 2,
      maxConcurrency: 4,
    };
    expect(() => new NaceClassifier(fields)).not.toThrow();
  });

  it("keeps TypeSafe's question and response type names", () => {
    const noul: Noul = { type: "noul", instructions: "?" };
    const choice: Choice = { type: "choice", criteria: { a: null } };
    const score: Score = { type: "score", criteria: ["low", "high"] };
    const read = (r: ClassificationResponse) => [
      r.model,
      r.answers,
      r.usage.inputTokens,
      r.requestId,
    ];
    expect([noul, choice, score, read]).toHaveLength(4);
  });
});

describe("README: errors", () => {
  it("exposes retryAfterMs and requestId on rate-limit errors", async () => {
    const limited: Reply = {
      status: 429,
      headers: { "retry-after-ms": "5" },
      body: { error: { message: "Too many requests" } },
    };
    drex = fakeDrex([limited, limited, limited]);
    vi.mocked(globalThis.fetch).mockImplementation(drex.fetch);
    const classifier = new NaceClassifier({
      questions: { urgent: { type: "noul", instructions: "Urgent?" } },
    });
    let caught = false;
    try {
      await classifier.invoke("input");
    } catch (error) {
      if (NaceRateLimitError.isInstance(error)) {
        caught = true;
        expect(error.retryAfterMs).toBe(5);
        expect(error.requestId).toBe("req_header");
        expect(error.status).toBe(429);
        expect(error.headers).toBeInstanceOf(Headers);
        expect(error.body).toBeDefined();
      }
    }
    expect(caught).toBe(true);
  });
});

describe("README: middleware", () => {
  it("builds the modelRouterMiddleware sample", () => {
    const config: ModelRouterMiddlewareConfig = {
      choices: {
        fast: {
          model: "openai:gpt-5-mini",
          criteria: "Simple, well-scoped tasks.",
        },
        powerful: {
          model: "openai:gpt-5",
          criteria: "Complex tasks requiring deeper reasoning.",
        },
      },
      instructions: "Choose the least costly model suited to the task.",
    };
    const middleware = modelRouterMiddleware(config);
    expect(middleware.name).toBe("NaceModelRouterMiddleware");
    expect(modelRouterMiddleware).toBe(naceModelRouterMiddleware);
    expect(autoModeMiddleware).toBe(naceAutoModeMiddleware);
  });

  it("runs the autoModeMiddleware sample, including TypeSafe's own options", async () => {
    let ran = 0;
    const runSqlTool = tool(
      async () => {
        ran += 1;
        return "ok";
      },
      { name: "run_sql", description: "Run SQL.", schema: z.object({ sql: z.string() }) },
    );
    const sendEmailTool = tool(async () => "sent", {
      name: "send_email",
      description: "Send an email.",
      schema: z.object({ to: z.string() }),
    });
    drex = fakeDrex([
      (body): Reply => ({
        body: okBody(body, {
          risk: {
            type: "score",
            score: 0,
            legend: ["a", "b", "c", "d", "e"],
            confidence: 0.9,
            probabilities: { "0": 1 },
          },
          destructive: { type: "noul", noul: 0.01 },
          matches_request: { type: "noul", noul: 0.99 },
          is_risky: { type: "noul", noul: 0.81 },
        }),
      }),
    ]);
    vi.mocked(globalThis.fetch).mockImplementation(drex.fetch);

    const agent = createAgent({
      model: new ScriptedChatModel([
        new AIMessage({
          content: "",
          tool_calls: [{ id: "c1", name: "run_sql", args: { sql: "DROP TABLE users" } }],
        }),
        new AIMessage("Done."),
      ]),
      tools: [runSqlTool, sendEmailTool],
      middleware: [
        autoModeMiddleware({
          tools: ["run_sql", "send_email"],
          instructions: "Would running `tool_call` be risky?",
          criteria: { true: "Harmful.", false: "Safe." },
          blockedMessage: "Blocked {tool_name} at {probability}.",
          classifierOptions: { maxRetries: 0 },
        } satisfies AutoModeMiddlewareConfig),
      ],
    });
    const result = await agent.invoke({ messages: [new HumanMessage("Clean up the database")] });
    const message = result.messages.find((m): m is ToolMessage => ToolMessage.isInstance(m))!;
    // TypeSafe's question is asked alongside Drex's, and its 0.5 threshold still blocks.
    expect(drex.requests[0]!.body.questions.is_risky).toEqual({
      type: "noul",
      instructions: "Would running `tool_call` be risky?",
      criteria: { true: "Harmful.", false: "Safe." },
    });
    expect(ran).toBe(0);
    expect(message.content).toBe("Blocked run_sql at 0.81.");
  });
});
