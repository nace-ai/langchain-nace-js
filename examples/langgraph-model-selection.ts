/**
 * Route a support request, then let Drex Router pick the model.
 *
 * `NaceClassifier` answers a Choice question: billing or support. A conditional
 * edge sends the request to that agent. `naceModelSelectorMiddleware` picks the
 * generative model inside the agent. The chat models are scripted. The Drex
 * calls are real. The selector and both agents are built once.
 *
 *     export NACE_API_KEY=nace_sk_...
 *     npx tsx examples/langgraph-model-selection.ts
 */
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AIMessage, type BaseMessage, HumanMessage } from "@langchain/core/messages";
import type { ChatResult } from "@langchain/core/outputs";
import {
  Annotation,
  Command,
  END,
  interrupt,
  MemorySaver,
  START,
  StateGraph,
} from "@langchain/langgraph";
import { type Choice, NaceModelSelector, naceConditionalEdge } from "@nace-ai/langchain-nace";
import { naceModelSelectorMiddleware } from "@nace-ai/langchain-nace/middleware";
import { createAgent } from "langchain";

const FAST_ID = "openai/gpt-4o-mini";
const STRONG_ID = "anthropic/claude-haiku-5.5";
const TEAMS: Choice = {
  type: "choice",
  instructions: "Which team should own this support request?",
  criteria: {
    billing: "Charges, invoices, refunds, plans.",
    support: "Bugs, outages, how-to questions.",
  },
};

const Request = Annotation.Root({
  text: Annotation<string>(),
  team: Annotation<string>(),
  model: Annotation<string | null>(),
  reason: Annotation<string>(),
  probabilities: Annotation<Record<string, number>>(),
});

/** Replies with one scripted message, so the example needs no model provider. */
class Scripted extends BaseChatModel {
  private index = 0;

  constructor(private readonly replies: AIMessage[]) {
    super({});
  }

  _llmType(): string {
    return "scripted";
  }

  bindTools(): this {
    return this;
  }

  async _generate(_messages: BaseMessage[]): Promise<ChatResult> {
    const reply = this.replies[Math.min(this.index, this.replies.length - 1)]!;
    this.index += 1;
    const message = new AIMessage({ content: reply.content, tool_calls: reply.tool_calls ?? [] });
    return { generations: [{ message, text: message.text }] };
  }
}

const selector = new NaceModelSelector({ timeout: 60_000 });

function teamAgent(team: string) {
  const fast = new Scripted([new AIMessage(`${team} reply from the fast candidate.`)]);
  const strong = new Scripted([new AIMessage(`${team} reply from the strong candidate.`)]);
  return createAgent({
    model: fast,
    tools: [],
    middleware: [
      naceModelSelectorMiddleware({
        candidates: {
          [FAST_ID]: { model: fast },
          [STRONG_ID]: { model: strong },
        },
        priority: "balanced",
        fallback: FAST_ID,
        timeout: 60_000,
        selector,
      }),
    ],
  });
}

const billingAgent = teamAgent("billing");
const supportAgent = teamAgent("support");
const agents = { billing: billingAgent, support: supportAgent };

async function answer(team: "billing" | "support", text: string) {
  const result = await agents[team].invoke({ messages: [new HumanMessage(text)] });
  const selection = result.modelSelection;
  if (!selection) throw new Error("no selection");
  return {
    team,
    model: selection.model,
    reason: selection.reason,
    probabilities: selection.probabilities,
  };
}

export function build() {
  const graph = new StateGraph(Request)
    .addNode("triage", () => ({}))
    .addNode("billing", (state) => answer("billing", state.text))
    .addNode("support", (state) => answer("support", state.text))
    .addNode("human_review", (state) => {
      const team = interrupt({ ticket: state.text }) as string;
      return answer(team === "billing" ? "billing" : "support", state.text);
    })
    .addEdge(START, "triage")
    .addConditionalEdges(
      "triage",
      naceConditionalEdge({
        question: TEAMS,
        state: (state: typeof Request.State) => state.text,
        minConfidence: 0.7,
      }),
      ["billing", "support", "human_review"],
    )
    .addEdge("billing", END)
    .addEdge("support", END)
    .addEdge("human_review", END);
  return graph.compile({ checkpointer: new MemorySaver() });
}

async function main(): Promise<void> {
  const app = build();
  const tickets = [
    "I was charged twice for March. Please refund one charge.",
    "The API returns 500 errors since this morning.",
  ];
  for (const [index, text] of tickets.entries()) {
    const config = { configurable: { thread_id: String(index) } };
    let result: typeof Request.State & { __interrupt__?: unknown[] } = await app.invoke(
      { text },
      config,
    );
    if (result.__interrupt__) {
      console.log(`${JSON.stringify(text)} -> Drex not sure, asking a person`);
      result = await app.invoke(new Command({ resume: "support" }), config);
    }
    console.log(`${JSON.stringify(text)} -> ${result.team}`);
    console.log(`model=${result.model}`);
    console.log(`reason=${result.reason}`);
    console.log(`probabilities=${JSON.stringify(result.probabilities)}`);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  await main();
}
