/**
 * LangGraph cookbook: route tickets with Drex, ask a person when Drex isn't sure.
 *
 * `naceConditionalEdge` asks Drex which team owns a ticket and follows the
 * label; below `minConfidence` it goes to `human_review`, which pauses the graph
 * with `interrupt()` until a person answers.
 *
 *     export NACE_API_KEY=nace_sk_...
 *     npx tsx examples/langgraph-router.ts
 */
import {
  Annotation,
  Command,
  END,
  interrupt,
  MemorySaver,
  START,
  StateGraph,
} from "@langchain/langgraph";

import { type Choice, naceConditionalEdge } from "@nace-ai/langchain-nace";

const TEAMS: Choice = {
  type: "choice",
  instructions: "Which team should own this support ticket?",
  criteria: {
    billing: "Charges, invoices, refunds, plans.",
    technical: "Bugs, outages, errors, integrations.",
    sales: "Pricing questions, new purchases, upgrades.",
  },
};

const Ticket = Annotation.Root({
  text: Annotation<string>(),
  team: Annotation<string>(),
});

export function build(minConfidence = 0.7) {
  const teams = Object.keys(TEAMS.criteria);
  const graph = new StateGraph(Ticket)
    .addNode("triage", () => ({}))
    .addNode("billing", () => ({ team: "billing" }))
    .addNode("technical", () => ({ team: "technical" }))
    .addNode("sales", () => ({ team: "sales" }))
    .addNode("human_review", (t) => ({ team: interrupt({ ticket: t.text }) as string }))
    .addEdge(START, "triage")
    .addConditionalEdges(
      "triage",
      naceConditionalEdge({
        question: TEAMS,
        state: (t: typeof Ticket.State) => t.text,
        minConfidence,
      }),
      [...teams, "human_review"] as ("billing" | "technical" | "sales" | "human_review")[],
    )
    .addEdge("billing", END)
    .addEdge("technical", END)
    .addEdge("sales", END)
    .addEdge("human_review", END);
  return graph.compile({ checkpointer: new MemorySaver() });
}

async function main(): Promise<void> {
  const app = build();
  const tickets = [
    "I was charged twice for March, please refund one charge.",
    "The API returns 500 errors since this morning.",
    "Hi, quick one about my account.",
  ];
  for (const [i, text] of tickets.entries()) {
    const config = { configurable: { thread_id: String(i) } };
    let result: any = await app.invoke({ text }, config);
    if (result.__interrupt__) {
      console.log(`${JSON.stringify(text)} -> Drex not sure, asking a person`);
      result = await app.invoke(new Command({ resume: "billing" }), config);
      console.log(`   person chose: ${result.team}`);
    } else {
      console.log(`${JSON.stringify(text)} -> ${result.team}`);
    }
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  await main();
}
