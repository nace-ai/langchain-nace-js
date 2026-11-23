/**
 * Pick the chat model for one agent with Drex Router.
 *
 * `naceModelSelectorMiddleware` sends the conversation to `POST /v1/route` and
 * runs the candidate Drex selects. The chat models are scripted. The route
 * call is real.
 *
 *     export NACE_API_KEY=nace_sk_...
 *     npx tsx examples/model-selection-agent.ts
 */
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AIMessage, type BaseMessage, HumanMessage } from "@langchain/core/messages";
import type { ChatResult } from "@langchain/core/outputs";
import { NaceModelSelector } from "@nace-ai/langchain-nace";
import { naceModelSelectorMiddleware } from "@nace-ai/langchain-nace/middleware";
import { createAgent } from "langchain";

const FAST_ID = "openai/gpt-4o-mini";
const STRONG_ID = "anthropic/claude-haiku-5.5";

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

const fast = new Scripted([new AIMessage("Reply from the fast candidate.")]);
const strong = new Scripted([new AIMessage("Reply from the strong candidate.")]);
const selector = new NaceModelSelector({ timeout: 60_000 });
const agent = createAgent({
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

const result = await agent.invoke({
  messages: [new HumanMessage("I was charged twice for March. Please refund one charge.")],
});
const selection = result.modelSelection;
if (!selection) throw new Error("no selection");
console.log(`model=${selection.model}`);
console.log(`reason=${selection.reason}`);
console.log(`probabilities=${JSON.stringify(selection.probabilities)}`);
