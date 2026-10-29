import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AIMessage, type BaseMessage } from "@langchain/core/messages";
import type { ChatResult } from "@langchain/core/outputs";

/** One request the fake Drex received. */
export interface Recorded {
  url: string;
  headers: Headers;
  body: any;
}

/** A scripted response: status, headers and body. */
export interface Reply {
  status?: number;
  headers?: Record<string, string>;
  body?: unknown;
  /** Delay before answering, in ms. */
  delayMs?: number;
}

type Answerer = (body: any, index: number) => Reply | undefined;

/** Answers every question plausibly: noul 0.9, first choice label, score at the middle level. */
export function autoAnswers(questions: Record<string, any>): Record<string, unknown> {
  const answers: Record<string, unknown> = {};
  for (const [id, q] of Object.entries(questions)) {
    if (q.type === "noul") answers[id] = { type: "noul", noul: 0.9 };
    else if (q.type === "choice") {
      const labels = Object.keys(q.criteria);
      const probabilities = Object.fromEntries(
        labels.map((l, i) => [l, i === 0 ? 0.8 : 0.2 / Math.max(labels.length - 1, 1)]),
      );
      answers[id] = { type: "choice", choice: labels[0], confidence: 0.8, probabilities };
    } else if (q.type === "score") {
      const n = q.criteria.length;
      answers[id] = {
        type: "score",
        score: (n - 1) / 2,
        legend: q.criteria,
        confidence: 0.7,
        probabilities: Object.fromEntries(q.criteria.map((_: unknown, i: number) => [i, 1 / n])),
      };
    } else answers[id] = { type: q.type, value: 42 };
  }
  return answers;
}

export function okBody(body: any, answers?: Record<string, unknown>) {
  return {
    model: body.model,
    answers: answers ?? autoAnswers(body.questions),
    usage: { input_tokens: 120, output_tokens: 0, image_tokens: 0 },
    evaluation_time_ms: 87,
    request_id: "req_body",
  };
}

/**
 * A fake Drex: a `fetch` that records each request and replies with the next
 * scripted reply, or a plausible 200 when the script has none.
 */
export function fakeDrex(script: (Reply | Answerer)[] = []) {
  const requests: Recorded[] = [];
  const fetch = async (input: string | URL | Request, init?: RequestInit): Promise<Response> => {
    const index = requests.length;
    const body = typeof init?.body === "string" ? JSON.parse(init.body) : undefined;
    requests.push({ url: String(input), headers: new Headers(init?.headers), body });
    const step = script[index];
    const reply = (typeof step === "function" ? step(body, index) : step) ?? {
      body: okBody(body),
    };
    if (reply.delayMs) await new Promise((resolve) => setTimeout(resolve, reply.delayMs));
    const status = reply.status ?? 200;
    const payload = reply.body ?? (status === 200 ? okBody(body) : {});
    return new Response(JSON.stringify(payload), {
      status,
      headers: {
        "content-type": "application/json",
        "x-request-id": "req_header",
        ...reply.headers,
      },
    });
  };
  return { fetch: fetch as typeof globalThis.fetch, requests };
}

/** Options for a classifier that talks to the fake. */
export function offline(fetch: typeof globalThis.fetch, extra: Record<string, unknown> = {}) {
  return { apiKey: "nace_sk_test", baseUrl: "https://drex.test", fetch, ...extra };
}

export const PNG_1x1 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

/** A chat model that replies with scripted messages, for agent tests. */
export class ScriptedChatModel extends BaseChatModel {
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
    const message = new AIMessage({
      content: reply.content,
      tool_calls: reply.tool_calls ?? [],
      id: `msg_${this.index}`,
    });
    return { generations: [{ message, text: message.text }] };
  }
}
