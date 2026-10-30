import { expect, it } from "vitest";
import { NaceBatchStoppedError, NaceClassifier } from "../../src/index.js";
import { fakeDrex, okBody, type Reply } from "./helpers.js";

const questions = { q: { type: "noul" as const, instructions: "Urgent?" } };
const ok = (body: any): Reply => ({ body: okBody(body), delayMs: 5 });
const status = (code: number): Reply => ({
  status: code,
  body: { error: { message: "no" } },
  delayMs: 5,
});
const settle = () => new Promise((resolve) => setTimeout(resolve, 100));

function classifier(script: (Reply | ((body: any) => Reply))[], extra = {}) {
  const server = fakeDrex(script);
  return {
    server,
    classifier: new NaceClassifier({
      apiKey: "nace_sk_test",
      baseUrl: "https://drex.test",
      fetch: server.fetch,
      questions,
      maxRetries: 0,
      ...extra,
    }),
  };
}
const inputs = Array.from({ length: 20 }, (_, i) => `ticket ${i}`);

it.each([401, 402, 403])(
  "stops sending after a %i, with or without returnExceptions",
  async (code) => {
    for (const returnExceptions of [false, true]) {
      const { server, classifier: c } = classifier(Array.from({ length: 20 }, () => status(code)));
      const run = c.batch(inputs, { maxConcurrency: 2 }, { returnExceptions } as any);
      if (returnExceptions) {
        const out = (await run) as unknown as Error[];
        expect(out[0]!.name).toMatch(/Nace/);
        expect(out.filter((r) => r instanceof NaceBatchStoppedError).length).toBeGreaterThan(15);
      } else await expect(run).rejects.toThrow(String(code));
      await settle();
      expect(server.requests.length).toBeLessThanOrEqual(2);
    }
  },
);

it("keeps sending after a per-input 422 with returnExceptions", async () => {
  const { server, classifier: c } = classifier([
    ok,
    ok,
    status(422),
    ...Array.from({ length: 17 }, () => ok),
  ]);
  const out = await c.batch(inputs, { maxConcurrency: 2 }, { returnExceptions: true });
  expect(server.requests).toHaveLength(20);
  expect(out.filter((r) => r instanceof Error)).toHaveLength(1);
});

it("rejects at the first error without starting queued inputs", async () => {
  const { server, classifier: c } = classifier([
    ok,
    status(422),
    ...Array.from({ length: 18 }, () => ok),
  ]);
  await expect(c.batch(inputs, { maxConcurrency: 2 })).rejects.toThrow("422");
  await settle();
  expect(server.requests.length).toBeLessThanOrEqual(3);
});

it("keeps input order and honours batchOptions.maxConcurrency", async () => {
  let inFlight = 0;
  let peak = 0;
  const { classifier: c } = classifier(
    Array.from({ length: 6 }, (_, i) => (body: any) => {
      inFlight += 1;
      peak = Math.max(peak, inFlight);
      setTimeout(() => (inFlight -= 1), 60 - i * 10);
      return { body: okBody(body, { q: { type: "noul", noul: i / 10 } }), delayMs: 60 - i * 10 };
    }),
  );
  const out = await c.batch(inputs.slice(0, 6), undefined, { maxConcurrency: 1 });
  expect(out.map((r) => r.nouls.q!.noul)).toEqual([0, 0.1, 0.2, 0.3, 0.4, 0.5]);
  expect(peak).toBe(1);
});

it("emits one run per input, including inputs that were not sent", async () => {
  const events: string[] = [];
  const handler = {
    handleLLMStart: () => void events.push("start"),
    handleLLMError: () => void events.push("error"),
    handleLLMEnd: () => void events.push("end"),
    handleChainStart: () => void events.push("start"),
    handleChainError: () => void events.push("error"),
    handleChainEnd: () => void events.push("end"),
  };
  const { classifier: c } = classifier([status(401)]);
  const out = await c.batch(
    inputs.slice(0, 5),
    { maxConcurrency: 1, callbacks: [handler] },
    { returnExceptions: true },
  );
  expect(out.every((r) => r instanceof Error)).toBe(true);
  expect(events.filter((e) => e === "start")).toHaveLength(5);
  expect(events.filter((e) => e === "error")).toHaveLength(5);
});
