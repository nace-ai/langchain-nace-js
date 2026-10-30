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

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}
const settled = <T>(promise: Promise<T>) =>
  promise.then(
    (value) => ({ value }),
    (error: unknown) => ({ error }),
  );

it.each([false, true])(
  "sends nothing for an input whose start callback outlives a 401 (returnExceptions: %s)",
  async (returnExceptions) => {
    const gate = deferred();
    const bWaiting = deferred();
    const aFailed = deferred();
    const handler = {
      awaitHandlers: true,
      handleChainStart: async (_chain: unknown, run: { input?: unknown }) => {
        if (run.input !== "b") return;
        bWaiting.resolve();
        await gate.promise;
      },
      handleChainError: () => aFailed.resolve(),
    };
    const { server, classifier: c } = classifier([status(401)]);
    const outcome = settled(
      c.batch(["a", "b"], { maxConcurrency: 2, callbacks: [handler] }, {
        returnExceptions,
      } as any) as Promise<any[]>,
    );
    await bWaiting.promise;
    await aFailed.promise;
    await sleep(20);
    gate.resolve();
    const result: any = await outcome;
    await settle();
    expect(server.requests.map((r) => r.body.state)).toEqual(["a"]);
    if (returnExceptions) {
      expect(result.value[0]).toBeInstanceOf(Error);
      expect(result.value[1]).toBeInstanceOf(NaceBatchStoppedError);
    } else expect(String(result.error)).toContain("401");
  },
);

it.each([false, true])(
  "sends nothing while the failed request's error callback runs (returnExceptions: %s)",
  async (returnExceptions) => {
    const many = Array.from({ length: 12 }, (_, i) => `ticket ${i}`);
    const { server, classifier: c } = classifier([
      status(401),
      (body: any) => ({ body: okBody(body), delayMs: 30 }),
    ]);
    let first = true;
    let sentDuringCallback = -1;
    const handler = {
      awaitHandlers: true,
      handleChainError: async () => {
        if (!first) return;
        first = false;
        await sleep(50);
        sentDuringCallback = server.requests.length;
      },
    };
    const outcome = settled(
      c.batch(many, { maxConcurrency: 2, callbacks: [handler] }, { returnExceptions } as any),
    );
    const result: any = await outcome;
    expect(sentDuringCallback).toBe(2);
    await settle();
    expect(server.requests).toHaveLength(2);
    if (returnExceptions) {
      expect(result.value.filter((r: unknown) => r instanceof NaceBatchStoppedError)).toHaveLength(
        10,
      );
    } else expect(String(result.error)).toContain("401");
  },
);

it("keeps cancellation to the input whose signal was aborted", async () => {
  const aborted = AbortSignal.abort();
  // One cancelled input fails alone.
  const one = classifier([]);
  const out = await one.classifier.batch(["a", "b"], [{ maxConcurrency: 1, signal: aborted }, {}], {
    returnExceptions: true,
  });
  expect(out[0]).toBeInstanceOf(Error);
  expect((out[0] as Error).name).toBe("AbortError");
  expect(out[1]).not.toBeInstanceOf(Error);
  expect((out[1] as any).nouls.q).toBeDefined();
  expect(one.server.requests.map((r) => r.body.state)).toEqual(["b"]);
  // A batch-wide cancel sends nothing; every input gets an abort error or a stopped error.
  const whole = classifier([]);
  const all = await whole.classifier.batch(
    ["a", "b", "c"],
    { signal: aborted },
    { returnExceptions: true },
  );
  expect(all.every((r) => r instanceof Error)).toBe(true);
  expect(whole.server.requests).toHaveLength(0);
});

it("gives inputs that were never sent a run when the batch rejects", async () => {
  const gate = deferred();
  const starts: unknown[] = [];
  let errors = 0;
  const handler = {
    awaitHandlers: true,
    handleChainStart: async (_chain: unknown, run: { input?: unknown }) => {
      starts.push(run.input);
      if (run.input === "c" || run.input === "d") await gate.promise;
    },
    handleChainError: () => {
      errors += 1;
    },
  };
  const unhandled: unknown[] = [];
  const onUnhandled = (reason: unknown) => void unhandled.push(reason);
  process.on("unhandledRejection", onUnhandled);
  try {
    const { server, classifier: c } = classifier([status(401), status(401)]);
    const outcome = settled(
      c.batch(["a", "b", "c", "d"], { maxConcurrency: 2, callbacks: [handler] }),
    );
    // The rejection does not wait for the runs of inputs that were not sent.
    const first: any = await Promise.race([outcome, sleep(1000).then(() => ({ late: true }))]);
    expect(first.late).toBeUndefined();
    expect(String(first.error)).toContain("401");
    gate.resolve();
    await settle();
    expect(starts).toHaveLength(4);
    expect(errors).toBe(4);
    expect(unhandled).toEqual([]);
    expect(server.requests).toHaveLength(2);
  } finally {
    process.off("unhandledRejection", onUnhandled);
  }
});
