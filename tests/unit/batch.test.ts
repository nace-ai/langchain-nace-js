import { ensureConfig, mergeConfigs, RunnableLambda } from "@langchain/core/runnables";
import { expect, it } from "vitest";
import {
  type ClassificationResponse,
  NaceBatchStoppedError,
  NaceClassifier,
} from "../../src/index.js";
import { fakeDrex, okBody, type Reply } from "./helpers.js";

const questions = { q: { type: "noul" as const, instructions: "Urgent?" } };
const ok = (body: any): Reply => ({ body: okBody(body), delayMs: 5 });
const status = (code: number): Reply => ({
  status: code,
  body: { error: { message: "no" } },
  delayMs: 5,
});
const settle = () => new Promise((resolve) => setTimeout(resolve, 100));

function classifierOf(
  Class: typeof NaceClassifier,
  script: (Reply | ((body: any) => Reply))[],
  extra = {},
) {
  const server = fakeDrex(script);
  return {
    server,
    classifier: new Class({
      apiKey: "nace_sk_test",
      baseUrl: "https://drex.test",
      fetch: server.fetch,
      questions,
      maxRetries: 0,
      ...extra,
    }),
  };
}
const classifier = (script: (Reply | ((body: any) => Reply))[], extra = {}) =>
  classifierOf(NaceClassifier, script, extra);
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

it("rejects at the first error and sends nothing once the batch has rejected", async () => {
  const { server, classifier: c } = classifier([
    ok,
    status(422),
    ...Array.from({ length: 18 }, () => ok),
  ]);
  await expect(c.batch(inputs, { maxConcurrency: 2 })).rejects.toThrow("422");
  // Two inputs in flight, plus the two that took the slots the first two freed.
  const atRejection = server.requests.length;
  expect(atRejection).toBeLessThanOrEqual(4);
  await settle();
  expect(server.requests).toHaveLength(atRejection);
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
    await settle();
    // Read inside the callback, 50 ms after the 401: nothing had been sent since.
    expect(sentDuringCallback).toBe(2);
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

it("stops sending when a start callback that raises rejects the batch", async () => {
  const handler = {
    awaitHandlers: true,
    raiseError: true,
    handleChainStart: async (_chain: unknown, run: { input?: unknown }) => {
      if (run.input === "a") throw new Error("handler failed");
    },
  };
  const strict = classifier([]);
  await expect(
    strict.classifier.batch(["a", "b", "c"], { maxConcurrency: 1, callbacks: [handler] }),
  ).rejects.toThrow("handler failed");
  await settle();
  expect(strict.server.requests).toHaveLength(0);
  // With returnExceptions, only a 401, 402 or 403 stops the batch; this failure stays with its input.
  const lenient = classifier([]);
  const out = await lenient.classifier.batch(
    ["a", "b", "c"],
    { maxConcurrency: 1, callbacks: [handler] },
    { returnExceptions: true },
  );
  expect((out[0] as Error).message).toBe("handler failed");
  expect(lenient.server.requests.map((r) => r.body.state)).toEqual(["b", "c"]);
});

it("stops sending once a cancellation has rejected the batch, while the request still settles", async () => {
  const controller = new AbortController();
  const release = deferred();
  const sent: unknown[] = [];
  const server = fakeDrex([]);
  // A transport that ignores the signal and reports the cancellation late.
  const transport = (async (input: string | URL | Request, init?: RequestInit) => {
    const state = JSON.parse(String(init?.body)).state;
    sent.push(state);
    if (state === "a") {
      await release.promise;
      throw new DOMException("aborted", "AbortError");
    }
    return server.fetch(input, init);
  }) as typeof globalThis.fetch;
  const c = new NaceClassifier({
    apiKey: "nace_sk_test",
    baseUrl: "https://drex.test",
    fetch: transport,
    questions,
    maxRetries: 0,
  });
  const outcome = settled(
    c.batch(["a", "b", "c"], [{ maxConcurrency: 1, signal: controller.signal }, {}, {}]),
  );
  while (sent.length === 0) await sleep(5);
  controller.abort();
  const result: any = await outcome;
  expect(result.error).toBeDefined();
  // "b" can take the slot "a" frees at the cancellation, before the batch has rejected.
  const atRejection = sent.length;
  release.resolve();
  await settle();
  expect(sent).toHaveLength(atRejection);
  expect(sent).not.toContain("c");
});

class Redacting extends NaceClassifier {
  override async invoke(_input: any, options?: any) {
    return super.invoke("redacted", options);
  }
}

it("sends what an overridden invoke sends, for batch as for invoke", async () => {
  const server = fakeDrex([]);
  const c = new Redacting({
    apiKey: "nace_sk_test",
    baseUrl: "https://drex.test",
    fetch: server.fetch,
    questions,
    maxRetries: 0,
  });
  await c.invoke("private");
  await c.batch(["private", "also private"], { maxConcurrency: 2 });
  expect(server.requests.map((r) => r.body.state)).toEqual(["redacted", "redacted", "redacted"]);
});

it("keeps batch bookkeeping out of callback metadata", async () => {
  const seen: Record<string, unknown>[] = [];
  const handler = {
    handleChainStart: (_c: unknown, _i: unknown, _r: unknown, _p: unknown, _t: unknown, m: any) =>
      void seen.push(m),
  };
  const { classifier: c } = classifier([]);
  await c.batch(["a", "b"], { callbacks: [handler], metadata: { mine: 1 } });
  expect(seen).toHaveLength(2);
  for (const metadata of seen) {
    expect(Object.getOwnPropertySymbols(metadata)).toEqual([]);
    expect(Object.keys(metadata).sort()).toEqual([
      "ls_model_name",
      "ls_model_type",
      "ls_provider",
      "mine",
    ]);
  }
});

// Overrides that normalise their options before calling `super.invoke`: LangChain's own
// helpers copy string keys only, so nothing the batch needs may ride on the options.
const normalisers = {
  ensureConfig: (options: any) => ensureConfig(options),
  mergeConfigs: (options: any) => mergeConfigs(options, {}),
};

it.each(
  Object.keys(normalisers).flatMap((name) => [
    [name, false],
    [name, true],
  ]) as [keyof typeof normalisers, boolean][],
)(
  "sends nothing for an input a %s override started before a 401 (returnExceptions: %s)",
  async (name, returnExceptions) => {
    const normalise = normalisers[name];
    class Normalising extends NaceClassifier {
      override async invoke(input: any, options?: any) {
        return super.invoke(input, normalise(options));
      }
    }
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
    const { server, classifier: c } = classifierOf(Normalising, [status(401)]);
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
    if (returnExceptions) expect(result.value[1]).toBeInstanceOf(NaceBatchStoppedError);
    else expect(String(result.error)).toContain("401");
  },
);

it.each([false, true])(
  "shows callbacks what an overridden invoke sends, for inputs not sent too (returnExceptions: %s)",
  async (returnExceptions) => {
    const starts: unknown[] = [];
    const handler = {
      awaitHandlers: true,
      handleChainStart: (_chain: unknown, run: { input?: unknown }) => void starts.push(run.input),
    };
    const { server, classifier: c } = classifierOf(Redacting, [status(401)]);
    await settled(
      c.batch(
        ["private a", "private b", "private c"],
        { maxConcurrency: 1, callbacks: [handler] },
        { returnExceptions } as any,
      ),
    );
    await settle();
    expect(starts).toEqual(["redacted", "redacted", "redacted"]);
    expect(server.requests.map((r) => r.body.state)).toEqual(["redacted"]);
  },
);

it("keeps sending after an override recovers from a failure", async () => {
  const seed = await classifier([]).classifier.invoke("seed");
  class Recovering extends NaceClassifier {
    override async invoke(input: any, options?: any) {
      try {
        return await super.invoke(input, options);
      } catch (error) {
        if ((error as { status?: number }).status === 422) return seed;
        throw error;
      }
    }
  }
  const { server, classifier: c } = classifierOf(Recovering, [status(422)]);
  const out = await c.batch(["a", "b", "c"], { maxConcurrency: 1 });
  expect(out).toHaveLength(3);
  expect(out[0]).toBe(seed);
  expect(out.some((r) => r instanceof Error)).toBe(false);
  expect(server.requests.map((r) => r.body.state)).toEqual(["a", "b", "c"]);
});

it("rejects when an input is cancelled with a stopped error, instead of resolving with errors", async () => {
  const reason = new NaceBatchStoppedError("cancelled by the caller");
  const { classifier: c } = classifier([]);
  const result: any = await settled(
    c.batch(["a", "b"], [{ maxConcurrency: 1, signal: AbortSignal.abort(reason) }, {}]),
  );
  expect(result.value).toBeUndefined();
  expect(result.error).toBe(reason);
});

it.each([
  ["call options", {}, { maxConcurrency: NaN }, undefined],
  ["batch options", {}, undefined, { maxConcurrency: NaN }],
  ["the constructor", { maxConcurrency: NaN }, undefined, undefined],
] as const)(
  "rejects a NaN maxConcurrency from %s like any Runnable",
  async (_source, extra, callOptions, batchOptions) => {
    const base: any = await settled(
      RunnableLambda.from(async (x: string) => x).batch(["a", "b"], { maxConcurrency: NaN }),
    );
    expect(base.error).toBeInstanceOf(TypeError);
    const starts: unknown[] = [];
    const handler = { handleChainStart: (_chain: unknown, run: unknown) => void starts.push(run) };
    const { server, classifier: c } = classifier([], extra);
    const result: any = await settled(
      c.batch(["a", "b"], { ...callOptions, callbacks: [handler] }, batchOptions),
    );
    expect(result.value).toBeUndefined();
    expect(result.error).toBeInstanceOf(TypeError);
    expect(result.error.message).toBe(base.error.message);
    expect(server.requests).toHaveLength(0);
    expect(starts).toHaveLength(0);
  },
);

// An override that answers an input with a batch of its own: the inner batch runs
// inside the outer input's run, so it shares the outer batch's fate.
it.each([false, true])(
  "sends nothing from a nested batch an override starts after the outer batch's 401 (returnExceptions: %s)",
  async (returnExceptions) => {
    const gate = deferred();
    const aFailed = deferred();
    class Nesting extends NaceClassifier {
      override async invoke(input: any, options?: any) {
        if (input !== "b") return super.invoke(input, options);
        await gate.promise;
        const [reply] = await this.batch(["inner-b"]);
        return reply!;
      }
    }
    const handler = { awaitHandlers: true, handleChainError: () => aFailed.resolve() };
    const { server, classifier: c } = classifierOf(Nesting, [status(401)]);
    const outcome = settled(
      c.batch(["a", "b"], { maxConcurrency: 2, callbacks: [handler] }, {
        returnExceptions,
      } as any) as Promise<any[]>,
    );
    await aFailed.promise;
    await sleep(20);
    gate.resolve();
    const result: any = await outcome;
    await settle();
    expect(server.requests.map((r) => r.body.state)).toEqual(["a"]);
    if (returnExceptions) expect(result.value[1]).toBeInstanceOf(Error);
    else expect(String(result.error)).toContain("401");
  },
);

it("stops the outer batch when a batch an override started inside it gets a 401", async () => {
  class Delegating extends NaceClassifier {
    override async invoke(input: any, options?: any) {
      if (input !== "a") return super.invoke(input, options);
      const [reply] = await this.batch(["inner-a"]);
      return reply!;
    }
  }
  const { server, classifier: c } = classifierOf(Delegating, [status(401)]);
  const out = (await c.batch(
    ["a", "b", "c"],
    { maxConcurrency: 1 },
    { returnExceptions: true },
  )) as Error[];
  await settle();
  expect(server.requests.map((r) => r.body.state)).toEqual(["inner-a"]);
  expect(String(out[0])).toContain("401");
  expect(out[1]).toBeInstanceOf(NaceBatchStoppedError);
  expect(out[2]).toBeInstanceOf(NaceBatchStoppedError);
});

it("keeps a nested batch's own non-terminal rejection to the nested batch", async () => {
  const seed = await classifier([]).classifier.invoke("seed");
  class Delegating extends NaceClassifier {
    override async invoke(input: any, options?: any) {
      if (input !== "a") return super.invoke(input, options);
      try {
        const [reply] = await this.batch(["inner-a"]);
        return reply!;
      } catch (error) {
        if ((error as { status?: number }).status === 422) return seed;
        throw error;
      }
    }
  }
  const { server, classifier: c } = classifierOf(Delegating, [status(422)]);
  const out = await c.batch(["a", "b", "c"], { maxConcurrency: 1 });
  expect(out).toHaveLength(3);
  expect(out[0]).toBe(seed);
  expect(server.requests.map((r) => r.body.state)).toEqual(["inner-a", "b", "c"]);
});

it("keeps independently started batches apart", async () => {
  const { server, classifier: c } = classifier([status(401)]);
  const first = await settled(c.batch(["a"]));
  expect(String((first as any).error)).toContain("401");
  // A new batch on the same classifier starts with a clean state.
  const second = await c.batch(["b", "c"], { maxConcurrency: 1 });
  expect(second).toHaveLength(2);
  expect(server.requests.map((r) => r.body.state)).toEqual(["a", "b", "c"]);
});

// A 401, 402 or 403 stops the batch it happens in and the batches nested in it; another
// batch running at the same time stops at its own first one. A background callback
// handler runs on LangChain's shared queue, so it can inherit the stopped state of the
// batch whose callback held the queue: a call made from it is then refused, unsent.
it("refuses, unsent, a call a callback makes while another batch's stopped state holds the shared queue", async () => {
  const gate = deferred();
  const auxDone = deferred();
  let aux: any;
  const holding = {
    handleChainStart: async () => {
      await gate.promise;
    },
  };
  const calling = {
    handleChainEnd: async () => {
      aux = await settled(c.invoke("aux-b"));
      auxDone.resolve();
    },
  };
  const { server, classifier: c } = classifier([status(401)]);
  // The background start callback holds LangChain's shared callback queue.
  const first: any = await settled(c.batch(["a"], { callbacks: [holding] }));
  expect(String(first.error)).toContain("401");
  // A concurrent batch on the same classifier completes; its own inputs are sent.
  const second = await c.batch(["b"], { callbacks: [calling] });
  expect(second).toHaveLength(1);
  expect(aux).toBeUndefined();
  gate.resolve();
  await auxDone.promise;
  expect(aux.error).toBeInstanceOf(NaceBatchStoppedError);
  expect(aux.error.cause).toBe(first.error);
  expect(server.requests.map((r) => r.body.state)).toEqual(["a", "b"]);
});

// Batches that overlap in time: B's start callback holds its inputs back while the
// independently started batch A gets a 401. A's stop does not reach B.
async function overlapping(bReply: (body: any) => Reply, bInputs: string[]) {
  const gate = deferred();
  const entered = deferred();
  const holding = {
    awaitHandlers: true,
    handleChainStart: async () => {
      entered.resolve();
      await gate.promise;
    },
  };
  const answer = (body: any) => (body.state === "a" ? status(401) : bReply(body));
  const { server, classifier: c } = classifier(Array.from({ length: 4 }, () => answer));
  const b = settled(
    c.batch(
      bInputs,
      { maxConcurrency: 1, callbacks: [holding] },
      {
        returnExceptions: true,
      },
    ) as Promise<(ClassificationResponse | Error)[]>,
  );
  await entered.promise;
  const a: any = await settled(c.batch(["a"]));
  expect(String(a.error)).toContain("401");
  expect(server.requests.map((r) => r.body.state)).toEqual(["a"]);
  gate.resolve();
  const out = await b;
  await settle();
  return { out: (out as { value: (ClassificationResponse | Error)[] }).value, server };
}

it("sends an overlapping batch's inputs after another batch's 401", async () => {
  const { out, server } = await overlapping(ok, ["b1", "b2"]);
  expect(server.requests.map((r) => r.body.state)).toEqual(["a", "b1", "b2"]);
  expect(out.filter((r) => r instanceof Error)).toHaveLength(0);
});

it("stops an overlapping batch at its own first 401, not before", async () => {
  const { out, server } = await overlapping(() => status(401), ["b1", "b2", "b3"]);
  expect(server.requests.map((r) => r.body.state)).toEqual(["a", "b1"]);
  expect(String(out[0])).toContain("401");
  expect(out[1]).toBeInstanceOf(NaceBatchStoppedError);
  expect(out[2]).toBeInstanceOf(NaceBatchStoppedError);
  // Their stop names B's own failure, not A's.
  expect((out[1] as Error).cause).toBe(out[0]);
});

it.each([false, true])(
  "stops the batch after an override recovers from a 401 (returnExceptions: %s)",
  async (returnExceptions) => {
    const seed = await classifier([]).classifier.invoke("seed");
    class Recovering extends NaceClassifier {
      override async invoke(input: any, options?: any) {
        try {
          return await super.invoke(input, options);
        } catch (error) {
          if ((error as { status?: number }).status === 401) return seed;
          throw error;
        }
      }
    }
    const { server, classifier: c } = classifierOf(Recovering, [status(401)]);
    const result: any = await settled(
      c.batch(["a", "b"], { maxConcurrency: 1 }, { returnExceptions } as any) as Promise<any[]>,
    );
    await settle();
    expect(server.requests.map((r) => r.body.state)).toEqual(["a"]);
    if (returnExceptions) {
      expect(result.value[0]).toBe(seed);
      expect(result.value[1]).toBeInstanceOf(NaceBatchStoppedError);
    } else expect(String(result.error)).toContain("401");
  },
);
