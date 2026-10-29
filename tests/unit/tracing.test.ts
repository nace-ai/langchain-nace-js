import { BaseTracer, type Run } from "@langchain/core/tracers/base";
import { describe, expect, it } from "vitest";

import { NaceClassifier } from "../../src/index.js";
import { fakeDrex, offline } from "./helpers.js";

class Collector extends BaseTracer {
  name = "collector";
  runs: Run[] = [];

  protected async persistRun(run: Run): Promise<void> {
    this.runs.push(run);
  }
}

describe("tracing", () => {
  it("records an llm run with the model, answers, usage and request id", async () => {
    const tracer = new Collector();
    const classifier = new NaceClassifier(
      offline(fakeDrex().fetch, { questions: { urgent: { type: "noul", instructions: "?" } } }),
    );
    await classifier.invoke("x", { callbacks: [tracer], metadata: { team: "support" } });
    const run = tracer.runs[0]!;
    expect(run.run_type).toBe("llm");
    expect(run.name).toBe("NaceClassifier");
    expect(run.extra?.metadata).toMatchObject({
      ls_provider: "nace",
      ls_model_name: "drex-max",
      team: "support",
    });
    expect(run.outputs).toMatchObject({
      answers: { urgent: { type: "noul", noul: 0.9 } },
      usage: { inputTokens: 120 },
      requestId: "req_header",
    });
  });
});
