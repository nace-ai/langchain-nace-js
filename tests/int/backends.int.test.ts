/** OpenRouter backend and uploaded-file images, live. Synthetic data only. */
import { readFileSync } from "node:fs";

import { NaceClient, type UploadedFile } from "nace-sdk";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

import {
  NaceAuthenticationError,
  NaceBadRequestError,
  NaceClassifier,
  NaceUnprocessableEntityError,
  type Questions,
} from "../../src/index.js";
import { BASE_URL, BLUE, fixture, KEY, live, OPENROUTER_KEY, RED, WHICH_RED } from "./live.js";

function openrouter(model: string, questions: Questions, apiKey = OPENROUTER_KEY) {
  return new NaceClassifier({
    apiKey,
    baseUrl: "https://openrouter.ai/api/v1",
    model,
    questions,
    maxRetries: 0,
  });
}

describe.skipIf(!OPENROUTER_KEY)("OpenRouter", () => {
  it.each(["nace-ai/drex-v2.0", "nace-ai/drex-max"])("answers text on %s", async (model) => {
    const r = await openrouter(model, {
      refund: { type: "noul", instructions: "Refund request?" },
    }).invoke("I was charged twice, refund please");
    expect(r.nouls.refund!.noul).toBeGreaterThan(0.7);
  });

  it("sends images in the state content array", async () => {
    const r = await openrouter("nace-ai/drex-max", WHICH_RED).invoke("Two images.", {
      images: [BLUE, RED],
    });
    expect(r.choices.red!.choice).toBe("Image 2");
  });

  it("makes OpenRouter's errors readable", async () => {
    const refund: Questions = { q: { type: "noul", instructions: "Refund?" } };
    const fail = (c: NaceClassifier) => c.invoke("refund please").catch((e) => e);
    const badKey = await fail(
      openrouter("nace-ai/drex-v2.0", refund, `sk-or-v1-${"0".repeat(64)}`),
    );
    const unknown = await fail(openrouter("nace-ai/drex-v99.9", refund));
    const tooMany: Questions = {};
    for (let i = 0; i < 17; i += 1)
      tooMany[`q${i}`] = { type: "noul", instructions: `mentions ${i}?` };
    const many = await fail(openrouter("nace-ai/drex-max", tooMany));
    const badType = await fail(
      openrouter("nace-ai/drex-v2.0", { q: { type: "rank", criteria: ["a"] } }),
    );
    expect(badKey).toBeInstanceOf(NaceAuthenticationError);
    expect(unknown).toBeInstanceOf(NaceBadRequestError);
    expect(many).toBeInstanceOf(NaceUnprocessableEntityError);
    expect(badType).toBeInstanceOf(NaceBadRequestError);
    for (const error of [badKey, unknown, many, badType]) {
      expect(error.message).toMatch(/^OpenRouter returned/);
      expect(error.message).not.toContain("HTTP 4");
    }
    expect(unknown.message).toContain("does not exist");
    expect(many.message).toContain("16");
    expect(badType.message).toContain("questions.q.type");
  });
});

describe.skipIf(!KEY)("uploaded files as images", () => {
  let client: NaceClient;
  const files: Record<string, UploadedFile> = {};

  beforeAll(async () => {
    client = new NaceClient({ apiKey: KEY, ...(BASE_URL ? { baseURL: BASE_URL } : {}) });
    const tag = Math.random().toString(16).slice(2, 10);
    for (const color of ["red", "blue"]) {
      files[color] = await client.documents.upload(readFileSync(fixture(`${color}.png`)), {
        file_name: `${tag}-${color}.png`,
        path: `langchain-nace-js-tests/${tag}-${color}.png`,
      });
    }
  });

  afterAll(async () => {
    for (const file of Object.values(files)) {
      await client.documents.deleteFile(file.workspace_id, file.file_id);
    }
  });

  it("reads uploaded files in both orders", async () => {
    const max = live("drex-max", { questions: WHICH_RED });
    const first = await max.invoke("Two uploaded images.", { images: [files.red!, files.blue!] });
    const second = await max.invoke("Two uploaded images.", { images: [files.blue!, files.red!] });
    expect(first.choices.red!.choice).toBe("Image 1");
    expect(second.choices.red!.choice).toBe("Image 2");
  });

  it("reads a workspace_file reference", async () => {
    const blue = files.blue!;
    const ref = { type: "workspace_file", workspace_id: blue.workspace_id, file_id: blue.file_id };
    const r = await live("drex-max", { questions: WHICH_RED }).invoke("Two images.", {
      images: [ref, files.red!],
    });
    expect(r.choices.red!.choice).toBe("Image 2");
  });
});
