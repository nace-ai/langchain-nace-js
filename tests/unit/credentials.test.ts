import { createServer } from "node:http";
import { load } from "@langchain/core/load";
import { afterEach, expect, it, vi } from "vitest";
import { NaceClassifier } from "../../src/index.js";
import { fakeDrex } from "./helpers.js";

const questions = { urgent: { type: "noul" as const, instructions: "Urgent?" } };
afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

it.each([
  "https://openrouter.ai/api/v1",
  "https://custom.example",
  "http://console.nace.ai",
  "https://console.nace.ai:444",
  "https://console.nace.ai.evil.example",
  "https://user@console.nace.ai",
])("refuses an ambient Nace key at %s before sending anything", (baseUrl) => {
  vi.stubEnv("NACE_API_KEY", "nace_sk_ambient");
  const server = fakeDrex();
  expect(() => new NaceClassifier({ baseUrl, fetch: server.fetch })).toThrow(/explicit/);
  expect(server.requests).toHaveLength(0);
});

it("refuses a custom environment destination and a blank explicit key", () => {
  vi.stubEnv("NACE_API_KEY", "nace_sk_ambient");
  vi.stubEnv("NACE_BASE_URL", "https://custom.example");
  expect(() => new NaceClassifier()).toThrow(/explicit/);
  expect(() => new NaceClassifier({ apiKey: " " })).toThrow(/empty/);
});

it.each(["console.nace.ai", "drex.nace.ai"])(
  "keeps ambient authentication working at %s",
  async (host) => {
    vi.stubEnv("NACE_API_KEY", "nace_sk_ambient");
    const server = fakeDrex();
    await new NaceClassifier({
      baseUrl: `https://${host}:443`,
      fetch: server.fetch,
      questions,
    }).invoke("x");
    expect(server.requests[0]!.headers.get("authorization")).toBe("Bearer nace_sk_ambient");
  },
);

it.each([
  ["https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"],
  ["https://openrouter.ai./api/v1", "OPENROUTER_API_KEY"],
  ["https://notopenrouter.ai/api/v1", "NACE_CUSTOM_API_KEY"],
  ["https://custom.example", "NACE_CUSTOM_API_KEY"],
])("uses a provider key and secret reference for %s", async (baseUrl, secret) => {
  vi.stubEnv("NACE_API_KEY", "nace_sk_ambient");
  const server = fakeDrex();
  const classifier = new NaceClassifier({
    baseUrl,
    apiKey: "provider-key",
    questions,
    fetch: server.fetch,
    defaultHeaders: {
      "X-Vercel-Protection-Bypass": "internal-token",
      "x-vercel-set-bypass-cookie": "true",
    },
  });
  await classifier.invoke("x");
  expect(server.requests[0]!.headers.get("authorization")).toBe("Bearer provider-key");
  expect(server.requests[0]!.headers.has("x-vercel-protection-bypass")).toBe(false);
  expect(server.requests[0]!.headers.has("x-vercel-set-bypass-cookie")).toBe(false);
  expect(classifier.lc_secrets).toEqual({ apiKey: secret });
  expect(classifier.openrouter).toBe(secret === "OPENROUTER_API_KEY");
  expect(JSON.stringify(classifier.toJSON())).not.toContain("provider-key");
  expect(JSON.stringify(classifier.toJSON())).not.toContain("internal-token");
});

it("pins an environment destination in the serialized constructor", () => {
  vi.stubEnv("NACE_API_KEY", "nace_sk_ambient");
  vi.stubEnv("NACE_BASE_URL", "https://drex.nace.ai");
  const classifier = new NaceClassifier();
  vi.stubEnv("NACE_BASE_URL", "https://custom.example");
  expect(classifier.toJSON()).toMatchObject({
    kwargs: { base_url: "https://drex.nace.ai" },
  });
});

it("reloads a custom provider using its own key even when the Nace key is available", async () => {
  const original = new NaceClassifier({
    baseUrl: "https://openrouter.ai/api/v1",
    apiKey: "original",
    questions,
    openrouter: false,
  });
  const server = fakeDrex();
  vi.stubGlobal("fetch", server.fetch);
  const restored = await load<NaceClassifier>(JSON.stringify(original), {
    secretsMap: { OPENROUTER_API_KEY: "provider-key", NACE_API_KEY: "nace_sk_ambient" },
    importMap: { classifiers__nace: { NaceClassifier } },
  });
  await restored.invoke("x");
  expect(server.requests[0]!.headers.get("authorization")).toBe("Bearer provider-key");
});

it("does not follow redirects carrying private request headers", async () => {
  let redirected = 0;
  const server = createServer((request, response) => {
    if (request.url === "/v1/systemone") {
      response.writeHead(307, { location: "/other" }).end();
    } else {
      redirected += 1;
      response.writeHead(200).end("{}");
    }
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const address = server.address();
    if (!address || typeof address === "string") throw new Error("No server address");
    const classifier = new NaceClassifier({
      baseUrl: `http://127.0.0.1:${address.port}`,
      apiKey: "provider-key",
      questions,
      maxRetries: 0,
    });
    await expect(classifier.invoke("x")).rejects.toThrow();
    expect(redirected).toBe(0);
  } finally {
    server.closeAllConnections();
    await new Promise<void>((resolve, reject) =>
      server.close((error) => (error ? reject(error) : resolve())),
    );
  }
});

it.each([
  ["HTTP-Referer", "X-OpenRouter-Title"],
  ["http-referer", "x-openrouter-title"],
  ["HTTP-REFERER", "X-OPENROUTER-TITLE"],
])("preserves caller attribution headers %s and %s", async (referer, title) => {
  const server = fakeDrex();
  const classifier = new NaceClassifier({
    baseUrl: "https://openrouter.ai/api/v1",
    apiKey: "provider-key",
    questions,
    fetch: server.fetch,
    defaultHeaders: { [referer]: "https://app.example", [title]: "Customer app" },
  });
  await classifier.invoke("x");
  const headers = server.requests[0]!.headers;
  expect(headers.get("http-referer")).toBe("https://app.example");
  expect(headers.get("x-openrouter-title")).toBe("Customer app");
});
