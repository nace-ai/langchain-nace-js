/** NDI components against a fake document service. All ids, keys and hosts are fake. */
import { fileURLToPath } from "node:url";
import { inspect } from "node:util";
import { afterEach, expect, it, vi } from "vitest";

import {
  NaceAPIConnectionError,
  type NaceAPIError,
  NaceAPITimeoutError,
  NaceAuthenticationError,
  NaceDocumentLoader,
  NaceError,
  NaceExtractTool,
  NaceInternalServerError,
  NaceSearchTool,
  NaceUnprocessableEntityError,
} from "../../src/index.js";
import { NdiClient } from "../../src/ndi/client.js";

const INVOICE = fileURLToPath(new URL("../fixtures/invoice.html", import.meta.url));

/** Fake ids and hosts: UUIDs for job, workspace and file ids. */
const WS = "11111111-1111-4111-8111-111111111111";
const FILE = "22222222-2222-4222-8222-222222222222";
const JOB = "33333333-3333-4333-8333-333333333333";
const UPLOAD_JOB = "44444444-4444-4444-8444-444444444444";
const REFUSED_JOB = "55555555-5555-4555-8555-555555555555";
const UPLOAD_URL = "https://upload.ndi.test/upload";

const MARKDOWN = "# Invoice\n\nTotal due: USD 400.00";
const FULL_MARKDOWN =
  "# Invoice INV-2026-0042\n\nTotal due: USD 400.00.\n\nSeller: Example Supplies Ltd.";

function parseResult() {
  return { document: { markdown: MARKDOWN, page_count: 2 }, disclosures: [] };
}

function extractResult() {
  return {
    data: { invoice_number: "INV-2026-0042", total: 400 },
    fields: [
      {
        path: "/invoice_number",
        value: "INV-2026-0042",
        status: "found",
        confidence: 0.99,
        citations: [{ location: { page: 1 }, source_text: "Invoice INV-2026-0042" }],
      },
      { path: "/po_number", value: null, status: "not_found", confidence: 0.9, citations: [] },
    ],
  };
}

const SEARCH_FACT = {
  answer: "The total due is USD 400.00.",
  confidence: "certain",
  evidences: [{ source_path: "invoice.html", page: 1, quote: "Total due: USD 400.00" }],
};

/** One request the fake NDI received. */
interface Recorded {
  method: string;
  url: string;
  headers: Headers;
  body: any;
}

/** A scripted response: status, headers and a JSON body, or raw `text`. */
interface Reply {
  status?: number;
  headers?: Record<string, string>;
  body?: unknown;
  text?: string;
  /** A raw body (for example a stream that never ends), used instead of `text` and `body`. */
  raw?: BodyInit;
  /** Wait this many milliseconds before answering. */
  delayMs?: number;
}

type Route =
  | "grant"
  | "upload"
  | "parse"
  | "extract"
  | "search"
  | "getJob"
  | "deleteJob"
  | "deleteFile"
  | "download";

function job(kind: string, status: string, result?: unknown, error?: unknown, jobId = JOB) {
  return {
    job_id: jobId,
    kind,
    status,
    ...(result !== undefined ? { result } : {}),
    ...(error !== undefined ? { error } : {}),
    created_at: "2026-11-20T10:00:00Z",
  };
}

function routeOf(method: string, pathname: string): Route {
  if (method === "POST" && pathname === "/v1/documents/upload-grants") return "grant";
  if (method === "POST" && pathname === "/upload") return "upload";
  if (method === "POST" && pathname === "/v1/documents/parse") return "parse";
  if (method === "POST" && pathname === "/v1/documents/extract") return "extract";
  if (method === "POST" && /^\/v1\/workspaces\/[^/]+\/automatic-search$/.test(pathname)) {
    return "search";
  }
  const jobs = /^\/v1\/documents\/jobs\/[0-9a-f-]+(\/.+)?$/.exec(pathname);
  if (jobs) {
    if (jobs[1]) return "download";
    if (method === "GET") return "getJob";
    if (method === "DELETE") return "deleteJob";
  }
  if (method === "DELETE" && /^\/v1\/documents\/workspaces\/[^/]+\/files\/[^/]+$/.test(pathname)) {
    return "deleteFile";
  }
  throw new Error(`Unexpected NDI request: ${method} ${pathname}`);
}

/** A fake NDI: a `fetch` that records each request and routes on method and path. */
function fakeNdi(overrides: Partial<Record<Route, () => Reply>> = {}) {
  const requests: Recorded[] = [];
  const defaults: Record<Route, () => Reply> = {
    grant: () => ({
      body: {
        workspace_id: WS,
        upload_url: UPLOAD_URL,
        token: "tok_test",
        expires_at: "2026-11-20T11:00:00Z",
        max_bytes: null,
      },
    }),
    upload: () => ({
      body: job(
        "file_upload",
        "succeeded",
        {
          file: { workspace_id: WS, file_id: FILE, path: "langchain-nace/x/invoice.html" },
        },
        undefined,
        UPLOAD_JOB,
      ),
    }),
    parse: () => ({ body: job("parse", "succeeded", parseResult()) }),
    extract: () => ({ body: job("extract", "succeeded", extractResult()) }),
    search: () => ({
      body: job("automatic_search", "succeeded", {
        result_type: "automatic_search",
        output: SEARCH_FACT,
      }),
    }),
    getJob: () => ({ body: job("parse", "succeeded", parseResult()) }),
    deleteJob: () => ({ status: 204 }),
    deleteFile: () => ({
      body: { object: "file_deletion", workspace_id: WS, file_id: FILE, status: "deleted" },
    }),
    download: () => ({ headers: { "content-type": "text/markdown" }, text: FULL_MARKDOWN }),
  };
  const fetch = async (input: string | URL | Request, init?: RequestInit): Promise<Response> => {
    const method = init?.method ?? "GET";
    const url = new URL(String(input));
    let body: any;
    if (typeof init?.body === "string") {
      try {
        body = JSON.parse(init.body);
      } catch {
        body = init.body;
      }
    } else if (init?.body instanceof FormData) {
      body = {};
      for (const [key, value] of init.body.entries()) {
        if (typeof value === "string") {
          try {
            body[key] = JSON.parse(value);
          } catch {
            body[key] = value;
          }
        } else if (value instanceof File) body[key] = value.name;
        else body[key] = String(value);
      }
    }
    const request: Recorded = {
      method,
      url: url.href,
      headers: new Headers(init?.headers),
      body,
    };
    requests.push(request);
    const reply = (
      overrides[routeOf(method, url.pathname)] ?? defaults[routeOf(method, url.pathname)]
    )();
    if (reply.delayMs !== undefined) {
      await new Promise((resolve) => setTimeout(resolve, reply.delayMs));
    }
    const status = reply.status ?? 200;
    const headers = {
      "content-type": "application/json",
      "x-request-id": "req_test",
      ...reply.headers,
    };
    if (status === 204) return new Response(null, { status, headers });
    const payload = reply.raw ?? reply.text ?? JSON.stringify(reply.body ?? {});
    return new Response(payload, { status, headers });
  };
  const find = (part: string): Recorded => {
    const request = requests.find((r) => r.url.includes(part));
    if (!request) throw new Error(`No request to ${part}`);
    return request;
  };
  const sent = (method: string, part: string): boolean =>
    requests.some((r) => r.method === method && r.url.includes(part));
  return { fetch: fetch as typeof globalThis.fetch, requests, find, sent };
}

/** Options for an NDI component that talks to the fake. */
function offline<T extends object>(fetch: typeof globalThis.fetch, extra: T = {} as T) {
  return { apiKey: "nace_sk_test", baseUrl: "https://ndi.test", fetch, ...extra };
}

/**
 * A fetch that stops like a real one when the request's signal aborts: the
 * fake itself ignores signals, so tests of deadline aborts wrap it in this.
 */
function abortable(inner: typeof globalThis.fetch): typeof globalThis.fetch {
  return async (input: string | URL | Request, init?: RequestInit) => {
    const signal = init?.signal;
    if (signal == null) return inner(input, init);
    const abort = () => new DOMException("This operation was aborted", "AbortError");
    const aborted = new Promise<never>((_, reject) => {
      if (signal.aborted) reject(abort());
      else signal.addEventListener("abort", () => reject(abort()), { once: true });
    });
    // The race loser must not surface as an unhandled rejection.
    aborted.catch(() => {});
    return Promise.race([inner(input, init), aborted]);
  };
}

afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

it("loads a local file into one document and deletes the job and the file", async () => {
  const ndi = fakeNdi();
  const loader = new NaceDocumentLoader(offline(ndi.fetch, { source: INVOICE }));
  const documents = await loader.load();
  expect(documents).toHaveLength(1);
  expect(documents[0]!.pageContent).toBe(MARKDOWN);
  expect(documents[0]!.metadata).toEqual({ source: INVOICE, job_id: JOB, page_count: 2 });

  const upload = ndi.find(UPLOAD_URL);
  expect(upload.method).toBe("POST");
  expect(upload.headers.get("x-upload-token")).toBe("tok_test");
  expect(upload.headers.has("authorization")).toBe(false);
  for (const request of ndi.requests) {
    if (request.url.startsWith("https://ndi.test")) {
      expect(request.headers.get("authorization")).toBe("Bearer nace_sk_test");
    }
  }
  const parse = ndi.find("/v1/documents/parse");
  expect(parse.url).toBe("https://ndi.test/v1/documents/parse?wait_seconds=60");
  expect(ndi.sent("DELETE", `/v1/documents/jobs/${JOB}`)).toBe(true);
  expect(ndi.sent("DELETE", `/v1/documents/workspaces/${WS}/files/${FILE}`)).toBe(true);
});

it("chunks a URL source into one document per chunk and uploads nothing", async () => {
  const ndi = fakeNdi({
    parse: () => ({ status: 202, body: job("parse", "queued") }),
    getJob: () => ({
      body: job("parse", "succeeded", {
        document: {
          chunks: [
            { id: "chunk_1", content: "# Invoice", location: { kind: "page", page: 1 } },
            { id: "chunk_2", content: "Total: USD 400.00", location: { kind: "page", page: null } },
          ],
          page_count: 2,
        },
      }),
    }),
  });
  const loader = new NaceDocumentLoader(
    offline(ndi.fetch, { source: "https://example.com/invoice.html", chunking: "page" }),
  );
  const documents = await loader.load();
  expect(documents).toHaveLength(2);
  expect(documents[0]!.pageContent).toBe("# Invoice");
  expect(documents[0]!.metadata).toEqual({
    source: "https://example.com/invoice.html",
    job_id: JOB,
    page_count: 2,
    chunk_id: "chunk_1",
    page: 1,
  });
  expect(documents[1]!.metadata).toMatchObject({ chunk_id: "chunk_2" });
  expect(documents[1]!.metadata).not.toHaveProperty("page");
  expect(ndi.requests.some((r) => r.url.includes("upload"))).toBe(false);
  const parse = ndi.find("/v1/documents/parse");
  expect(parse.body.source).toEqual({
    type: "url",
    url: "https://example.com/invoice.html",
    file_name: "invoice.html",
  });
  expect(parse.body.chunking).toEqual({ strategy: "page" });
  expect(parse.body.parse_mode).toBe("low");
  expect(parse.body.output).toEqual({ formats: ["markdown"] });
});

it("sends parseKwargs without letting them replace chunking, source or wait_seconds", async () => {
  const ndi = fakeNdi();
  const loader = new NaceDocumentLoader(
    offline(ndi.fetch, {
      source: "https://example.com/invoice.pdf",
      parseKwargs: {
        wait_seconds: 0,
        ocr: { mode: "force" },
        source: { type: "url", url: "https://other.example/x.pdf", file_name: "x.pdf" },
        chunking: { strategy: "section" },
      },
    }),
  );
  await loader.load();
  const parse = ndi.find("/v1/documents/parse");
  expect(parse.url).toBe("https://ndi.test/v1/documents/parse?wait_seconds=60");
  expect(parse.body.ocr).toEqual({ mode: "force" });
  expect(parse.body.source.url).toBe("https://example.com/invoice.pdf");
  expect(parse.body.chunking).toEqual({ strategy: "none" });
});

it("keeps a parseKwargs password out of the loader's serialization", async () => {
  const ndi = fakeNdi();
  const loader = new NaceDocumentLoader(
    offline(ndi.fetch, {
      source: "https://example.com/locked.pdf",
      parseKwargs: { password: "fake-pdf-password", page_ranges: [{ start: 1, end: 2 }] },
    }),
  );
  // `parseKwargs` can hold secrets (a PDF's `password`): serializing or
  // inspecting the loader must not show them.
  expect(JSON.stringify(loader)).not.toContain("fake-pdf-password");
  expect(inspect(loader, { depth: 5 })).not.toContain("fake-pdf-password");
  // The field stays readable, and the parse request still carries the password.
  expect(loader.parseKwargs).toEqual({
    password: "fake-pdf-password",
    page_ranges: [{ start: 1, end: 2 }],
  });
  await loader.load();
  const parse = ndi.find("/v1/documents/parse");
  expect(parse.body.password).toBe("fake-pdf-password");
});

it("accepts an uppercase HTTPS scheme and sends it as a url source", async () => {
  const ndi = fakeNdi();
  const loader = new NaceDocumentLoader(
    offline(ndi.fetch, { source: "HTTPS://example.com/a.pdf" }),
  );
  await loader.load();
  const parse = ndi.find("/v1/documents/parse");
  expect(parse.body.source).toEqual({
    type: "url",
    url: "HTTPS://example.com/a.pdf",
    file_name: "a.pdf",
  });
  expect(ndi.requests.some((r) => r.url.includes("upload"))).toBe(false);
});

it("strips the query of a signed source URL from the document metadata", async () => {
  const signed = "https://files.example/report.pdf?sig=fake-signature&expires=123";
  const ndi = fakeNdi();
  const loader = new NaceDocumentLoader(offline(ndi.fetch, { source: signed }));
  // The loader's own enumerable `source` field keeps the URL without the
  // query: serializing the loader must not leak the signature.
  expect(loader.source).toBe("https://files.example/report.pdf");
  expect(JSON.stringify(loader)).not.toContain("fake-signature");
  expect(inspect(loader, { depth: 5 })).not.toContain("fake-signature");
  const documents = await loader.load();
  // The query of a signed link is a credential; it must not reach the metadata.
  expect(documents[0]!.metadata.source).toBe("https://files.example/report.pdf");
  expect(JSON.stringify(documents[0])).not.toContain("fake-signature");
  // NDI itself still gets the full signed URL, or it could not fetch the file.
  const parse = ndi.find("/v1/documents/parse");
  expect(parse.body.source.url).toBe(signed);
});

it.each([
  // A path with a space: the URL class redacts it (and percent-encodes the
  // space), where a pattern that stops at whitespace would keep the query.
  [
    "https://files.example/my report.pdf?sig=fake-signature&expires=123",
    "https://files.example/my%20report.pdf",
  ],
  // Credentials in the URL itself, and a fragment: both dropped.
  [
    "https://user:fake-user@files.example/report.pdf?sig=fake-signature#page=2",
    "https://files.example/report.pdf",
  ],
])("redacts a signed source URL with unusual characters: %s", async (signed, shown) => {
  const ndi = fakeNdi();
  const loader = new NaceDocumentLoader(offline(ndi.fetch, { source: signed }));
  expect(loader.source).toBe(shown);
  expect(JSON.stringify(loader)).not.toContain("fake-signature");
  expect(JSON.stringify(loader)).not.toContain("fake-user");
  const documents = await loader.load();
  expect(documents[0]!.metadata.source).toBe(shown);
  // NDI itself still gets the URL exactly as given.
  const parse = ndi.find("/v1/documents/parse");
  expect(parse.body.source.url).toBe(signed);
});

it("falls back to text when the parse output format has no markdown", async () => {
  const ndi = fakeNdi({
    parse: () => ({
      body: job("parse", "succeeded", {
        document: {
          markdown: null,
          text: "Invoice INV-2026-0042\nTotal due: USD 400.00",
          page_count: 2,
        },
        disclosures: [],
      }),
    }),
  });
  const loader = new NaceDocumentLoader(
    offline(ndi.fetch, {
      source: "https://example.com/invoice.pdf",
      parseKwargs: { output: { formats: ["text"] } },
    }),
  );
  const documents = await loader.load();
  expect(documents).toHaveLength(1);
  expect(documents[0]!.pageContent).toBe("Invoice INV-2026-0042\nTotal due: USD 400.00");
  expect(documents[0]!.metadata).toEqual({
    source: "https://example.com/invoice.pdf",
    job_id: JOB,
    page_count: 2,
  });
});

it("refuses a parse result with neither markdown nor text, and deletes the job", async () => {
  const ndi = fakeNdi({
    parse: () => ({
      body: job("parse", "succeeded", {
        document: { markdown: null, text: null, page_count: 2 },
      }),
    }),
  });
  const loader = new NaceDocumentLoader(
    offline(ndi.fetch, {
      source: "https://example.com/invoice.pdf",
      parseKwargs: { output: { formats: ["blocks"] } },
    }),
  );
  await expect(loader.load()).rejects.toThrow(/no `markdown` and no `text`/);
  expect(ndi.sent("DELETE", `/v1/documents/jobs/${JOB}`)).toBe(true);
});

it("downloads the full markdown when the parse result is truncated", async () => {
  const ndi = fakeNdi({
    parse: () => ({
      body: job("parse", "succeeded", {
        document: {
          markdown: "# Invoice INV-2026-0042 (truncated)",
          content_url: `https://ndi.test/v1/documents/jobs/${JOB}/document-content`,
          page_count: 50,
        },
        disclosures: [{ code: "content_truncated" }],
      }),
    }),
  });
  const loader = new NaceDocumentLoader(
    offline(ndi.fetch, { source: "https://example.com/report.pdf" }),
  );
  const documents = await loader.load();
  expect(documents).toHaveLength(1);
  expect(documents[0]!.pageContent).toBe(FULL_MARKDOWN);
  const download = ndi.find("/document-content");
  expect(download.method).toBe("GET");
  expect(download.url).toContain("redirect=false");
  expect(download.headers.get("authorization")).toBe("Bearer nace_sk_test");
});

it("reports the timeout when a poll answers after it, and deletes the job", async () => {
  const ndi = fakeNdi({
    parse: () => ({ status: 202, body: job("parse", "queued") }),
    // The job wait checks its deadline only after each poll, so this success
    // arrives after the configured timeout.
    getJob: () => ({ body: job("parse", "succeeded", parseResult()), delayMs: 40 }),
  });
  const loader = new NaceDocumentLoader(
    offline(ndi.fetch, { source: "https://example.com/invoice.pdf", timeout: 5 }),
  );
  let error: unknown;
  try {
    await loader.load();
  } catch (caught) {
    error = caught;
  }
  expect(error).toBeInstanceOf(NaceAPITimeoutError);
  expect((error as Error).message).toMatch(/did not finish in 5 ms/);
  expect((error as NaceAPITimeoutError).timeoutMs).toBe(5);
  expect(ndi.sent("DELETE", `/v1/documents/jobs/${JOB}`)).toBe(true);
});

it("stops a truncated-result download that outlasts the timeout, and deletes the job", async () => {
  const ndi = fakeNdi({
    parse: () => ({
      body: job("parse", "succeeded", {
        document: {
          markdown: "# Invoice (truncated)",
          content_url: `https://ndi.test/v1/documents/jobs/${JOB}/document-content`,
          page_count: 50,
        },
        disclosures: [{ code: "content_truncated" }],
      }),
    }),
    // The download holds no deadline of its own: this body never ends.
    download: () => ({
      headers: { "content-type": "text/markdown" },
      raw: new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode("# partial"));
        },
      }),
    }),
  });
  const loader = new NaceDocumentLoader(
    offline(ndi.fetch, { source: "https://example.com/report.pdf", timeout: 50 }),
  );
  let error: unknown;
  try {
    await loader.load();
  } catch (caught) {
    error = caught;
  }
  expect(error).toBeInstanceOf(NaceAPITimeoutError);
  expect((error as Error).message).toMatch(/did not finish in 50 ms/);
  expect(ndi.sent("DELETE", `/v1/documents/jobs/${JOB}`)).toBe(true);
});

it("keeps the truncated text preview instead of downloading Markdown over it", async () => {
  const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
  const ndi = fakeNdi({
    parse: () => ({
      body: job("parse", "succeeded", {
        document: {
          markdown: null,
          text: "Text preview cut short by NDI",
          content_url: `https://ndi.test/v1/documents/jobs/${JOB}/document-content`,
          page_count: 50,
        },
        disclosures: [{ code: "content_truncated" }],
      }),
    }),
  });
  const loader = new NaceDocumentLoader(
    offline(ndi.fetch, {
      source: "https://example.com/report.pdf",
      parseKwargs: { output: { formats: ["text"] } },
    }),
  );
  const documents = await loader.load();
  // The download carries the full Markdown: it must not replace the text the
  // caller asked for. The preview stays, with a warning.
  expect(documents[0]!.pageContent).toBe("Text preview cut short by NDI");
  expect(ndi.requests.some((r) => r.url.includes("/document-content"))).toBe(false);
  expect(warn).toHaveBeenCalledTimes(1);
  expect(warn.mock.calls[0]![0]).toMatch(/full content downloads as Markdown/);
});

it("stops the create at the deadline, so no job exists to delete", async () => {
  const ndi = fakeNdi({
    // The create is still in flight when the deadline arrives.
    parse: () => ({ body: job("parse", "succeeded", parseResult()), delayMs: 100 }),
  });
  const loader = new NaceDocumentLoader(
    offline(abortable(ndi.fetch), {
      source: "https://example.com/invoice.pdf",
      timeout: 20,
      maxRetries: 0,
    }),
  );
  let error: unknown;
  try {
    await loader.load();
  } catch (caught) {
    error = caught;
  }
  expect(error).toBeInstanceOf(NaceAPITimeoutError);
  expect((error as Error).message).toMatch(/did not finish in 20 ms/);
  // The create was cut off before it made a job: nothing to delete.
  expect(ndi.requests.some((r) => r.method === "DELETE")).toBe(false);
});

it("stops a poll that is still in flight at the deadline", async () => {
  const ndi = fakeNdi({
    parse: () => ({ status: 202, body: job("parse", "queued") }),
    // A slow poll: without the deadline signal, the SDK sits out its answer,
    // hundreds of milliseconds past the deadline.
    getJob: () => ({ body: job("parse", "queued"), delayMs: 400 }),
  });
  const loader = new NaceDocumentLoader(
    offline(abortable(ndi.fetch), {
      source: "https://example.com/invoice.pdf",
      timeout: 20,
      maxRetries: 0,
    }),
  );
  const began = Date.now();
  let error: unknown;
  try {
    await loader.load();
  } catch (caught) {
    error = caught;
  }
  expect(error).toBeInstanceOf(NaceAPITimeoutError);
  expect((error as Error).message).toMatch(/did not finish in 20 ms/);
  // The load settles at the deadline, not when the slow poll answers, and no
  // second poll is attempted.
  expect(Date.now() - began).toBeLessThan(150);
  const polls = ndi.requests.filter(
    (r) => r.method === "GET" && r.url.includes(`/v1/documents/jobs/${JOB}`),
  );
  expect(polls).toHaveLength(1);
  expect(ndi.sent("DELETE", `/v1/documents/jobs/${JOB}`)).toBe(true);
});

it.each(["none", "page"] as const)(
  "refuses a truncated result with no content to download (chunking %s), and deletes the job",
  async (chunking) => {
    const ndi = fakeNdi({
      parse: () => ({
        body: job("parse", "succeeded", {
          document: { markdown: MARKDOWN, chunks: [], page_count: 3 },
          disclosures: [{ code: "content_truncated" }],
        }),
      }),
    });
    const loader = new NaceDocumentLoader(
      offline(ndi.fetch, { source: "https://example.com/ledger.xlsx", chunking }),
    );
    await expect(loader.load()).rejects.toThrow(/over 1 MiB/);
    expect(ndi.sent("DELETE", `/v1/documents/jobs/${JOB}`)).toBe(true);
  },
);

it("reports a failed job and still deletes it and the uploaded file", async () => {
  const ndi = fakeNdi({
    parse: () => ({
      body: job("parse", "failed", undefined, {
        code: "parse_failed",
        message:
          "The source URL answered 404 Not Found for https://files.example/a.pdf?sig=fake-signature",
      }),
    }),
  });
  const loader = new NaceDocumentLoader(offline(ndi.fetch, { source: INVOICE }));
  let error: unknown;
  try {
    await loader.load();
  } catch (caught) {
    error = caught;
  }
  expect(error).toBeInstanceOf(NaceError);
  expect((error as Error).message).toContain("parse_failed");
  expect((error as Error).message).toContain(JOB);
  // The query of a URL that NDI repeats can be a credential; it is dropped.
  expect((error as Error).message).toContain("https://files.example/a.pdf");
  expect((error as Error).message).not.toContain("fake-signature");
  expect(ndi.sent("DELETE", `/v1/documents/jobs/${JOB}`)).toBe(true);
  expect(ndi.sent("DELETE", `/v1/documents/workspaces/${WS}/files/${FILE}`)).toBe(true);
});

it("strips the query of a URL that an NDI error repeats", async () => {
  const ndi = fakeNdi({
    parse: () => ({
      status: 422,
      body: {
        error: {
          type: "invalid_request_error",
          code: "invalid_source",
          message: "Could not fetch https://files.example/a.pdf?sig=fake-signature",
        },
      },
    }),
  });
  const loader = new NaceDocumentLoader(
    offline(ndi.fetch, { source: "https://example.com/invoice.pdf" }),
  );
  let error: unknown;
  try {
    await loader.load();
  } catch (caught) {
    error = caught;
  }
  expect(error).toBeInstanceOf(NaceUnprocessableEntityError);
  expect((error as Error).message).toContain("https://files.example/a.pdf");
  expect((error as Error).message).not.toContain("fake-signature");
});

it("strips the query of a signed URL with spaces that an NDI error repeats", async () => {
  const ndi = fakeNdi({
    parse: () => ({
      body: job("parse", "failed", undefined, {
        code: "parse_failed",
        message:
          "The source URL answered 404 Not Found for " +
          "https://files.example/my report.pdf?sig=fake-signature&expires=123",
      }),
    }),
  });
  const loader = new NaceDocumentLoader(
    offline(ndi.fetch, { source: "https://example.com/invoice.pdf" }),
  );
  let error: unknown;
  try {
    await loader.load();
  } catch (caught) {
    error = caught;
  }
  expect(error).toBeInstanceOf(NaceError);
  // A pattern that stops at whitespace would keep the query of a URL whose
  // path has a space; the URL-class redaction removes it.
  expect((error as Error).message).not.toContain("fake-signature");
  expect((error as Error).message).not.toContain("expires=123");
  expect((error as Error).message).toContain("my%20report.pdf");
});

it("scrubs a URL query from the whole cause chain of a connection error", async () => {
  const ndi = fakeNdi();
  const failing = async (input: string | URL | Request, init?: RequestInit): Promise<Response> => {
    if (new URL(String(input)).pathname === "/v1/documents/parse") {
      throw new TypeError("fetch failed for HTTPS://files.example/a.pdf?sig=fake-signature");
    }
    return ndi.fetch(input, init);
  };
  const loader = new NaceDocumentLoader(
    offline(failing, { source: "https://example.com/invoice.pdf", maxRetries: 0 }),
  );
  let error: unknown;
  try {
    await loader.load();
  } catch (caught) {
    error = caught;
  }
  expect(error).toBeInstanceOf(NaceAPIConnectionError);
  expect((error as Error).message).toContain("HTTPS://files.example/a.pdf");
  expect((error as Error).message).not.toContain("fake-signature");
  expect(inspect(error, { depth: 6 })).not.toContain("fake-signature");
});

it("deletes the job id a refused create kept, given in the response header", async () => {
  const ndi = fakeNdi({
    parse: () => ({
      status: 422,
      headers: { "x-drex-document-job": REFUSED_JOB },
      body: { error: { code: "invalid_source", message: "The source URL answered 404." } },
    }),
  });
  const loader = new NaceDocumentLoader(
    offline(ndi.fetch, { source: "https://example.com/invoice.pdf" }),
  );
  await expect(loader.load()).rejects.toBeInstanceOf(NaceUnprocessableEntityError);
  expect(ndi.sent("DELETE", `/v1/documents/jobs/${REFUSED_JOB}`)).toBe(true);
});

it("reports a succeeded job whose result was not retained, and deletes it", async () => {
  const ndi = fakeNdi({
    parse: () => ({ status: 202, body: job("parse", "queued") }),
    getJob: () => ({
      body: { ...job("parse", "succeeded", null), result_state: "not_retained" },
    }),
  });
  const loader = new NaceDocumentLoader(
    offline(ndi.fetch, { source: "https://example.com/invoice.pdf" }),
  );
  let error: unknown;
  try {
    await loader.load();
  } catch (caught) {
    error = caught;
  }
  expect(error).toBeInstanceOf(NaceError);
  expect((error as Error).message).toMatch(/not_retained/);
  expect(ndi.sent("DELETE", `/v1/documents/jobs/${JOB}`)).toBe(true);
});

it("turns a 401 into NaceAuthenticationError without echoing the key", async () => {
  const ndi = fakeNdi({
    parse: () => ({
      status: 401,
      // A provider can echo the bearer key it was sent back in the message.
      body: {
        error: {
          message: "Invalid API key: Bearer nace_sk_test is not a valid key.",
          type: "authentication_error",
        },
      },
    }),
  });
  const loader = new NaceDocumentLoader(
    offline(ndi.fetch, { source: "https://example.com/invoice.pdf" }),
  );
  let error: unknown;
  try {
    await loader.load();
  } catch (caught) {
    error = caught;
  }
  expect(error).toBeInstanceOf(NaceAuthenticationError);
  expect((error as Error).message).toMatch(/^NDI returned 401/);
  expect((error as Error).message).not.toContain("nace_sk_test");
  // The echoed key must not survive the error, its cause chain, or a log of it.
  expect(inspect(error, { depth: 6 })).not.toContain("nace_sk_test");
  // No job existed and the refusal carried no id to delete, so nothing is deleted.
  expect(ndi.requests.some((request) => request.method === "DELETE")).toBe(false);
});

it.each(["https://openrouter.ai/api/v1", "https://OPENROUTER.AI./api/v1"])(
  "refuses an OpenRouter base URL %s before any request",
  (baseUrl) => {
    const ndi = fakeNdi();
    expect(() => new NaceDocumentLoader(offline(ndi.fetch, { source: INVOICE, baseUrl }))).toThrow(
      /not available through OpenRouter/,
    );
    expect(ndi.requests).toHaveLength(0);
  },
);

it("refuses an OpenRouter endpoint from NACE_BASE_URL before any request", () => {
  vi.stubEnv("NACE_API_KEY", "nace_sk_ambient");
  vi.stubEnv("NACE_BASE_URL", "https://openrouter.ai/api/v1");
  const ndi = fakeNdi();
  expect(() => new NaceDocumentLoader({ source: INVOICE, fetch: ndi.fetch })).toThrow(
    /not available through OpenRouter/,
  );
  expect(ndi.requests).toHaveLength(0);
});

it("refuses a custom baseUrl with only the ambient Nace key", () => {
  vi.stubEnv("NACE_API_KEY", "nace_sk_ambient");
  const ndi = fakeNdi();
  expect(
    () =>
      new NaceDocumentLoader({
        source: INVOICE,
        baseUrl: "https://custom.example",
        fetch: ndi.fetch,
      }),
  ).toThrow(/explicit `apiKey`/);
  expect(ndi.requests).toHaveLength(0);
});

it("pairs NACE_API_KEY with NACE_BASE_URL", async () => {
  vi.stubEnv("NACE_API_KEY", "nace_sk_onprem");
  vi.stubEnv("NACE_BASE_URL", "https://onprem.test");
  const ndi = fakeNdi();
  const loader = new NaceDocumentLoader({
    source: "https://example.com/invoice.pdf",
    fetch: ndi.fetch,
  });
  await loader.load();
  const parse = ndi.find("/v1/documents/parse");
  expect(parse.url).toBe("https://onprem.test/v1/documents/parse?wait_seconds=60");
  expect(parse.headers.get("authorization")).toBe("Bearer nace_sk_onprem");
});

it.each([
  ["http://example.com/invoice.pdf", /Only https:\/\/ URLs are supported\./],
  ["https://example.com/reports/latest", /NaceDocumentLoader takes the name as fileName/],
])("refuses the source %s before any request", async (source, message) => {
  const ndi = fakeNdi();
  const loader = new NaceDocumentLoader(offline(ndi.fetch, { source }));
  await expect(loader.load()).rejects.toThrow(message);
  expect(ndi.requests).toHaveLength(0);
});

const EXTRACT_SCHEMA = {
  type: "object",
  properties: {
    invoice_number: { type: "string" },
    total: { type: "number" },
    po_number: { type: "string" },
  },
};

it("extract tool returns data, field statuses and quotes, and deletes the job", async () => {
  const ndi = fakeNdi();
  const extract = new NaceExtractTool(
    offline(ndi.fetch, {
      extractionSchema: EXTRACT_SCHEMA,
      instructions: "Extract the invoice fields.",
    }),
  );
  const message = await extract.invoke({
    name: "nace_extract",
    args: { source: "https://example.com/invoice.pdf" },
    id: "call_1",
    type: "tool_call",
  });
  const parsed = JSON.parse(message.content as string);
  expect(parsed.data).toEqual({ invoice_number: "INV-2026-0042", total: 400 });
  expect(parsed.fields[0]).toMatchObject({
    path: "/invoice_number",
    status: "found",
    quotes: ["Invoice INV-2026-0042"],
  });
  expect(parsed.fields[1]).toMatchObject({ path: "/po_number", status: "not_found", quotes: [] });
  expect(message.artifact).toEqual(extractResult());
  const request = ndi.find("/v1/documents/extract");
  expect(request.body.schema).toEqual(EXTRACT_SCHEMA);
  expect(request.body.instructions).toBe("Extract the invoice fields.");
  expect(request.body.parse_mode).toBe("low");
  expect(ndi.sent("DELETE", `/v1/documents/jobs/${JOB}`)).toBe(true);
});

it("extract tool refuses a local path unless allowLocalFiles, then uploads and deletes it", async () => {
  const blocked = fakeNdi();
  const refusing = new NaceExtractTool(
    offline(blocked.fetch, { extractionSchema: EXTRACT_SCHEMA }),
  );
  await expect(
    refusing.invoke({
      name: "nace_extract",
      args: { source: INVOICE },
      id: "call_2",
      type: "tool_call",
    }),
  ).rejects.toThrow(/public https/);
  expect(blocked.requests).toHaveLength(0);

  const ndi = fakeNdi();
  const allowing = new NaceExtractTool(
    offline(ndi.fetch, { extractionSchema: EXTRACT_SCHEMA, allowLocalFiles: true }),
  );
  const message = await allowing.invoke({
    name: "nace_extract",
    args: { source: INVOICE },
    id: "call_3",
    type: "tool_call",
  });
  expect(JSON.parse(message.content as string).data).toEqual({
    invoice_number: "INV-2026-0042",
    total: 400,
  });
  expect(ndi.requests.some((r) => r.url.includes("/v1/documents/upload-grants"))).toBe(true);
  expect(ndi.sent("DELETE", `/v1/documents/workspaces/${WS}/files/${FILE}`)).toBe(true);
  expect(ndi.sent("DELETE", `/v1/documents/jobs/${JOB}`)).toBe(true);
});

it("keeps the idempotency key on a 503 job_pending, so a retry resumes the job", async () => {
  const ndi = fakeNdi({
    extract: () => ({
      status: 503,
      body: {
        error: {
          type: "service_unavailable",
          code: "job_pending",
          message:
            "The job's create is not confirmed yet. It may be running, and is billed once " +
            "if it ran. Send the same body again with the same key to get it.",
        },
      },
    }),
  });
  const extract = new NaceExtractTool(
    offline(ndi.fetch, { extractionSchema: EXTRACT_SCHEMA, maxRetries: 0 }),
  );
  let error: unknown;
  try {
    await extract.invoke({
      name: "nace_extract",
      args: { source: "https://example.com/invoice.pdf" },
      id: "call_pending",
      type: "tool_call",
    });
  } catch (caught) {
    error = caught;
  }
  expect(error).toBeInstanceOf(NaceInternalServerError);
  // The key the create carried: resending the same body with it resumes the
  // job instead of starting (and billing) a second one.
  const sent = ndi.find("/v1/documents/extract");
  expect(sent.headers.get("idempotency-key")).toBeTruthy();
  expect((error as NaceAPIError).idempotencyKey).toBe(sent.headers.get("idempotency-key"));
});

it("mints a fresh idempotency key on a retry, so a tool cannot resume a pending job", async () => {
  const pending = () => ({
    status: 503,
    body: {
      error: {
        type: "service_unavailable",
        code: "job_pending",
        message: "The job's create is not confirmed yet.",
      },
    },
  });
  const ndi = fakeNdi({ extract: pending });
  const extract = new NaceExtractTool(
    offline(ndi.fetch, { extractionSchema: EXTRACT_SCHEMA, maxRetries: 0 }),
  );
  const invoke = () =>
    extract.invoke({
      name: "nace_extract",
      args: { source: "https://example.com/invoice.pdf" },
      id: "call_retry",
      type: "tool_call",
    });
  let first: unknown;
  try {
    await invoke();
  } catch (caught) {
    first = caught;
  }
  let second: unknown;
  try {
    await invoke();
  } catch (caught) {
    second = caught;
  }
  const keys = ndi.requests
    .filter((r) => r.url.includes("/v1/documents/extract"))
    .map((r) => r.headers.get("idempotency-key"));
  expect(keys).toHaveLength(2);
  // A retried tool call mints a fresh key instead of resuming: resuming needs
  // the SDK with the key the first error carries.
  expect(keys[0]).toBeTruthy();
  expect(keys[1]).not.toBe(keys[0]);
  expect((second as NaceAPIError).idempotencyKey).toBe(keys[1]);
  expect((first as NaceAPIError).idempotencyKey).toBe(keys[0]);
});

it.each([
  ["fact", SEARCH_FACT],
  ["agent", { answer: SEARCH_FACT, routing: "agent" }],
])("search tool answers in the %s shape", async (_shape, output) => {
  const ndi = fakeNdi({
    search: () => ({
      body: job("automatic_search", "succeeded", {
        result_type: "automatic_search",
        output,
      }),
    }),
  });
  const search = new NaceSearchTool(offline(ndi.fetch, { workspaceId: WS, mode: "fact", topK: 3 }));
  const message = await search.invoke({
    name: "nace_search",
    args: { query: "What is the total due?" },
    id: "call_4",
    type: "tool_call",
  });
  const parsed = JSON.parse(message.content as string);
  expect(parsed.answer).toBe("The total due is USD 400.00.");
  expect(parsed.confidence).toBe("certain");
  expect(parsed.evidences).toEqual([
    { source_path: "invoice.html", page: 1, quote: "Total due: USD 400.00" },
  ]);
  // A complete answer is not marked partial.
  expect(parsed.exhausted).toBe(false);
  expect(message.artifact).toEqual({ result_type: "automatic_search", output });
  const request = ndi.find(`/v1/workspaces/${WS}/automatic-search`);
  expect(request.body).toMatchObject({ query: "What is the total due?", mode: "fact", top_k: 3 });
  expect(request.body).not.toHaveProperty("path_prefix");
  expect(ndi.sent("DELETE", `/v1/documents/jobs/${JOB}`)).toBe(true);
});

it("marks a partial agent answer as exhausted in the content", async () => {
  const ndi = fakeNdi({
    search: () => ({
      body: job("automatic_search", "succeeded", {
        result_type: "automatic_search",
        output: { answer: { ...SEARCH_FACT, exhausted: true }, routing: "agent" },
      }),
    }),
  });
  const search = new NaceSearchTool(offline(ndi.fetch, { workspaceId: WS }));
  const message = await search.invoke({
    name: "nace_search",
    args: { query: "List every invoice over USD 100" },
    id: "call_partial",
    type: "tool_call",
  });
  // The agent ran out of steps: the answer is partial, and the content says so.
  const parsed = JSON.parse(message.content as string);
  expect(parsed.answer).toBe("The total due is USD 400.00.");
  expect(parsed.exhausted).toBe(true);
  expect(parsed.evidences).toEqual([
    { source_path: "invoice.html", page: 1, quote: "Total due: USD 400.00" },
  ]);
});

it("reports the real error when a cancelled call has a getter-only message", async () => {
  const controller = new AbortController();
  const ndi = fakeNdi({
    search: () => ({ status: 202, body: job("automatic_search", "queued") }),
  });
  const aborting = async (input: string | URL | Request, init?: RequestInit): Promise<Response> => {
    if (
      (init?.method ?? "GET") === "GET" &&
      new URL(String(input)).pathname === `/v1/documents/jobs/${JOB}`
    ) {
      controller.abort();
      throw new DOMException("This operation was aborted", "AbortError");
    }
    return ndi.fetch(input, init);
  };
  const search = new NaceSearchTool(offline(aborting, { workspaceId: WS }));
  let error: unknown;
  try {
    await search.invoke(
      {
        name: "nace_search",
        args: { query: "What is the total due?" },
        id: "call_abort",
        type: "tool_call",
      },
      { signal: controller.signal },
    );
  } catch (caught) {
    error = caught;
  }
  expect(error).toBeInstanceOf(NaceError);
  expect(error).not.toBeInstanceOf(TypeError);
  expect(ndi.sent("DELETE", `/v1/documents/jobs/${JOB}`)).toBe(true);
});

it("passes an error whose cause chain holds a DOMException through without a TypeError", async () => {
  const client = new NdiClient(offline(fakeNdi().fetch));
  let error: unknown;
  try {
    await client.call(() =>
      Promise.reject(
        new Error("outer", {
          cause: new DOMException("This operation was aborted", "AbortError"),
        }),
      ),
    );
  } catch (caught) {
    error = caught;
  }
  expect(error).toBeInstanceOf(Error);
  expect(error).not.toBeInstanceOf(TypeError);
});

it("warns once when the job delete fails, and keeps the result", async () => {
  const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
  const ndi = fakeNdi({
    deleteJob: () => ({ status: 500, body: { error: { message: "boom" } } }),
  });
  const loader = new NaceDocumentLoader(
    offline(ndi.fetch, { source: "https://example.com/invoice.pdf", maxRetries: 0 }),
  );
  const documents = await loader.load();
  expect(documents[0]!.pageContent).toBe(MARKDOWN);
  expect(warn).toHaveBeenCalledTimes(1);
  expect(warn.mock.calls[0]![0]).toBe(`[langchain-nace] Could not delete NDI job ${JOB}.`);
});

it("keeps the key out of JSON.stringify and util.inspect of every component", () => {
  const ndi = fakeNdi();
  const loader = new NaceDocumentLoader(
    offline(ndi.fetch, { source: "https://example.com/i.pdf" }),
  );
  const extract = new NaceExtractTool(offline(ndi.fetch, { extractionSchema: EXTRACT_SCHEMA }));
  const search = new NaceSearchTool(offline(ndi.fetch, { workspaceId: WS }));
  for (const component of [loader, extract, search]) {
    expect(JSON.stringify(component)).not.toContain("nace_sk_test");
    expect(inspect(component, { depth: 5 })).not.toContain("nace_sk_test");
  }
});
