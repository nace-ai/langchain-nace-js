/** Live NDI contract tests (prod by default). Synthetic data only. */
import { readFileSync } from "node:fs";

import { NaceClient, TERMINAL_STATUSES } from "nace-sdk";
import { describe, expect, it } from "vitest";

import { NaceDocumentLoader, NaceExtractTool, NaceSearchTool } from "../../src/index.js";
import { BASE_URL, fixture, KEY } from "./live.js";

const INVOICE = fixture("invoice.html");

const EXTRACT_SCHEMA = {
  type: "object",
  properties: {
    invoice_number: { type: "string" },
    total: { type: "number" },
    po_number: { type: "string" },
  },
};

describe.skipIf(!KEY)("NDI contract", () => {
  it("parses the fixture into markdown", async () => {
    const loader = new NaceDocumentLoader({ apiKey: KEY, baseUrl: BASE_URL, source: INVOICE });
    const documents = await loader.load();
    expect(documents).toHaveLength(1);
    expect(documents[0]!.pageContent).toContain("INV-2026-0042");
    expect(documents[0]!.metadata.source).toBe(INVOICE);
  });

  it("extracts fields from the fixture, with not_found for the absent one", async () => {
    const extract = new NaceExtractTool({
      apiKey: KEY,
      baseUrl: BASE_URL,
      extractionSchema: EXTRACT_SCHEMA,
      instructions: "Extract the invoice fields.",
      allowLocalFiles: true,
    });
    const message = await extract.invoke({
      name: "nace_extract",
      args: { source: INVOICE },
      id: "call_extract_invoice",
      type: "tool_call",
    });
    const parsed = JSON.parse(message.content as string);
    expect(parsed.data.invoice_number).toBe("INV-2026-0042");
    expect(
      parsed.fields.find((field: { path: string }) => field.path === "/po_number")?.status,
    ).toBe("not_found");
  });

  it("answers a workspace search with evidence", async () => {
    // One HTTP attempt must outlast the 60 s that the ingestion create may hold.
    const client = new NaceClient({ apiKey: KEY, baseURL: BASE_URL, timeout: 120_000 });
    const name = `langchain-nace-js-tests-${crypto.randomUUID().slice(0, 8)}`;
    const workspace = await client.workspaces.create({ name });
    const workspaceId = workspace.workspace_id;
    try {
      const file = await client.documents.upload(readFileSync(INVOICE), {
        file_name: "invoice.html",
        workspace_id: workspaceId,
      });
      let ingest = await client.workspaces.ingest(workspaceId, {
        file_ids: [file.file_id],
        wait_seconds: 60,
      });
      if (!TERMINAL_STATUSES.has(String(ingest.status))) {
        ingest = await client.jobs.wait(ingest.job_id);
      }
      expect(ingest.status).toBe("succeeded");
      const search = new NaceSearchTool({
        apiKey: KEY,
        baseUrl: BASE_URL,
        workspaceId,
        mode: "fact",
      });
      const message = await search.invoke({
        name: "nace_search",
        args: { query: "What is the total due on the invoice?" },
        id: "call_search_total",
        type: "tool_call",
      });
      const parsed = JSON.parse(message.content as string);
      expect(parsed.answer).toContain("400");
      expect(parsed.evidences.length).toBeGreaterThan(0);
    } finally {
      await client.workspaces.delete(workspaceId, { confirm_name: name });
    }
  });
});
