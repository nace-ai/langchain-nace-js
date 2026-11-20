/** NDI Extract and automatic workspace search as LangChain.js tools. */
import type { CallbackManagerForToolRun } from "@langchain/core/callbacks/manager";
import { StructuredTool, type ToolRunnableConfig } from "@langchain/core/tools";
import type { ExtractParams } from "nace-sdk";
import * as z from "zod/v4";

import { NaceError } from "../errors.js";
import { isRecord, type NaceNdiFields, NdiClient } from "./client.js";

function extractInput(allowLocalFiles: boolean) {
  return z.object({
    source: z
      .string()
      .describe(
        allowLocalFiles
          ? "The public https URL of the document, or the path of a local file."
          : "The public https URL of the document.",
      ),
  });
}
type ExtractInput = ReturnType<typeof extractInput>;

const SEARCH_INPUT = z.object({
  query: z.string().describe("The question to answer from the workspace files."),
});

export interface NaceExtractToolFields extends NaceNdiFields {
  /** JSON Schema of the fields to extract. NDI subset: no `$ref`, `anyOf`, `format`. */
  extractionSchema: Record<string, unknown>;
  /** Guidance on how to fill the schema. */
  instructions?: string;
  /** Parse effort. Default: `low`. */
  parseMode?: "low" | "medium" | "high";
  /** Let the model name local files. Default: `false` (https URLs only). */
  allowLocalFiles?: boolean;
  /** Workspace that gets the upload of a local file. */
  workspaceId?: string;
  /** Tool name. Default: `nace_extract`. */
  name?: string;
  /** Tool description for the model. */
  description?: string;
}

function extractDescription(allowLocalFiles: boolean): string {
  return (
    "Extract structured fields from one document. Input: the public https URL of the " +
    `document${allowLocalFiles ? ", or the path of a local file" : ""}. Returns the extracted ` +
    "data and, for each field, its status (found, not_found or ambiguous), confidence " +
    "and the quoted source text."
  );
}

/**
 * A tool that extracts fields matching a JSON Schema from one document with NDI
 * Extract. The content is a JSON string `{ data, fields }`, where each field has
 * its `path`, `value`, `status`, `confidence` and the `quotes` it cites; the
 * artifact is the raw NDI result.
 *
 * @example
 * ```ts
 * const extract = new NaceExtractTool({
 *   extractionSchema: {
 *     type: "object",
 *     properties: { invoice_number: { type: "string" } },
 *   },
 * });
 * const message = await extract.invoke({
 *   name: "nace_extract",
 *   args: { source: "https://example.com/invoice.pdf" },
 *   id: "call_1",
 *   type: "tool_call",
 * });
 * const { data, fields } = JSON.parse(message.content as string);
 * ```
 */
export class NaceExtractTool extends StructuredTool<ExtractInput> {
  override name: string;

  override description: string;

  override schema: ExtractInput;

  readonly extractionSchema: Record<string, unknown>;

  readonly instructions: string | undefined;

  readonly parseMode: "low" | "medium" | "high";

  readonly allowLocalFiles: boolean;

  readonly workspaceId: string | undefined;

  readonly #client: NdiClient;

  constructor(fields: NaceExtractToolFields) {
    // Connection fields stay out of `lc_kwargs`; the client is a `#` private field.
    super({ responseFormat: "content_and_artifact" });
    this.#client = new NdiClient(fields);
    this.extractionSchema = fields.extractionSchema;
    this.instructions = fields.instructions;
    this.parseMode = fields.parseMode ?? "low";
    this.allowLocalFiles = fields.allowLocalFiles ?? false;
    this.workspaceId = fields.workspaceId;
    this.name = fields.name ?? "nace_extract";
    this.schema = extractInput(this.allowLocalFiles);
    this.description = fields.description ?? extractDescription(this.allowLocalFiles);
  }

  // The model controls `source`, so a local path needs an explicit opt-in.
  protected override async _call(
    input: { source: string },
    _runManager?: CallbackManagerForToolRun,
    parentConfig?: ToolRunnableConfig,
  ): Promise<[string, unknown]> {
    const signal = parentConfig?.signal;
    if (!input.source.includes("://") && this.allowLocalFiles !== true) {
      throw new NaceError("source must be a public https:// URL.");
    }
    return this.#client.withSource(
      input.source,
      { workspaceId: this.workspaceId, signal },
      (source) =>
        this.#client.withJob(
          // The signal aborts at the job's deadline (the caller's signal is
          // forwarded onto it), so the create stops with the operation.
          (client, waitSeconds, signal) => {
            const params: ExtractParams = {
              source,
              schema: this.extractionSchema,
              parse_mode: this.parseMode,
              wait_seconds: waitSeconds,
            };
            if (this.instructions !== undefined) params.instructions = this.instructions;
            return client.documents.extract(params, { signal });
          },
          { signal },
          async (job) => {
            const result = isRecord(job.result) ? job.result : {};
            const fields = (Array.isArray(result.fields) ? result.fields : [])
              .filter((field): field is Record<string, unknown> => isRecord(field))
              .map((field) => ({
                path: field.path,
                value: field.value,
                status: field.status,
                confidence: field.confidence,
                quotes: (Array.isArray(field.citations) ? field.citations : [])
                  .filter((citation): citation is Record<string, unknown> => isRecord(citation))
                  .map((citation) => citation.source_text)
                  .filter((quote): quote is string => typeof quote === "string"),
              }));
            return [JSON.stringify({ data: result.data, fields }), job.result] as [string, unknown];
          },
        ),
    );
  }
}

export interface NaceSearchToolFields extends NaceNdiFields {
  /** An ingested workspace. */
  workspaceId: string;
  /** `auto` (default) routes the question; `fact` and `agent` skip routing. */
  mode?: "auto" | "fact" | "agent";
  /** How many evidences a fact answer cites, 1 to 50. */
  topK?: number;
  /** Search only the files under this folder. */
  pathPrefix?: string;
  /** Tool name. Default: `nace_search`. */
  name?: string;
  /** Tool description for the model. */
  description?: string;
}

const SEARCH_DESCRIPTION =
  "Answer a question from the files in a Nace workspace. Returns the answer, its " +
  "confidence and the evidence it cites (file path, page, quote). It searches the " +
  "workspace files, not the web.";

/**
 * A tool that answers a question over the ingested files of one Nace workspace
 * with NDI automatic search. The content is a JSON string `{ answer, confidence,
 * evidences, exhausted }`: `exhausted` is `true` when the agent ran out of
 * steps and the answer is partial; the artifact is the raw NDI result.
 *
 * @example
 * ```ts
 * const search = new NaceSearchTool({ workspaceId: "..." });
 * const message = await search.invoke({
 *   name: "nace_search",
 *   args: { query: "What is the total due on the last invoice?" },
 *   id: "call_1",
 *   type: "tool_call",
 * });
 * const { answer, evidences, exhausted } = JSON.parse(message.content as string);
 * ```
 */
export class NaceSearchTool extends StructuredTool<typeof SEARCH_INPUT> {
  override name: string;

  override description: string;

  override schema = SEARCH_INPUT;

  readonly workspaceId: string;

  readonly mode: "auto" | "fact" | "agent" | undefined;

  readonly topK: number | undefined;

  readonly pathPrefix: string | undefined;

  readonly #client: NdiClient;

  constructor(fields: NaceSearchToolFields) {
    super({ responseFormat: "content_and_artifact" });
    if (typeof fields.workspaceId !== "string" || fields.workspaceId.trim() === "") {
      throw new NaceError("`workspaceId` must be a non-empty string.");
    }
    this.#client = new NdiClient(fields);
    this.workspaceId = fields.workspaceId;
    this.mode = fields.mode;
    this.topK = fields.topK;
    this.pathPrefix = fields.pathPrefix;
    this.name = fields.name ?? "nace_search";
    this.description = fields.description ?? SEARCH_DESCRIPTION;
  }

  protected override async _call(
    input: { query: string },
    _runManager?: CallbackManagerForToolRun,
    parentConfig?: ToolRunnableConfig,
  ): Promise<[string, unknown]> {
    const signal = parentConfig?.signal;
    return this.#client.withJob(
      // The signal aborts at the job's deadline (the caller's signal is
      // forwarded onto it), so the create stops with the operation.
      (client, waitSeconds, signal) => {
        const params: {
          query: string;
          wait_seconds: number;
          mode?: "auto" | "fact" | "agent";
          top_k?: number;
          path_prefix?: string;
        } = { query: input.query, wait_seconds: waitSeconds };
        if (this.mode !== undefined) params.mode = this.mode;
        if (this.topK !== undefined) params.top_k = this.topK;
        if (this.pathPrefix !== undefined) params.path_prefix = this.pathPrefix;
        return client.search.automatic(this.workspaceId, params, { signal });
      },
      { signal },
      async (job) => {
        const result = isRecord(job.result) ? job.result : {};
        const output = isRecord(result.output) ? result.output : {};
        const answer = isRecord(output.answer) ? output.answer : output;
        const evidences = Array.isArray(answer.evidences) ? answer.evidences : [];
        return [
          JSON.stringify({
            answer: answer.answer ?? null,
            confidence: answer.confidence ?? null,
            // `true` when the agent ran out of steps: the answer is partial.
            exhausted: answer.exhausted === true,
            evidences: evidences
              .filter((evidence): evidence is Record<string, unknown> => isRecord(evidence))
              .map((evidence) => ({
                source_path: evidence.source_path,
                page: evidence.page,
                quote: evidence.quote,
              })),
          }),
          job.result,
        ] as [string, unknown];
      },
    );
  }
}
