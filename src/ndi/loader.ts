/** NDI Parse as a LangChain.js document loader. */
import { BaseDocumentLoader } from "@langchain/core/document_loaders/base";
import { Document } from "@langchain/core/documents";
import type { DocumentJob } from "nace-sdk";

import { NaceError } from "../errors.js";
import {
  fileNameOf,
  isRecord,
  type NaceNdiFields,
  NdiClient,
  type NdiSource,
  redactUrl,
} from "./client.js";

export interface NaceDocumentLoaderFields extends NaceNdiFields {
  /** A public https URL, a local file path, or the bytes of the file (then `fileName` is necessary). */
  source: NdiSource;
  /** The name of the file with its extension. Necessary for bytes and for a URL without an extension. */
  fileName?: string;
  /** `none` (default): one Document for the file. `page` or `section`: one Document for each chunk. */
  chunking?: "none" | "page" | "section";
  /** Parse effort. Default: `low`. */
  parseMode?: "low" | "medium" | "high";
  /** Workspace that gets the upload. Default: the account's own storage. */
  workspaceId?: string;
  /**
   * More Parse request fields (for example `page_ranges`, `ocr`). They replace the defaults
   * `parse_mode` and `output`; they cannot replace `chunking`, `source` or `wait_seconds`.
   */
  parseKwargs?: Record<string, unknown>;
}

/**
 * Load one document as Markdown with NDI Parse.
 *
 * `chunking: "none"` (the default) returns one `Document` with the whole
 * content (the Markdown, or the plain text when `parseKwargs.output.formats`
 * asks for `text`); `"page"` and `"section"` return one `Document` per chunk,
 * with its `chunk_id` and `page` in the metadata. A local file or bytes are
 * uploaded under a one-off path and deleted again after the read, together
 * with the job.
 *
 * @example
 * ```ts
 * const loader = new NaceDocumentLoader({
 *   source: "https://example.com/invoice.pdf",
 *   chunking: "page",
 * });
 * const documents = await loader.load();
 * ```
 */
export class NaceDocumentLoader extends BaseDocumentLoader {
  /** The source as shown: a URL without its query, the path, or the bytes as given. */
  readonly source: NdiSource;

  readonly fileName: string | undefined;

  readonly chunking: "none" | "page" | "section";

  readonly parseMode: "low" | "medium" | "high";

  readonly workspaceId: string | undefined;

  /**
   * More Parse request fields (for example `page_ranges`, `ocr`). They replace the defaults
   * `parse_mode` and `output`; they cannot replace `chunking`, `source` or `wait_seconds`.
   * Non-enumerable: they can hold secrets (a PDF's `password`), which must never
   * reach serialization or inspection of the loader. `loader.parseKwargs` reads them.
   */
  declare readonly parseKwargs: Record<string, unknown> | undefined;

  readonly #client: NdiClient;

  /**
   * The full source, a signed URL with its query included. Only the request
   * uses it; a private field never reaches serialization.
   */
  readonly #fullSource: NdiSource;

  /** The `source` metadata: the URL without its query, the path, the file name for bytes. */
  readonly #sourceLabel: string;

  constructor(fields: NaceDocumentLoaderFields) {
    super();
    // Configuration errors appear early, in the constructor.
    this.#client = new NdiClient(fields);
    this.#fullSource = fields.source;
    let label: string | undefined;
    if (typeof fields.source === "string") {
      // The query, the fragment and any user name or password of a signed
      // link are credentials; the metadata and the enumerable `source` field
      // keep the URL without them (`redactUrl`). Only the request carries the
      // URL exactly as given.
      label = fields.source.includes("://") ? redactUrl(fields.source) : fields.source;
    } else {
      label = fields.fileName ?? fileNameOf(fields.source);
    }
    if (label === undefined) {
      throw new NaceError(
        "`source` bytes need `fileName` with the file's extension, " +
          "or a `File`, which carries its own name.",
      );
    }
    this.source = typeof fields.source === "string" ? label : fields.source;
    this.fileName = fields.fileName;
    this.chunking = fields.chunking ?? "none";
    this.parseMode = fields.parseMode ?? "low";
    this.workspaceId = fields.workspaceId;
    // Non-enumerable (see the field's doc): `JSON.stringify` and `util.inspect`
    // of the loader never show what `parseKwargs` holds.
    Object.defineProperty(this, "parseKwargs", {
      value: fields.parseKwargs === undefined ? undefined : { ...fields.parseKwargs },
      enumerable: false,
      writable: false,
      configurable: true,
    });
    this.#sourceLabel = label;
  }

  /**
   * Parse the document. Everything is read from the job before it is deleted.
   */
  override async load(): Promise<Document[]> {
    return this.#client.withSource(
      this.#fullSource,
      { fileName: this.fileName, workspaceId: this.workspaceId },
      (source) =>
        this.#client.withJob(
          (client, waitSeconds, signal) =>
            client.documents.parse(
              {
                parse_mode: this.parseMode,
                output: { formats: ["markdown"] },
                ...this.parseKwargs,
                chunking: { strategy: this.chunking },
                source,
                wait_seconds: waitSeconds,
              },
              { signal },
            ),
          {},
          (job, deadline) => this.#documents(job, deadline),
        ),
    );
  }

  /** The `Document`s of one finished parse job. */
  async #documents(job: DocumentJob, deadline: number): Promise<Document[]> {
    const result = isRecord(job.result) ? job.result : {};
    const document = isRecord(result.document) ? result.document : {};
    const metadata: Record<string, unknown> = {
      source: this.#sourceLabel,
      job_id: job.job_id,
      page_count: document.page_count,
    };
    const truncated =
      Array.isArray(result.disclosures) &&
      result.disclosures.some((entry) => isRecord(entry) && entry.code === "content_truncated");
    const contentUrl = document.content_url;
    if (truncated && typeof contentUrl !== "string") {
      // A workbook over 1 MiB: NDI returns a preview per sheet and no full
      // content to download, so the preview must not pass as the result. This
      // holds for every chunking mode.
      throw new NaceError(
        "The parse result is over 1 MiB and NDI cut it short. Parse fewer sheets or pages " +
          "with parseKwargs (spreadsheet.sheets, page_ranges).",
      );
    }
    if (this.chunking === "none") {
      const markdown = typeof document.markdown === "string" ? document.markdown : undefined;
      const text = typeof document.text === "string" ? document.text : undefined;
      let content = markdown ?? text;
      if (content === undefined) {
        // A format that renders no text (["blocks"], or nothing asked for)
        // must not pass as an empty document.
        throw new NaceError(
          "The parse result has no `markdown` and no `text`. NaceDocumentLoader reads those " +
            "output formats; ask for one in `parseKwargs.output.formats`.",
        );
      }
      if (truncated && typeof contentUrl === "string") {
        if (markdown === undefined) {
          // The download carries the full Markdown, which must not replace the
          // text the caller asked for. Keep the preview and say so.
          console.warn(
            "[langchain-nace] The parse result is over 1 MiB and NDI cut it short. The full " +
              "content downloads as Markdown, so the text preview was kept; ask for `markdown` " +
              "in parseKwargs.output.formats to read the document in full.",
          );
        } else {
          const bytes = await this.#client.download(job.job_id, contentUrl, deadline);
          content = new TextDecoder().decode(bytes);
        }
      }
      return [new Document({ pageContent: content, metadata })];
    }
    const chunks = Array.isArray(document.chunks) ? document.chunks : [];
    return chunks
      .filter((chunk): chunk is Record<string, unknown> => isRecord(chunk))
      .map((chunk) => {
        const chunkMetadata: Record<string, unknown> = { ...metadata, chunk_id: chunk.id };
        const location = isRecord(chunk.location) ? chunk.location : {};
        if (typeof location.page === "number") chunkMetadata.page = location.page;
        return new Document({
          pageContent: typeof chunk.content === "string" ? chunk.content : "",
          metadata: chunkMetadata,
        });
      });
  }
}
