import { getRetryable, LangChainError } from "@langchain/core/errors";
import { AIMessage, HumanMessage, ToolMessage } from "@langchain/core/messages";
import { describe, expect, it } from "vitest";
import { unwrapGatewayMessage } from "../../src/errors.js";
import {
  asDataUrl,
  encodeState,
  NaceAPIError,
  NaceClassifier,
  NaceError,
  NaceImageError,
  NaceRateLimitError,
  NaceUnprocessableEntityError,
  toDrexImage,
} from "../../src/index.js";
import { latestImages, withoutMedia } from "../../src/middleware/index.js";
import { fakeDrex, offline, PNG_1x1 } from "./helpers.js";

describe("errors", () => {
  it("unwraps OpenRouter's raw error text", () => {
    expect(unwrapGatewayMessage('HTTP 422: {"detail": "model does not exist"}')).toEqual({
      message: "model does not exist",
      issues: [],
    });
    expect(
      unwrapGatewayMessage('[{"path": ["questions", "q", "type"], "message": "Invalid input"}]'),
    ).toEqual({
      message: "the request is invalid",
      issues: [{ path: "questions.q.type", message: "Invalid input" }],
    });
    expect(unwrapGatewayMessage("plain text")).toEqual({ message: "plain text", issues: [] });
  });

  it("brands every error under LangChainError and NaceError", () => {
    const error = new NaceUnprocessableEntityError("x", { status: 422 });
    expect(LangChainError.isInstance(error)).toBe(true);
    expect(NaceError.isInstance(error)).toBe(true);
    expect(NaceAPIError.isInstance(error)).toBe(true);
    expect(NaceRateLimitError.isInstance(error)).toBe(false);
    expect(error.name).toBe("NaceUnprocessableEntityError");
  });

  it("marks client errors as not retryable and 5xx as retryable", async () => {
    const make = (status: number) =>
      new NaceClassifier(
        offline(fakeDrex([{ status, body: { error: { message: "x" } } }]).fetch, {
          questions: { q: { type: "noul", instructions: "?" } },
          maxRetries: 0,
        }),
      )
        .invoke("x")
        .catch((e) => e);
    expect(getRetryable(await make(422))).toBe(false);
    expect(getRetryable(await make(503))).toBe(true);
  });
});

describe("images", () => {
  it("types base64 from its first bytes", () => {
    expect(asDataUrl(PNG_1x1)).toBe(`data:image/png;base64,${PNG_1x1}`);
    expect(asDataUrl("/9j/4AAQSkZJRg==")).toMatch(/^data:image\/jpeg;base64,/);
    expect(asDataUrl("data:image/webp;base64,AAAA")).toBe("data:image/webp;base64,AAAA");
  });

  it("refuses what it can't read, without echoing more than a prefix", () => {
    expect(() => toDrexImage("not an image or a path")).toThrow(NaceImageError);
    expect(() => toDrexImage({ type: "image", fileId: "file_1" })).toThrow(NaceImageError);
    expect(() => toDrexImage(42)).toThrow(NaceImageError);
    expect(() => toDrexImage(new Uint8Array([1, 2, 3]))).toThrow(/PNG, JPEG and WebP/);
  });

  it("turns an UploadedFile into a workspace_file reference", () => {
    expect(toDrexImage({ workspace_id: "ws", file_id: "f", path: "a.png" })).toEqual({
      type: "workspace_file",
      workspace_id: "ws",
      file_id: "f",
    });
  });
});

describe("state", () => {
  it("keeps unknown content blocks as JSON and drops tool_use blocks repeated in tool_calls", () => {
    const { state } = encodeState(
      [
        new AIMessage({
          content: [
            { type: "thinking", thinking: "hmm" },
            { type: "text", text: "Deleting." },
            { type: "tool_use", id: "c1", name: "delete", input: { id: 1 } },
          ],
          tool_calls: [{ id: "c1", name: "delete", args: { id: 1 } }],
        }),
      ],
      { images: "lift" },
    );
    expect((state as any)[0].content).toEqual([
      { type: "thinking", thinking: "hmm" },
      { type: "text", text: "Deleting." },
    ]);
  });

  it("replaces media with placeholders when asked", () => {
    const { state, images } = encodeState(
      [
        new HumanMessage({
          content: [
            { type: "image", source_type: "base64", data: PNG_1x1, mime_type: "image/png" },
            { type: "audio", data: "AAAA", mimeType: "audio/wav" },
          ],
        }),
      ],
      { images: "placeholder", media: "placeholder" },
    );
    expect(images).toEqual([]);
    expect((state as any)[0].content).toEqual([
      { type: "text", text: "[image omitted]" },
      { type: "text", text: "[audio omitted]" },
    ]);
  });

  it("keeps a tool message's error status", () => {
    const { state } = encodeState(
      [new ToolMessage({ content: "boom", tool_call_id: "c1", status: "error" })],
      { images: "lift" },
    );
    expect((state as any)[0]).toEqual({
      role: "tool",
      content: "boom",
      tool_call_id: "c1",
      status: "error",
    });
  });
});

describe("middleware media helpers", () => {
  const inline = { type: "image", source_type: "base64", data: PNG_1x1, mime_type: "image/png" };
  const remote = { type: "image_url", image_url: { url: "https://example.com/a.png" } };

  it("keeps inline images only when asked, and always drops remote ones", () => {
    const message = new HumanMessage({ content: [{ type: "text", text: "hi" }, inline, remote] });
    const kept = withoutMedia(message, { keepImages: true });
    expect(HumanMessage.isInstance(kept)).toBe(true);
    expect(kept.content).toEqual([
      { type: "text", text: "hi" },
      inline,
      { type: "text", text: "[image omitted]" },
    ]);
    const plain = withoutMedia({ role: "user", content: [inline] }, { keepImages: false });
    expect(plain.content).toEqual([{ type: "text", text: "[image omitted]" }]);
    const text = new HumanMessage("just text");
    expect(withoutMedia(text, { keepImages: false })).toBe(text);
  });

  it("finds the newest inline images, oldest first", () => {
    const a = { ...inline, data: "AAAA" };
    const b = { ...inline, data: "BBBB" };
    const c = { ...inline, data: "CCCC" };
    const messages = [
      new HumanMessage({ content: [a] }),
      { role: "tool", content: [b, remote] },
      new HumanMessage({ content: [c] }),
    ];
    expect(latestImages(messages, 2)).toEqual([b, c]);
  });
});
