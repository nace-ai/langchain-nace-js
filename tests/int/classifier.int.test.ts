/** Live contract tests against Drex (prod by default). Synthetic data only. */
import { readFileSync } from "node:fs";

import { HumanMessage } from "@langchain/core/messages";
import { NaceClient } from "nace-sdk";
import { describe, expect, it } from "vitest";

import {
  NaceAPIError,
  NaceAuthenticationError,
  NaceClassifier,
  NaceImageModelError,
  NaceUnprocessableEntityError,
  type Questions,
} from "../../src/index.js";
import {
  BASE_URL,
  BLUE,
  checkResponse,
  fixture,
  KEY,
  live,
  MODELS,
  RED,
  TICKET,
  WHICH_RED,
} from "./live.js";

describe.skipIf(!KEY)("Drex contract", () => {
  describe.each(MODELS)("%s", (model) => {
    it("answers a single noul", async () => {
      const r = await live(model, {
        questions: {
          refund: { type: "noul", instructions: "Is the customer asking for a refund?" },
        },
      }).invoke(TICKET);
      expect(r.nouls.refund!.noul).toBeGreaterThan(0.7);
      checkResponse(r);
    });

    it("answers a single choice with every label's probability", async () => {
      const r = await live(model, {
        questions: {
          team: {
            type: "choice",
            instructions: "Which team should handle this?",
            criteria: { billing: "Payments, invoices, refunds", technical: "Bugs and outages" },
          },
        },
      }).invoke(TICKET);
      expect(r.choices.team!.choice).toBe("billing");
      expect(Object.keys(r.choices.team!.probabilities).sort()).toEqual(["billing", "technical"]);
      checkResponse(r);
    });

    it("answers a single score with its legend", async () => {
      const r = await live(model, {
        questions: {
          mood: {
            type: "score",
            instructions: "How upset is the customer?",
            criteria: ["calm", "annoyed", "furious"],
          },
        },
      }).invoke(TICKET);
      expect(Object.keys(r.scores.mood!.probabilities).sort()).toEqual(["0", "1", "2"]);
      expect(r.scores.mood!.legend).toEqual({ "0": "calm", "1": "annoyed", "2": "furious" });
      checkResponse(r);
    });

    it("answers mixed questions in one request", async () => {
      const r = await live(model, {
        questions: {
          refund: { type: "noul", instructions: "Is the customer asking for a refund?" },
          team: {
            type: "choice",
            instructions: "Which team?",
            criteria: { billing: null, technical: null },
          },
          mood: {
            type: "score",
            instructions: "How upset?",
            criteria: ["calm", "annoyed", "furious"],
          },
        },
      }).invoke(TICKET);
      expect(Object.keys(r.answers).sort()).toEqual(["mood", "refund", "team"]);
      checkResponse(r);
    });

    it("takes a noul with criteria only and a null choice description", async () => {
      const r = await live(model, {
        questions: {
          refund: { type: "noul", criteria: { true: "The customer asks for money back." } },
          tone: {
            type: "choice",
            instructions: "What is the tone?",
            criteria: { polite: null, rude: "Insults or shouting" },
          },
        },
      }).invoke(TICKET);
      expect(r.nouls.refund!.noul).toBeGreaterThan(0.5);
      checkResponse(r);
    });
  });

  describe("drex-max images", () => {
    it("reads one image", async () => {
      const r = await live("drex-max", {
        questions: { red: { type: "noul", instructions: "Is Image 1 mostly red?" } },
      }).invoke("One image is attached.", { images: [RED] });
      expect(r.nouls.red!.noul).toBeGreaterThan(0.7);
      expect(r.usage.imageTokens ?? 0).toBeGreaterThan(0);
      checkResponse(r);
    });

    it.each([
      [[RED, BLUE], "Image 1"],
      [[BLUE, RED], "Image 2"],
    ])("keeps two images in order (%#)", async (images, expected) => {
      const r = await live("drex-max", { questions: WHICH_RED }).invoke(
        "Two images are attached.",
        {
          images,
        },
      );
      expect(r.choices.red!.choice).toBe(expected);
      checkResponse(r);
    });

    const blocks: [string, (d: string) => Record<string, unknown>][] = [
      ["v1 data", (d) => ({ type: "image", data: d, mimeType: "image/png" })],
      [
        "v0 source_type",
        (d) => ({ type: "image", source_type: "base64", data: d, mime_type: "image/png" }),
      ],
      [
        "image_url",
        (d) => ({ type: "image_url", image_url: { url: `data:image/png;base64,${d}` } }),
      ],
    ];
    for (const [name, block] of blocks) {
      for (const asObject of [false, true]) {
        it(`reads ${name} blocks inside ${asObject ? "plain { role, content } objects" : "messages"}`, async () => {
          const content = [{ type: "text", text: "Two pictures:" }, block(BLUE), block(RED)];
          const message = asObject
            ? { role: "user", content }
            : new HumanMessage({ content: content as never });
          const r = await live("drex-max", { questions: WHICH_RED }).invoke([message as never]);
          expect(r.choices.red!.choice).toBe("Image 2");
          expect(r.usage.imageTokens ?? 0).toBeGreaterThan(0);
        });
      }
    }

    it("reads images from a file path and from bytes", async () => {
      const r = await live("drex-max", { questions: WHICH_RED }).invoke("Two pictures.", {
        images: [fixture("red.png"), readFileSync(fixture("blue.png"))],
      });
      expect(r.choices.red!.choice).toBe("Image 1");
    });

    it("uses drex-max by default, so images work with no model given", async () => {
      const classifier = new NaceClassifier({
        apiKey: KEY,
        baseUrl: BASE_URL,
        questions: { red: { type: "noul", instructions: "Is Image 1 red?" } },
      });
      const r = await classifier.invoke("One image.", { images: [RED] });
      expect(classifier.model).toBe("drex-max");
      expect(r.model).toBe("drex-max");
      expect(r.nouls.red!.noul).toBeGreaterThan(0.7);
    });
  });

  describe("limits and errors", () => {
    it("refuses 17 questions on drex-max with a questions path, or answers all (Drex's limit)", async () => {
      const questions: Questions = {};
      for (let i = 0; i < 17; i += 1) {
        questions[`q${i}`] = {
          type: "noul",
          instructions: `Does the text mention the number ${i}?`,
        };
      }
      try {
        const r = await live("drex-max", { questions }).invoke(TICKET);
        expect(Object.keys(r.nouls)).toHaveLength(17);
      } catch (error) {
        expect(error).toBeInstanceOf(NaceUnprocessableEntityError);
        const issues = (error as NaceUnprocessableEntityError).issues;
        expect(issues.some((i) => i.path.startsWith("questions"))).toBe(true);
      }
    });

    it("refuses images on drex-v2.0 before sending", async () => {
      await expect(
        live("drex-v2.0", { questions: { q: { type: "noul", instructions: "Red?" } } }).invoke(
          "x",
          {
            images: [RED],
          },
        ),
      ).rejects.toThrow(NaceImageModelError);
    });

    it("shows a 422 as path: message", async () => {
      const error = await live("drex-v2.0", { questions: { q: { type: "rank", criteria: ["a"] } } })
        .invoke(TICKET)
        .catch((e) => e);
      expect(error).toBeInstanceOf(NaceUnprocessableEntityError);
      expect(error.issues.some((i: { path: string }) => i.path.startsWith("questions"))).toBe(true);
      expect(error.message).toMatch(/^Drex returned 422/);
      expect(error.message).toContain("questions");
    });

    it("makes a bad key a clear 401", async () => {
      const error = await new NaceClassifier({
        apiKey: "nace_sk_definitely_not_a_real_key",
        baseUrl: BASE_URL,
        questions: { q: { type: "noul", instructions: "?" } },
      })
        .invoke("x")
        .catch((e) => e);
      expect(error).toBeInstanceOf(NaceAuthenticationError);
      expect(error.status).toBe(401);
    });

    it("sends an unknown drex id as given: refused on `model`, or served by drex-latest", async () => {
      try {
        const r = await live("drex-v99.9", {
          questions: { q: { type: "noul", instructions: "?" } },
        }).invoke("x");
        expect(r.model.startsWith("drex-")).toBe(true);
      } catch (error) {
        expect(error).toBeInstanceOf(NaceUnprocessableEntityError);
        expect((error as NaceUnprocessableEntityError).issues.some((i) => i.path === "model")).toBe(
          true,
        );
      }
    });

    it("refuses a non-Drex model with path model", async () => {
      const error = await live("jev-latest", {
        questions: { q: { type: "noul", instructions: "?" } },
      })
        .invoke("x")
        .catch((e) => e);
      expect(error).toBeInstanceOf(NaceUnprocessableEntityError);
      expect(error.issues.some((i: { path: string }) => i.path === "model")).toBe(true);
    });

    it("lets Drex refuse 9 images", async () => {
      const error = await live("drex-max", {
        questions: { q: { type: "noul", instructions: "Red?" } },
      })
        .invoke("x", { images: Array(9).fill(RED) })
        .catch((e) => e);
      expect(error).toBeInstanceOf(NaceUnprocessableEntityError);
      expect(error.issues.some((i: { path: string }) => i.path.startsWith("images"))).toBe(true);
    });

    it("lets Drex refuse a body over 4 MiB", async () => {
      const error = await live("drex-v2.0", {
        questions: { q: { type: "noul", instructions: "?" } },
        maxRetries: 0,
      })
        .invoke("a ".repeat(2_200_000))
        .catch((e) => e);
      expect(error).toBeInstanceOf(NaceAPIError);
      expect([400, 413, 422]).toContain(error.status);
    });
  });

  describe("volume, models and JSON state", () => {
    it("answers a batch of 20 without rate-limit errors", async () => {
      const texts = Array.from(
        { length: 20 },
        (_, i) => `Order #${i}: ${i % 2 ? "refund please" : "where is my package"}`,
      );
      const results = await live("drex-v2.0", {
        questions: { refund: { type: "noul", instructions: "Refund request?" } },
      }).batch(texts);
      expect(results.map((r) => r.nouls.refund!.noul > 0.5)).toEqual(
        texts.map((_, i) => i % 2 === 1),
      );
    });

    it("answers on every listed model", async () => {
      const client = new NaceClient({ apiKey: KEY, ...(BASE_URL ? { baseURL: BASE_URL } : {}) });
      const listed = (await client.models.list())
        .filter((m) => !m.description.startsWith("Retired"))
        .map((m) => m.name);
      expect(listed).toContain("drex-v2.0");
      for (const name of listed) {
        const r = await live(name, {
          questions: { refund: { type: "noul", instructions: "Refund request?" } },
        }).invoke(TICKET);
        expect(r.nouls.refund!.noul).toBeGreaterThanOrEqual(0);
        checkResponse(r);
      }
    });

    it("reads JSON state with messages inside and long questions", async () => {
      const r = await live("drex-v2.0", {
        questions: {
          churn: {
            type: "noul",
            instructions: `Is the customer trying to cancel? ${"Consider plan and tenure. ".repeat(50)}`,
          },
        },
      }).invoke({
        customer: { plan: "pro", tenure_months: 30 },
        messages: [new HumanMessage("Cancel my plan now.")],
      });
      expect(r.nouls.churn!.noul).toBeGreaterThan(0.7);
    });
  });
});
