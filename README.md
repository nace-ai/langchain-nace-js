<h1 align="center">langchain-nace</h1>

<p align="center">
  <b>Calibrated decisions for LangChain.js agents, powered by <a href="https://console.nace.ai/docs">Drex</a>.</b><br/>
  Route requests, pick models and check tool calls before they run, with real probabilities instead of generated text.
</p>

<p align="center">
  <a href="https://www.npmjs.com/package/langchain-nace"><img alt="npm" src="https://img.shields.io/npm/v/langchain-nace?logo=npm&color=cb3837"></a>
  <a href="https://github.com/nace-ai/langchain-nace-js/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/nace-ai/langchain-nace-js/ci.yml?branch=main&label=CI&logo=github"></a>
  <a href="https://github.com/nace-ai/langchain-nace-js/actions/workflows/live.yml"><img alt="Live contract" src="https://img.shields.io/github/actions/workflow/status/nace-ai/langchain-nace-js/live.yml?branch=main&label=live%20contract&logo=github"></a>
  <img alt="Node 20+" src="https://img.shields.io/badge/node-%E2%89%A520-339933?logo=node.js&logoColor=white">
  <img alt="TypeScript" src="https://img.shields.io/badge/types-included-3178c6?logo=typescript&logoColor=white">
  <a href="https://github.com/nace-ai/langchain-nace-js/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>

---

**Drex** is Nace's family of decision models. It reads text, JSON, LangChain messages and, on Drex Max, images. It answers typed questions (yes/no, pick one, score on a rubric) with a probability for every option. This package brings it to LangChain.js:

|                                             |                                                                                                                                                                 |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🧭 **`NaceClassifier`**                     | A Runnable that answers many questions about one state in a single request, with full probabilities, usage and a request id.                                    |
| 🔀 **`NaceRouter` / `naceConditionalEdge`** | Route LCEL chains and LangGraph graphs on Drex's choice; send unsure cases to a person.                                                                         |
| 🛡️ **`naceAutoModeMiddleware`**             | Check every tool call before it runs: allow, ask a person, or block. Remembers the user's original request, reads screenshots, and blocks if Drex can't answer. |
| 💸 **`naceModelRouterMiddleware`**          | Pick the cheapest chat model that can handle each run; escalate when unsure.                                                                                    |
| 🔁 **Drop-in for `@langchain/typesafe`**    | Change the imports and the API key; your code keeps working.                                                                                                    |

**Contents:** [Install](#install) · [Quickstart](#quickstart) · [Switching from TypeSafe](#switching-from-langchaintypesafe) · [Models](#models) · [Images](#images-drex-max) · [Routing](#routing) · [Agent middleware](#agent-middleware-experimental) · [Backends](#backends) · [Errors](#errors) · [Tracing](#tracing) · [Development](#development)

## Install

```bash
npm install langchain-nace @langchain/core
npm install langchain @langchain/langgraph   # only for langchain-nace/middleware
export NACE_API_KEY=nace_sk_...
```

- **Node 20 or later.** ESM and CommonJS, with TypeScript types.
- **Peers:** `@langchain/core` ≥ 1.2.16; for the middleware, `langchain` ≥ 1.3.0 and `@langchain/langgraph`.
- **One copy of `@langchain/core` and `@langchain/langgraph`.** If your package manager installs two, add `overrides`/`resolutions`. With two copies, a `Command` from one isn't recognized by agents built on the other, so resuming after a person answers fails.

Looking for Python? The package of the same name is on [PyPI](https://pypi.org/project/langchain-nace/).

## Quickstart

```ts
import { NaceClassifier } from "langchain-nace";

const classifier = new NaceClassifier({
  // model: "drex-max" by default
  questions: {
    refund: { type: "noul", instructions: "Is the customer asking for a refund?" },
    team: {
      type: "choice",
      instructions: "Which team should handle this?",
      criteria: { billing: "Payments and refunds", technical: "Bugs" },
    },
    mood: {
      type: "score",
      instructions: "How upset is the customer?",
      criteria: ["calm", "annoyed", "furious"],
    },
  },
});

const response = await classifier.invoke(
  "I was charged twice for March and I'm annoyed. Please refund one.",
);
response.nouls.refund.noul; // 0.98: probability of yes
response.choices.team.probabilities; // { billing: 0.99, technical: 0.01 }
response.scores.mood.score; // 1.1: expected level
response.requestId; // quote it in support requests
```

| Question | Ask                                       | Get back                                                                   |
| -------- | ----------------------------------------- | -------------------------------------------------------------------------- |
| `noul`   | a yes/no proposition                      | `noul`: the probability of yes                                             |
| `choice` | pick one of your labels                   | `choice`, `confidence`, and `probabilities` for every label                |
| `score`  | a level on an ordered rubric (low → high) | `score` (may fall between levels), `legend`, `probabilities`, `confidence` |

All questions about one state go in one request, and answers keep their full probabilities, so thresholds stay in your code. The state can be a string, any JSON, a LangChain message, or messages anywhere inside objects and arrays. `batch` works as on any Runnable; it runs at most 8 requests at once unless you set `maxConcurrency`. To ask different questions on one call, pass `{ questions }` in the call options.

## Switching from `@langchain/typesafe`

Change the imports and set `NACE_API_KEY`. Question objects, constructor options, response fields, middleware names and middleware options all stay the same:

```diff
- import { TypeSafeClassifier } from "@langchain/typesafe";
- import { autoModeMiddleware, modelRouterMiddleware } from "@langchain/typesafe/middleware";
+ import { NaceClassifier } from "langchain-nace";
+ import { autoModeMiddleware, modelRouterMiddleware } from "langchain-nace/middleware";
```

| `@langchain/typesafe`                                                                                                  | `langchain-nace`                                              |
| ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| `TypeSafeClassifier`                                                                                                   | `NaceClassifier` (default model `drex-max`)                   |
| question objects; `Noul`, `Choice`, `Score`                                                                            | the same                                                      |
| constructor options (`questions`, `model`, `apiKey`, `baseUrl`, `timeout`, `fetch`, `maxRetries`, `maxConcurrency`, …) | the same                                                      |
| `model`, `answers`, `usage`, `requestId`, `nouls`, `choices`, `scores`                                                 | the same, plus `evaluationTimeMs`, `usage.imageTokens`, `raw` |
| `modelRouterMiddleware({ choices, instructions, classifierOptions })`                                                  | the same; the decision is in state under `modelRoute`         |
| `autoModeMiddleware({ tools, instructions, criteria, blockedMessage, classifierOptions })`                             | the same                                                      |
| `TypeSafe*Error`                                                                                                       | `Nace*Error`, with `.isInstance()`                            |
| `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL`                                                                                | `NACE_API_KEY`, `NACE_BASE_URL`                               |

What behaves differently, on purpose:

- **Different model, different numbers.** Thresholds tuned on Jev's probabilities may need retuning.
- **The timeout is 60 s** (TypeSafe: 30 s), because Drex may take up to 55 s.
- **The tool gate asks more.** Besides your `instructions`/`criteria` question, which still blocks at 0.5, Drex scores the risk and answers "destructive?" and "did the user ask for this?". The gate can also ask a person (see below).
- **The model router doesn't throw.** When Drex fails or isn't confident it uses the strongest model; with no user message, the agent's own model.
- **`dangerouslyAllowBrowser` defaults to `false`.**

## Models

| Model                    | Reads                 | Questions per request |
| ------------------------ | --------------------- | --------------------- |
| **`drex-max`** (default) | text, JSON and images | up to 16              |
| `drex-v2.0`              | text and JSON         | up to 512             |

Any other model id is sent as given (`drex-v1.5`, `drex-latest`, and models released later). Drex enforces each model's limits, and its errors say what to change. Pin a model with `model` or `NACE_DEFAULT_DECISION_MODEL`. The package default moves to the newest flagship with each Drex release, so pin one if you need results that never change.

## Images (Drex Max)

```ts
const damage = new NaceClassifier({
  questions: {
    damaged: {
      type: "choice",
      instructions: "Which image shows damage?",
      criteria: { "Image 1": null, "Image 2": null },
    },
  },
});
await damage.invoke("Two product photos.", {
  images: [readFileSync("a.png"), "data:image/jpeg;base64,..."],
});
```

- **Accepted:**
  - bytes (`Buffer`, `Uint8Array`, `ArrayBuffer`), file paths, base64 and data URLs;
  - LangChain image blocks (`{ type: "image", data, mimeType }`, `source_type: "base64"`, `image_url`);
  - provider blocks: OpenAI `input_image`, Anthropic `source`, Google `media`;
  - files uploaded with [`nace-sdk`](https://www.npmjs.com/package/nace-sdk).
- **Images inside messages** are moved to `images` automatically and referred to as "Image N". This works for message objects and for plain `{ role, content: [...] }` objects.
- **Never dropped silently.** Remote URLs are refused, because Drex doesn't fetch them. Images sent to a text-only model (`drex-v2.0`, `drex-v1.5`) throw `NaceImageModelError` before anything is sent.

## Routing

```ts
import { NaceRouter } from "langchain-nace";

const router = new NaceRouter({
  question: {
    type: "choice",
    instructions: "Which team?",
    criteria: { billing: null, technical: null },
  },
  state: (input: { text: string }) => input.text,
  minConfidence: 0.6,
  fallback: "human",
});
const chain = router.branch(
  { billing: billingChain, technical: techChain, human: escalate },
  escalate,
);
```

**LangGraph:** `naceConditionalEdge({ question, state, minConfidence: 0.7 })` is a path function for `addConditionalEdges`. Below the threshold it returns `"human_review"`, so that node can `interrupt()` for a person. See [`examples/langgraph-router.ts`](https://github.com/nace-ai/langchain-nace-js/blob/main/examples/langgraph-router.ts).

## Agent middleware (experimental)

```ts
import { MemorySaver } from "@langchain/langgraph";
import { createAgent } from "langchain";
import { naceAutoModeMiddleware, naceModelRouterMiddleware } from "langchain-nace/middleware";

const router = naceModelRouterMiddleware({
  choices: {
    fast: { model: "anthropic:claude-haiku-5-5", criteria: "Short lookups" },
    strong: { model: "anthropic:claude-sonnet-5-5", criteria: "Multi-step changes" },
  },
  instructions: "Pick the cheapest model that can do this safely.",
});
const gate = naceAutoModeMiddleware({ screenshots: true }); // drex-max reads the screenshots

const agent = createAgent({
  model: "anthropic:claude-haiku-5-5",
  tools,
  middleware: [router, gate],
  checkpointer: new MemorySaver(), // lets the gate pause for a person
});
```

### `naceAutoModeMiddleware`: the tool gate

Before each tool call, Drex gets one request with:

- the user's **original request**, kept however long the run gets;
- the latest user message and the last 20 messages;
- the tool call and its description;
- with `screenshots: true` and Drex Max, the newest screenshots.

It answers three questions: a risk score, "destructive?" and "did the user ask for this?". The default `thresholdPolicy()` then decides:

| Decision            | When                                          |
| ------------------- | --------------------------------------------- |
| 🛑 **Block**        | risk ≥ 0.75, or destructive and not asked for |
| ✋ **Ask a person** | risk ≥ 0.4, or destructive and asked for      |
| ✅ **Allow**        | otherwise                                     |
- **Asking a person** raises a LangGraph `interrupt` with a `{ type: "nace_tool_approval", name, args, verdict, … }` value. Resume with `new Command({ resume })`, where `resume` is `{ type: "approve" }`, `{ type: "reject", message }` or `{ type: "edit", args }`.
  - Asking needs a checkpointer. Without one the call is blocked, and the message says so. Use `askHuman: false` to always block instead.
- **If Drex can't answer, the call is blocked** (`onError: "block"`). A screenshot Drex can't read is dropped, and the check runs again without it.
- **Everything is replaceable:** pass `questions`, `thresholdPolicy({ ... })`, or your own `(verdict) => GateDecision`.
- **Resuming:** a pending human-approval interrupt keeps its verdict in memory while the tool and decision context stay the same. Completed calls and changed context are checked again. If context changes during a pending approval, the call is blocked and must be requested again; the old approval cannot authorize the changed context. A resume in another process asks Drex again.

### `naceModelRouterMiddleware`: the model picker

The router picks the model once per run. Below `minConfidence` (default 0.6), or if Drex can't be reached, it uses `escalateTo`: the last choice by default, so list choices cheapest first. The decision and probabilities are kept in state under `modelRoute`.

[`examples/support-agent.ts`](https://github.com/nace-ai/langchain-nace-js/blob/main/examples/support-agent.ts) runs four live scenarios:
- an allowed read;
- a blocked unrequested delete;
- a person approving a requested delete;
- a screenshot-based block on a computer-use click.

## Backends

| Backend                   | `baseUrl`                      | Notes                                                                                      |
| ------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------ |
| Nace (default)            | `https://console.nace.ai`      | or `NACE_BASE_URL`                                                                         |
| OpenRouter                | `https://openrouter.ai/api/v1` | models `nace-ai/drex-v2.0`, `nace-ai/drex-max`; images are sent the way OpenRouter expects |
| TypeSafe-compatible hosts | the host's System One root     | same request format as Nace                                                                |

Every request sends `User-Agent` and `X-Nace-Client: langchain-nace-js/<version>`. On OpenRouter it also sends `HTTP-Referer` and `X-OpenRouter-Title`.

Custom `defaultHeaders` are sent on requests but omitted from object inspection and serialized classifiers because they may contain credentials. Construct a new classifier with those headers when restoring a setup that needs them.

### Credentials and custom providers

`NACE_API_KEY` is read automatically only for HTTPS on port 443 at
`console.nace.ai` and its legacy production alias `drex.nace.ai`. Other hosts, including localhost, require an explicit
provider key. A blank explicit key is refused.

```ts
const classifier = new NaceClassifier({
  baseUrl: "https://openrouter.ai/api/v1",
  apiKey: process.env.OPENROUTER_API_KEY,
});
```

Pass the same settings through `classifierOptions` when constructing a router.
Vercel bypass headers are stripped for custom destinations. Requests refuse HTTP
redirects; configure the final API URL directly. Serialized classifiers keep the
resolved destination and use `OPENROUTER_API_KEY` for OpenRouter or
`NACE_CUSTOM_API_KEY` for other custom hosts.

**Upgrading from 0.1.0:** that version could send `NACE_API_KEY` to a custom
provider when no explicit `apiKey` was supplied. Upgrade to 0.1.1 or later.
If you used that configuration with real credentials, replace and revoke them;
an upgrade cannot remove copies already received by another provider. Recreate
and serialize custom-provider classifiers again: older saved manifests can
still refer to `NACE_API_KEY`, even after upgrading.

Malformed probabilities, confidence values or score rubrics raise `NaceAPIResponseValidationError`. The default tool-gate error policy blocks the call.

## Errors

Every error extends LangChain's `LangChainError`. Check errors with `.isInstance()`, which keeps working when two copies of the package are installed. Messages never include your state.

| Error                                                     | When                                                 |
| --------------------------------------------------------- | ---------------------------------------------------- |
| `NaceAuthenticationError`                                 | 401                                                  |
| `NaceInsufficientCreditError`                             | 402, with the Billing link                           |
| `NaceUnprocessableEntityError`                            | 422; lists each issue's `path: message`              |
| `NaceRateLimitError`                                      | 429, after retries (`retryAfterMs`)                  |
| `NaceOverloadedError`                                     | 529, after retries                                   |
| `NaceAPIConnectionError`, `NaceAPITimeoutError`           | network problems                                     |
| `NaceImageModelError`, `NaceImageError`, `NaceMediaError` | images or media that can't be sent; nothing was sent |
- **Retries:** 408, 429, 5xx and 529 responses are retried (`maxRetries: 2`), waiting as long as `retry-after-ms` asks. Each attempt may take up to 60 s.
- **Cancelling:** a call's `signal` or `timeout`, or a cancelled agent run, also aborts the HTTP request.

## Tracing

Calls show up in LangSmith as `llm` runs (`ls_provider: "nace"`) with the questions, the full answers and probabilities, token usage and the Drex request id. The router and the middlewares add their thresholds to the run metadata.

## Development

```bash
pnpm install
pnpm lint && pnpm typecheck && pnpm test     # offline, against a fake Drex
NACE_API_KEY_LIVE=nace_sk_... pnpm test:int  # live contract tests
pnpm build && pnpm smoke                     # install the packed tarball in a fresh project
```

## License

[MIT](https://github.com/nace-ai/langchain-nace-js/blob/main/LICENSE)
