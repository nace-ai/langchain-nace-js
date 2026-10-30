# Changelog

## 0.1.1

- Save each tool-gate decision in the agent's checkpoint (a `NaceGateDecision` task) and replay it on resume, on any worker, without asking Drex again. A pending approval no longer survives a change in the call, its context or the policy, and a lower-risk recheck can no longer bypass a pending rejection, including for custom decisions. Editing the paused state or resuming an earlier checkpoint makes the gate check the call again as a new call, and an answer that is already waiting blocks it instead of applying to a new decision. A custom `GateDecision` needs an `id` to be resumed after a pause. Screenshots are captured inside that task on a first run, so a failing screenshot source is a failed check like any other. With `onError: "throw"`, a failed check fails the run and retrying the run checks again; in runs with a `thread_id`, an outer middleware can't catch it and continue (without one, or outside a LangGraph run, the gate decides directly, and its error is an ordinary error). With `onError: "block"`, the block is saved and replayed. A resumed call whose screenshots can't be captured again is blocked. Put the gate before retry middleware, and don't call LangGraph tasks or interrupts from a `screenshots` function.
- Test the oldest supported `@langchain/core`, `@langchain/langgraph` and `langchain` releases with a single copy of each.
- Exclude custom headers from inspection and serialization while preserving outgoing request headers.
- Reject out-of-range probabilities, confidences and scores, and invalid score rubrics.
- Preserve special answer IDs in typed response accessors.
- Require explicit provider keys outside the production Nace HTTPS hosts; reject blank keys.
- Pin the resolved destination in serialization and use provider-specific secret references.
- Strip Vercel bypass headers for custom hosts and refuse HTTP redirects.
- Match OpenRouter by exact hostname, including its DNS trailing-dot form.
- Document upgrading from 0.1.0 and replacing credentials used with custom providers.
- Refuse a response that doesn't answer the questions asked: a missing answer, an answer of another type, a choice that was not offered, or a score legend of a different length than the rubric. Extra answers and unknown question types are still accepted. The error names the question (quoted, at most 64 characters) and never repeats text from the response.
- Read the tool gate's risk against the rubric that was asked: an answer with a legend of another length is blocked as unchecked, not scaled by its own legend (a 100-level legend turned "Severe" into 0.04 and allowed the call).
- `naceConditionalEdge` refuses an answer that is not one of the question's labels, so a label such as `__end__` can no longer pick the next node.
- Keep API error messages short: an HTML error page from a gateway becomes one line, other text is cut to 500 characters, and the billing link and request id are added after the cut. The full `body` and every issue stay on the error. The error's `cause` is a one-line summary instead of the SDK's error, so printing an error no longer prints the whole response.
- A 402 from OpenRouter points to OpenRouter's credits page (`OPENROUTER_CREDITS_URL`) instead of the Drex billing page.
- Refuse `modelKwargs` that set `model`, `state`, `questions` or `images`: `images` used to reach a text-only model and skip its refusal.

## 0.1.0

First release.

Before release, a review found and fixed: an API key printed by `console.log(router)`; the tool gate reusing a cached verdict for a different call with the same tool call id; OpenAI `input_image`, Anthropic `source` and Google `media` image blocks sent as text; cancellation and `timeout` not aborting the HTTP request; an unsendable screenshot blocking every later call; `__proto__` keys dropped; noul criteria with unknown keys dropped silently.

- `NaceClassifier`: a Runnable that asks Drex typed questions (`noul`, `choice`, `score`) about text, JSON, LangChain messages and images, and returns full probabilities, usage and the request id. Default model `drex-max`.
- Images on Drex Max from bytes, paths, base64, data URLs, LangChain image blocks and `nace-sdk` uploads; images inside messages (objects or `{ role, content }`) are moved to `images` as "Image N".
- `NaceRouter` with `.branch()`, and `naceConditionalEdge` for LangGraph with a `human_review` fallback.
- `langchain-nace/middleware`: `naceAutoModeMiddleware` (tool gate: risk, destructive, asked-for; block, ask a person, or allow; fails closed) and `naceModelRouterMiddleware` (one Drex choice per run; escalates when unsure or on errors).
- Drop-in for `@langchain/typesafe`: same question objects, options, response fields, and middleware names (`autoModeMiddleware`, `modelRouterMiddleware`).
- OpenRouter backend; attribution headers `User-Agent` and `X-Nace-Client: langchain-nace-js/<version>`.
