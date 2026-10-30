# Changelog

## 0.1.1

- Require explicit provider keys outside the production Nace HTTPS hosts; reject blank keys.
- Pin the resolved destination in serialization and use provider-specific secret references.
- Strip Vercel bypass headers for custom hosts and refuse HTTP redirects.
- Match OpenRouter by exact hostname, including its DNS trailing-dot form.
- Document upgrading from 0.1.0 and replacing credentials used with custom providers.

## 0.1.0

First release.

Before release, a review found and fixed: an API key printed by `console.log(router)`; the tool gate reusing a cached verdict for a different call with the same tool call id; OpenAI `input_image`, Anthropic `source` and Google `media` image blocks sent as text; cancellation and `timeout` not aborting the HTTP request; an unsendable screenshot blocking every later call; `__proto__` keys dropped; noul criteria with unknown keys dropped silently.

- `NaceClassifier`: a Runnable that asks Drex typed questions (`noul`, `choice`, `score`) about text, JSON, LangChain messages and images, and returns full probabilities, usage and the request id. Default model `drex-max`.
- Images on Drex Max from bytes, paths, base64, data URLs, LangChain image blocks and `nace-sdk` uploads; images inside messages (objects or `{ role, content }`) are moved to `images` as "Image N".
- `NaceRouter` with `.branch()`, and `naceConditionalEdge` for LangGraph with a `human_review` fallback.
- `langchain-nace/middleware`: `naceAutoModeMiddleware` (tool gate: risk, destructive, asked-for; block, ask a person, or allow; fails closed) and `naceModelRouterMiddleware` (one Drex choice per run; escalates when unsure or on errors).
- Drop-in for `@langchain/typesafe`: same question objects, options, response fields, and middleware names (`autoModeMiddleware`, `modelRouterMiddleware`).
- OpenRouter backend; attribution headers `User-Agent` and `X-Nace-Client: langchain-nace-js/<version>`.
