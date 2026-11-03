# Third-party notices

`@nace-ai/langchain-nace` includes material adapted from the following package.

## @langchain/typesafe 0.0.2

- Package: [`@langchain/typesafe`](https://www.npmjs.com/package/@langchain/typesafe), version 0.0.2
- Licence: MIT
- Copyright: Copyright (c) LangChain, Inc.

`@nace-ai/langchain-nace` is a drop-in replacement for `@langchain/typesafe`, and keeps its API shape so that
existing code only needs new imports and an API key. The following material was adapted from it:

- Doc comments reproduced word for word from its type declarations:
  - `src/types.ts`: "Any value expressible in JSON."; "Noul answers only, keyed by question id.
    Non-enumerable."; the matching "Choice answers only..." and "Score answers only..." comments.
  - `src/state.ts`: "A value legal anywhere below the root. `Date` becomes an ISO string."
  - `src/classifier.ts`: "Custom transport, for tests, proxies or shared connection pools."
  - `src/errors.ts`: "Base class for every error this package throws."
- The README samples that `tests/unit/compat.test.ts` reproduces, with only the imports and the API
  key variable changed, to check that code written for `@langchain/typesafe` keeps working.

The full licence text of `@langchain/typesafe`, copied unchanged from the published 0.0.2 package:

---

MIT License

Copyright (c) LangChain, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
