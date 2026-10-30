// The lowest-versions CI job: fail unless the package, langchain and
// @langchain/langgraph all resolve one copy of @langchain/core and of
// @langchain/langgraph, at the versions given as arguments.
import { realpathSync } from "node:fs";
import { createRequire } from "node:module";

const [core, langgraph] = process.argv.slice(2);
const want = { "@langchain/core": core, "@langchain/langgraph": langgraph };
const version = (pkg, dir) => createRequire(`${realpathSync(dir)}/`)(`${pkg}/package.json`).version;
const seen = [
  ["@langchain/langgraph", "."],
  ["@langchain/langgraph", "node_modules/langchain"],
  ["@langchain/core", "."],
  ["@langchain/core", "node_modules/langchain"],
  ["@langchain/core", "node_modules/@langchain/langgraph"],
].map(([pkg, dir]) => ({ pkg, from: dir, version: version(pkg, dir) }));
console.table(seen);
const wrong = seen.filter(({ pkg, version }) => version !== want[pkg]);
if (wrong.length > 0) {
  console.error("More than one copy, or an unexpected version:", wrong);
  process.exit(1);
}
