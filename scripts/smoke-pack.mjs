#!/usr/bin/env node
/**
 * Install the package the way a user would and check it works.
 *
 *   node scripts/smoke-pack.mjs              # pack this checkout
 *   node scripts/smoke-pack.mjs --registry   # install langchain-nace@<version> from npm
 *
 * In a fresh temp project: installs the tarball (or the published version) with
 * its peers, imports it from ESM and CommonJS, type-checks a TypeScript user
 * file, and, when NACE_API_KEY is set, makes real calls (text, image, router).
 */
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const pkg = JSON.parse(readFileSync(join(root, "package.json"), "utf8"));
const fromRegistry = process.argv.includes("--registry");
const run = (cmd, args, cwd) => execFileSync(cmd, args, { cwd, stdio: "inherit" });

const dir = mkdtempSync(join(tmpdir(), "langchain-nace-smoke-"));
let spec = `${pkg.name}@${pkg.version}`;
if (!fromRegistry) {
  const out = execFileSync("npm", ["pack", "--pack-destination", dir, "--json"], { cwd: root });
  spec = join(dir, JSON.parse(out.toString())[0].filename);
}
console.log(`smoke: installing ${spec} into ${dir}`);
writeFileSync(
  join(dir, "package.json"),
  JSON.stringify({ name: "smoke", private: true, type: "module" }),
);
run(
  "npm",
  [
    "install",
    "--no-audit",
    "--no-fund",
    spec,
    "@langchain/core",
    "langchain",
    "@langchain/langgraph",
    "typescript@5",
    "@types/node",
  ],
  dir,
);

const red = readFileSync(join(root, "tests/fixtures/red.png")).toString("base64");

writeFileSync(
  join(dir, "esm.mjs"),
  `
import { NaceClassifier, NaceError, VERSION } from "langchain-nace";
import { naceAutoModeMiddleware, naceModelRouterMiddleware } from "langchain-nace/middleware";
if (VERSION !== ${JSON.stringify(pkg.version)}) throw new Error("wrong version " + VERSION);
if (typeof naceAutoModeMiddleware !== "function" || typeof naceModelRouterMiddleware !== "function") throw new Error("middleware missing");
console.log("esm: imports ok", VERSION);
if (process.env.NACE_API_KEY) {
  const c = new NaceClassifier({ questions: {
    refund: { type: "noul", instructions: "Is the customer asking for a refund?" },
    team: { type: "choice", instructions: "Which team?", criteria: { billing: null, technical: null } },
  } });
  const r = await c.invoke("I was charged twice, please refund one.");
  if (!(r.nouls.refund.noul > 0.7) || r.choices.team.choice !== "billing") throw new Error("unexpected answers");
  console.log("esm: live text ok", r.model, r.requestId);
  const img = await new NaceClassifier({ questions: { red: { type: "noul", instructions: "Is Image 1 red?" } } })
    .invoke("One image.", { images: [${JSON.stringify(red)}] });
  if (!(img.nouls.red.noul > 0.7)) throw new Error("image not read");
  console.log("esm: live image ok", img.model);
  const v2 = new NaceClassifier({ model: "drex-v2.0", questions: { q: { type: "noul", instructions: "?" } } });
  try { await v2.invoke("x", { images: [${JSON.stringify(red)}] }); throw new Error("v2 accepted an image"); }
  catch (e) { if (!NaceError.isInstance(e)) throw e; }
  console.log("esm: text-only refusal ok");
}
`,
);

writeFileSync(
  join(dir, "cjs.cjs"),
  `
const { NaceClassifier, NaceRateLimitError } = require("langchain-nace");
const { naceAutoModeMiddleware } = require("langchain-nace/middleware");
if (typeof NaceClassifier !== "function" || typeof NaceRateLimitError.isInstance !== "function") throw new Error("cjs exports missing");
if (typeof naceAutoModeMiddleware !== "function") throw new Error("cjs middleware missing");
console.log("cjs: imports ok");
`,
);

writeFileSync(
  join(dir, "user.ts"),
  `
import { NaceClassifier, type ClassificationResponse, type Noul } from "langchain-nace";
import { naceAutoModeMiddleware, type GateVerdict } from "langchain-nace/middleware";
const q: Noul = { type: "noul", instructions: "Urgent?" };
const c = new NaceClassifier({ questions: { q }, apiKey: "nace_sk_x" });
const f = async (): Promise<number | undefined> => {
  const r: ClassificationResponse = await c.invoke("x");
  return r.nouls.q?.noul;
};
const g = (v: GateVerdict) => v.risk;
void f; void g; void naceAutoModeMiddleware;
`,
);
writeFileSync(
  join(dir, "tsconfig.json"),
  JSON.stringify({
    compilerOptions: {
      module: "NodeNext",
      moduleResolution: "NodeNext",
      target: "ES2022",
      strict: true,
      noEmit: true,
      skipLibCheck: false,
      types: ["node"],
    },
    files: ["user.ts"],
  }),
);

run("node", ["esm.mjs"], dir);
run("node", ["cjs.cjs"], dir);
run("npx", ["tsc", "-p", "."], dir);
console.log("smoke: all checks passed");
