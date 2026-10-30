// The lowest-versions CI job: fail unless the project, langchain and @langchain/langgraph
// all resolve one physical copy of @langchain/core and of @langchain/langgraph, at the
// versions given as arguments. Two installs of the same version pass a version check but
// break `instanceof` and interrupts, so the resolved package directories must be identical.
import { existsSync, readFileSync, realpathSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";

const [core, langgraph] = process.argv.slice(2);
const want = { "@langchain/core": core, "@langchain/langgraph": langgraph };

const readJson = (file) => JSON.parse(readFileSync(file, "utf8"));

/** The real directory of `pkg` as `importer` (a directory) resolves it. */
function packageDir(pkg, importer) {
  const require = createRequire(join(importer, "package.json"));
  try {
    return dirname(realpathSync(require.resolve(`${pkg}/package.json`)));
  } catch (error) {
    if (error?.code !== "ERR_PACKAGE_PATH_NOT_EXPORTED") throw error;
  }
  // `exports` hides package.json: resolve the main entry and walk up to the package root.
  let dir = dirname(realpathSync(require.resolve(pkg)));
  while (
    !(existsSync(join(dir, "package.json")) && readJson(join(dir, "package.json")).name === pkg)
  ) {
    const parent = dirname(dir);
    if (parent === dir) throw new Error(`No package.json found above the entry of ${pkg}`);
    dir = parent;
  }
  return dir;
}

const project = process.cwd();
const importers = {
  project,
  langchain: packageDir("langchain", project),
  "@langchain/langgraph": packageDir("@langchain/langgraph", project),
};

const seen = [];
for (const pkg of Object.keys(want)) {
  for (const [importer, dir] of Object.entries(importers)) {
    if (importer === pkg) continue; // a package does not import itself
    const path = packageDir(pkg, dir);
    seen.push({ pkg, importer, version: readJson(join(path, "package.json")).version, path });
  }
}
console.table(seen);

const wrongVersion = seen.filter(({ pkg, version }) => version !== want[pkg]);
const copies = Object.keys(want).flatMap((pkg) => {
  const paths = [...new Set(seen.filter((row) => row.pkg === pkg).map((row) => row.path))];
  return paths.length > 1 ? [{ pkg, paths }] : [];
});
if (wrongVersion.length > 0) console.error("Unexpected version:", wrongVersion);
if (copies.length > 0) console.error("More than one physical copy:", copies);
if (wrongVersion.length > 0 || copies.length > 0) process.exit(1);
