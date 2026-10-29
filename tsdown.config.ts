import { defineConfig } from "tsdown";

export default defineConfig({
  entry: ["./src/index.ts", "./src/middleware/index.ts"],
  format: ["esm", "cjs"],
  dts: true,
  clean: true,
  platform: "node",
  target: "node20",
  sourcemap: true,
});
