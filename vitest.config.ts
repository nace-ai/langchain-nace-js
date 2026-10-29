import { configDefaults, defineConfig } from "vitest/config";

export default defineConfig(({ mode }) => {
  const live = mode === "int";
  return {
    test: {
      environment: "node",
      setupFiles: ["dotenv/config"],
      include: live ? ["tests/int/**/*.int.test.ts"] : ["tests/unit/**/*.test.ts"],
      exclude: [...configDefaults.exclude],
      testTimeout: live ? 180_000 : 20_000,
      hookTimeout: live ? 180_000 : 20_000,
      hideSkippedTests: !live,
      typecheck: live ? undefined : { enabled: true, include: ["tests/unit/**/*.test-d.ts"] },
    },
  };
});
