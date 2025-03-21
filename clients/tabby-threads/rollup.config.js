import typescript from "@rollup/plugin-typescript";
import resolve from "@rollup/plugin-node-resolve";
import { defineConfig } from "rollup";

export default defineConfig([{
  input: "source/index.ts",
  output: [
    {
      dir: "dist",
      format: "esm",
      entryFileNames: "esm/[name].mjs",
      preserveModules: true,
      preserveModulesRoot: "source",
    },
    {
      dir: "dist",
      format: "cjs",
      entryFileNames: "cjs/[name].cjs",
      preserveModules: true,
      preserveModulesRoot: "source",
    },
  ],
  plugins: [
    typescript({
      tsconfig: "./tsconfig.json",
      noEmitOnError: true,
    }),
  ],
  external: ["@quilted/events", "@preact/signals"],
}]);
