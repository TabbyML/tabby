import { defineConfig } from "tsup";
import { polyfillNode } from "esbuild-plugin-polyfill-node";
import { dependencies } from "./package.json";

export default async () => [
  defineConfig({
    name: "lib-node",
    entry: ["src/index.ts"],
    platform: "node",
    format: ["cjs"],
    sourcemap: true,
    clean: true,
  }),
  defineConfig({
    name: "lib-browser",
    entry: ["src/index.ts"],
    platform: "browser",
    format: ["iife"],
    globalName: "Tabby",
    sourcemap: true,
    esbuildPlugins: [polyfillNode()],
    clean: true,
  }),
  defineConfig({
    name: "lib-typedefs",
    entry: ["src/index.ts"],
    dts: {
      only: true,
    },
    clean: true,
  }),
  defineConfig({
    name: "cli",
    entry: ["src/cli.ts"],
    platform: "node",
    noExternal: Object.keys(dependencies),
    minify: true,
    clean: true,
  }),
];
