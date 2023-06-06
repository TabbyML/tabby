import { defineConfig } from "tsup";
import { polyfillNode } from "esbuild-plugin-polyfill-node";
import { dependencies } from "./package.json";

export default async () => [
  defineConfig({
    name: "node-cjs",
    entry: ["src/index.ts"],
    platform: "node",
    format: ["cjs"],
    sourcemap: true,
    clean: true,
  }),
  defineConfig({
    name: "browser-iife",
    entry: ["src/index.ts"],
    platform: "browser",
    format: ["iife"],
    globalName: "Tabby",
    minify: true,
    sourcemap: true,
    esbuildPlugins: [
      polyfillNode({
        polyfills: { fs: true },
      }),
    ],
    clean: true,
  }),
  defineConfig({
    name: "browser-esm",
    entry: ["src/index.ts"],
    platform: "browser",
    format: ["esm"],
    // FIXME: bundle all dependencies to reduce module resolving problems, not a good solution
    noExternal: Object.keys(dependencies),
    sourcemap: true,
    esbuildPlugins: [
      polyfillNode({
        polyfills: { fs: true },
      }),
    ],
    clean: true,
  }),
  defineConfig({
    name: "type-defs",
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
    sourcemap: true,
    clean: true,
  }),
];
