import { defineConfig } from "tsup";
import { polyfillNode } from "esbuild-plugin-polyfill-node";
import { dependencies } from "./package.json";

const defineEnvs = (targetOptions, envs: { browser: boolean }) => {
  targetOptions["define"] = {
    ...targetOptions["define"],
    "process.env.IS_TEST": "false",
    "process.env.IS_BROWSER": Boolean(envs?.browser).toString(),
  };
  return targetOptions;
};

export default async () => [
  defineConfig({
    name: "node-cjs",
    entry: ["src/index.ts"],
    platform: "node",
    format: ["cjs"],
    sourcemap: true,
    esbuildOptions(options) {
      defineEnvs(options, { browser: false });
    },
    clean: true,
  }),
  defineConfig({
    name: "browser-iife",
    entry: ["src/index.ts"],
    platform: "browser",
    format: ["iife"],
    globalName: "Tabby",
    treeshake: "smallest",
    minify: true,
    sourcemap: true,
    esbuildPlugins: [
      polyfillNode({
        polyfills: { fs: true },
      }),
    ],
    esbuildOptions(options) {
      defineEnvs(options, { browser: true });
    },
    clean: true,
  }),
  defineConfig({
    name: "browser-esm",
    entry: ["src/index.ts"],
    platform: "browser",
    format: ["esm"],
    treeshake: true,
    sourcemap: true,
    esbuildPlugins: [
      polyfillNode({
        polyfills: { fs: true },
      }),
    ],
    esbuildOptions(options) {
      defineEnvs(options, { browser: true });
    },
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
    treeshake: "smallest",
    minify: true,
    sourcemap: true,
    esbuildOptions(options) {
      defineEnvs(options, { browser: false });
    },
    clean: true,
  }),
];
