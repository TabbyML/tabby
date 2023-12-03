import type { BuildOptions, Plugin } from "esbuild";
import { defineConfig } from "tsup";
import { copy } from "esbuild-plugin-copy";
import { polyfillNode } from "esbuild-plugin-polyfill-node";
import { dependencies } from "./package.json";

function markSideEffects(value: boolean, packages: string[]): Plugin {
  return {
    name: "sideEffects",
    setup: (build) => {
      build.onResolve({ filter: /. */ }, async (args) => {
        if (args.pluginData || !packages.includes(args.path)) {
          return;
        }
        const { path, ...rest } = args;
        rest.pluginData = true;
        const result = await build.resolve(path, rest);
        result.sideEffects = value;
        return result;
      });
    },
  };
}

function defineEnvs(targetOptions: BuildOptions, envs: { browser: boolean }) {
  targetOptions["define"] = {
    ...targetOptions["define"],
    "process.env.IS_TEST": "false",
    "process.env.IS_BROWSER": Boolean(envs?.browser).toString(),
  };
  return targetOptions;
}

export default async () => [
  defineConfig({
    name: "node-cjs",
    entry: ["src/index.ts"],
    platform: "node",
    target: "node18",
    format: ["cjs"],
    sourcemap: true,
    esbuildOptions(options) {
      defineEnvs(options, { browser: false });
    },
    clean: true,
  }),
  defineConfig({
    name: "browser-esm",
    entry: ["src/index.ts"],
    platform: "browser",
    format: ["esm"],
    treeshake: "smallest", // To remove unused libraries in browser.
    sourcemap: true,
    esbuildPlugins: [
      polyfillNode({
        polyfills: { fs: true },
      }),
      // Mark sideEffects false for tree-shaking unused libraries in browser.
      markSideEffects(false, ["chokidar", "rotating-file-stream"]),
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
    target: "node18",
    noExternal: Object.keys(dependencies),
    treeshake: "smallest",
    minify: true,
    sourcemap: true,
    esbuildPlugins: [
      copy({
        assets: [
          {
            from: "./wasm/*",
            to: "./wasm",
          },
        ],
      }),
    ],
    esbuildOptions(options) {
      defineEnvs(options, { browser: false });
    },
    clean: true,
  }),
];
