import { defineConfig } from "tsup";
import { copy } from "esbuild-plugin-copy";
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
