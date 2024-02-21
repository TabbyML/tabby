import path from "path";
import fs from "fs-extra";
import type { BuildOptions, Plugin } from "esbuild";
import { defineConfig } from "tsup";
import { copy } from "esbuild-plugin-copy";
import { polyfillNode } from "esbuild-plugin-polyfill-node";
import { dependencies } from "./package.json";
import dedent from "dedent";

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

function handleWinCaNativeBinaries(): Plugin {
  return {
    name: "handleWinCaNativeBinaries",
    setup: (build) => {
      build.onLoad({ filter: /win-ca\/lib\/crypt32-\w*.node$/ }, async (args) => {
        // As win-ca fallback is used, skip not required `.node` binaries
        return {
          contents: "",
          loader: "empty",
        };
      });
      build.onLoad({ filter: /win-ca\/lib\/fallback.js$/ }, async (args) => {
        // Copy `roots.exe` binary to `dist/win-ca`, and the LICENSE
        const binaryName = "roots.exe";
        const winCaPackagePath = path.join(path.dirname(args.path), "..");
        const license = await fs.readFile(path.join(winCaPackagePath, "LICENSE"));
        const packageJson = await fs.readJSON(path.join(winCaPackagePath, "package.json"));
        const exePath = path.join(path.dirname(args.path), binaryName);
        const outDir = path.join(build.initialOptions.outdir ?? "", "win-ca");
        build.onEnd(async () => {
          await fs.ensureDir(outDir);
          await fs.copyFile(exePath, path.join(outDir, binaryName));
          await fs.writeFile(
            path.join(outDir, "LICENSE"),
            dedent`
          win-ca v${packageJson.version}
          ${packageJson.homepage}

          ${license}
          `,
          );
        });
        return {};
      });
    },
  };
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
      markSideEffects(false, ["chokidar", "file-stream-rotator"]),
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
    banner: {
      js: dedent`
        /**
         * Tabby Agent
         * https://github.com/tabbyml/tabby/tree/main/clients/tabby-agent
         * Copyright (c) 2023-2024 TabbyML, Inc.
         * Licensed under the Apache License 2.0.
         */`,
    },
    esbuildPlugins: [
      copy({
        assets: [
          {
            from: "./wasm/*",
            to: "./wasm",
          },
        ],
      }),
      handleWinCaNativeBinaries(),
    ],
    esbuildOptions(options) {
      defineEnvs(options, { browser: false });
    },
    clean: true,
  }),
];
