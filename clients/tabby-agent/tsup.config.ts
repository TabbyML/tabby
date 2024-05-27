import { defineConfig } from "tsup";
import path from "path";
import fs from "fs-extra";
import type { Plugin } from "esbuild";
import { polyfillNode } from "esbuild-plugin-polyfill-node";
import dedent from "dedent";

function processWinCa(copyRootsExe: boolean = false): Plugin {
  return {
    name: "processWinCa",
    setup: (build) => {
      build.onLoad({ filter: /win-ca\/lib\/crypt32-\w*.node$/ }, async () => {
        // As win-ca fallback is used, skip not required `.node` binaries
        return {
          contents: "",
          loader: "empty",
        };
      });
      build.onLoad({ filter: /win-ca\/lib\/fallback.js$/ }, async (args) => {
        if (copyRootsExe) {
          // Copy `roots.exe` binary to `dist/win-ca`, and the LICENSE
          const binaryName = "roots.exe";
          const winCaPackagePath = path.join(path.dirname(args.path), "..");
          const license = await fs.readFile(path.join(winCaPackagePath, "LICENSE"));
          const packageJson = await fs.readJSON(path.join(winCaPackagePath, "package.json"));
          const exePath = path.join(path.dirname(args.path), binaryName);
          const outDir = path.join(build.initialOptions.outdir ?? "", "win-ca");
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
          return {};
        }
      });
    },
  };
}

const banner = dedent`
  /**
   * Tabby Agent
   * https://github.com/tabbyml/tabby/tree/main/clients/tabby-agent
   * Copyright (c) 2023-2024 TabbyML, Inc.
   * Licensed under the Apache License 2.0.
   */`;

export default defineConfig(async () => {
  return [
    {
      name: "lsp-protocol",
      entry: ["src/lsp/protocol.ts"],
      dts: true,
      banner: {
        js: banner,
      },
    },
    {
      name: "lsp-node",
      entry: ["src/lsp/index.ts"],
      outDir: "dist/node",
      platform: "node",
      target: "node18",
      sourcemap: true,
      banner: {
        js: banner,
      },
      define: {
        "process.env.IS_TEST": "false",
        "process.env.IS_BROWSER": "false",
      },
      esbuildPlugins: [processWinCa(true)],
      clean: true,
    },
    {
      name: "lsp-browser",
      entry: ["src/lsp/index.ts"],
      outDir: "dist/browser",
      platform: "browser",
      format: "esm",
      treeshake: "smallest", // Required for browser to cleanup fs related libs
      sourcemap: true,
      banner: {
        js: banner,
      },
      external: ["glob", "fs-extra", "chokidar", "file-stream-rotator", "win-ca", "mac-ca"],
      define: {
        "process.env.IS_TEST": "false",
        "process.env.IS_BROWSER": "true",
      },
      esbuildPlugins: [
        processWinCa(),
        polyfillNode({
          polyfills: {},
        }),
      ],
      clean: true,
    },
  ];
});
