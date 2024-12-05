import { defineConfig } from "tsup";
import type { Plugin, PluginBuild } from "esbuild";
import path from "path";
import fs from "fs-extra";
import { polyfillNode } from "esbuild-plugin-polyfill-node";
import dedent from "dedent";

function processWinCa(copyRootsExe: boolean = false): Plugin {
  return {
    name: "processWinCa",
    setup: (build: PluginBuild) => {
      build.onLoad({ filter: /win-ca[\\/]lib[\\/]crypt32-\w*.node$/ }, async () => {
        // As win-ca fallback is used, skip not required `.node` binaries
        return {
          contents: "",
          loader: "empty",
        };
      });
      build.onLoad({ filter: /win-ca[\\/]lib[\\/]fallback.js$/ }, async (args) => {
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
      entry: ["src/protocol.ts"],
      dts: true,
      external: ["vscode-languageserver-protocol"],
      banner: {
        js: banner,
      },
    },
    {
      name: "lsp-node",
      loader: {
        ".md": "text",
      },
      entry: ["src/index.ts"],
      outDir: "dist/node",
      platform: "node",
      target: "node18",
      sourcemap: true,
      clean: true,
      define: {
        "process.env.IS_TEST": "false",
        "process.env.IS_BROWSER": "false",
      },
      treeshake: {
        preset: "smallest",
        moduleSideEffects: "no-external",
      },
      external: ["vscode-languageserver/browser"],
      esbuildPlugins: [processWinCa(true)],
      banner: {
        js: banner,
      },
    },
    {
      name: "lsp-browser",
      loader: {
        ".md": "text",
      },
      entry: ["src/index.ts"],
      outDir: "dist/browser",
      platform: "browser",
      format: "esm",
      sourcemap: true,
      clean: true,
      define: {
        "process.env.IS_TEST": "false",
        "process.env.IS_BROWSER": "true",
      },
      treeshake: {
        preset: "smallest",
        moduleSideEffects: "no-external",
      },
      external: [
        "glob",
        "fs-extra",
        "chokidar",
        "file-stream-rotator",
        "win-ca",
        "mac-ca",
        "vscode-languageserver/node",
        "undici",
      ],
      esbuildPlugins: [
        polyfillNode({
          polyfills: {},
        }),
      ],
      esbuildOptions: (options) => {
        // disable warning for `import is undefined`:
        // src/lsp/index.ts:9:6:
        // 9 |  dns.setDefaultResultOrder("ipv4first");
        options.logOverride = { "import-is-undefined": "info" };
      },
      banner: {
        js: banner,
      },
    },
  ];
});
