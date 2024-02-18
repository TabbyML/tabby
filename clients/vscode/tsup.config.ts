import type { Plugin } from "esbuild";
import { defineConfig } from "tsup";
import { copy } from "esbuild-plugin-copy";
import { polyfillNode } from "esbuild-plugin-polyfill-node";
import { dependencies } from "./package.json";

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
    },
  };
}

export default () => [
  defineConfig({
    name: "node",
    entry: ["src/extension.ts"],
    outDir: "dist/node",
    platform: "node",
    target: "node18",
    external: ["vscode"],
    noExternal: Object.keys(dependencies),
    esbuildPlugins: [
      copy({
        assets: [
          {
            from: "../tabby-agent/dist/wasm/*",
            to: "./wasm",
          },
          {
            from: "../tabby-agent/dist/win-ca/*",
            to: "./win-ca",
          },
        ],
      }),
      handleWinCaNativeBinaries(),
    ],
    clean: true,
  }),
  defineConfig({
    name: "browser",
    entry: ["src/extension.ts"],
    outDir: "dist/web",
    platform: "browser",
    external: ["vscode"],
    noExternal: Object.keys(dependencies),
    esbuildPlugins: [
      polyfillNode({
        polyfills: { fs: true },
      }),
    ],
    clean: true,
  }),
];
