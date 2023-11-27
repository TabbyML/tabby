import { defineConfig } from "tsup";
import { copy } from "esbuild-plugin-copy";
import { polyfillNode } from "esbuild-plugin-polyfill-node";
import { dependencies } from "./package.json";

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
            from: "../tabby-agent/dist/wasm/*.wasm",
            to: "./wasm",
          },
        ],
      }),
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
