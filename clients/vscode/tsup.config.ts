import { defineConfig } from "tsup";
import { polyfillNode } from "esbuild-plugin-polyfill-node";
import { dependencies } from "./package.json";

export default () => [
  defineConfig({
    name: "node",
    entry: ["src/extension.ts"],
    outDir: "dist/node",
    platform: "node",
    external: ["vscode"],
    noExternal: Object.keys(dependencies),
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
