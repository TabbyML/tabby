import { defineConfig } from "tsup";
import { copy } from "esbuild-plugin-copy";
import { dependencies } from "./package.json";

export default () => [
  defineConfig({
    name: "node",
    entry: ["src/extension.ts"],
    outDir: "dist",
    platform: "node",
    target: "node18",
    external: ["vscode"],
    noExternal: Object.keys(dependencies),
    clean: true,
    esbuildPlugins: [
      copy({
        assets: [
          {
            from: "../tabby-agent/dist/cli.js",
            to: "./server/tabby-agent.js",
          },
          {
            from: "../tabby-agent/dist/wasm/*",
            to: "./server/wasm",
          },
        ],
      }),
    ],
  }),
];
