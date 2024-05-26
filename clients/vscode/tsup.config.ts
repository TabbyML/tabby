import { defineConfig } from "tsup";
import path from "path";
import { getInstalledPath } from "get-installed-path";
import { copy } from "esbuild-plugin-copy";
import { polyfillNode } from "esbuild-plugin-polyfill-node";
import dedent from "dedent";
import { dependencies } from "./package.json";

const banner = dedent`
  /**
   * Tabby VSCode Extension
   * https://github.com/tabbyml/tabby/tree/main/clients/vscode
   * Copyright (c) 2023-2024 TabbyML, Inc.
   * Licensed under the Apache License 2.0.
   */`;

export default defineConfig(async () => {
  const tabbyAgentDist = path.join(await getInstalledPath("tabby-agent", { local: true }), "dist");
  return [
    {
      name: "node",
      entry: ["src/extension.ts"],
      outDir: "dist/node",
      platform: "node",
      target: "node18",
      treeshake: "smallest",
      minify: true,
      sourcemap: true,
      banner: {
        js: banner,
      },
      external: ["vscode"],
      noExternal: Object.keys(dependencies),
      esbuildPlugins: [
        copy({
          assets: { from: `${tabbyAgentDist}/**`, to: "dist/tabby-agent" },
          resolveFrom: "cwd",
        }),
      ],
      define: {
        "process.env.IS_BROWSER": "false",
      },
      clean: true,
    },
    {
      name: "browser",
      entry: ["src/extension.ts"],
      outDir: "dist/browser",
      platform: "browser",
      target: "node18",
      treeshake: "smallest",
      minify: true,
      sourcemap: true,
      banner: {
        js: banner,
      },
      external: ["vscode"],
      noExternal: Object.keys(dependencies),
      esbuildPlugins: [
        polyfillNode({
          polyfills: { fs: true },
        }),
      ],
      define: {
        "process.env.IS_BROWSER": "true",
      },
      clean: true,
    },
  ];
});
