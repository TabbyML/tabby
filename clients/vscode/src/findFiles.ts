// We want to use the findFiles2 API, but it is still in the proposed state.
// Therefore, we implement a findFiles function here that can use the following rules in the exclude pattern:
// 1. .gitignore files
// 2. Settings from `files.exclude` and `search.exclude`
//
// See https://github.com/microsoft/vscode/blob/main/src/vscode-dts/vscode.proposed.findFiles2.d.ts

import { GlobPattern, RelativePattern, Uri, WorkspaceFolder, CancellationToken, workspace } from "vscode";
import path from "path";
import { getLogger } from "./logger";

const logger = getLogger("FindFiles");

// Map from workspace folder to gitignore patterns
const gitIgnorePatternsMap = new Map<string, Set<string>>();

function gitIgnoreItemToExcludePatterns(item: string, prefix?: string | undefined): string[] {
  let pattern = item.trim();
  if (pattern.length === 0) {
    return [];
  }
  if (pattern.indexOf("/") === -1 || pattern.indexOf("/") === pattern.length - 1) {
    if (!pattern.startsWith("**/")) {
      pattern = `**/${pattern}`;
    }
  } else if (pattern.startsWith("/")) {
    pattern = pattern.slice(1);
  }
  return [path.join(prefix ?? "", pattern), path.join(prefix ?? "", pattern, "/**")];
}

async function buildGitIgnorePatterns(workspaceFolder: WorkspaceFolder) {
  const patterns = new Set<string>();
  logger.debug(`Building gitignore patterns for workspace folder: ${workspaceFolder.uri.toString()}`);

  // Read parent gitignore files
  let current = workspaceFolder.uri;
  let parent = current.with({ path: path.dirname(current.path) });
  while (parent.path !== current.path) {
    const gitignore = parent.with({ path: path.join(parent.path, ".gitignore") });
    try {
      const content = (await workspace.fs.readFile(gitignore)).toString();
      content.split(/\r?\n/).forEach((line) => {
        if (!line.trim().startsWith("#")) {
          gitIgnoreItemToExcludePatterns(line).forEach((pattern) => patterns.add(pattern));
        }
      });
    } catch (error) {
      // ignore
    }

    // next
    current = parent;
    parent = current.with({ path: path.dirname(current.path) });
  }

  // Read subdirectories gitignore files
  const ignoreFiles = await workspace.findFiles(new RelativePattern(workspaceFolder, "**/.gitignore"));
  await Promise.all(
    ignoreFiles.map(async (ignoreFile) => {
      const prefix = path.relative(workspaceFolder.uri.path, path.dirname(ignoreFile.path));
      try {
        const content = (await workspace.fs.readFile(ignoreFile)).toString();
        content.split(/\r?\n/).forEach((line) => {
          if (!line.trim().startsWith("#")) {
            gitIgnoreItemToExcludePatterns(line, prefix).forEach((pattern) => patterns.add(pattern));
          }
        });
      } catch (error) {
        // ignore
      }
    }),
  );
  // Update map
  logger.debug(
    `Completed building git ignore patterns for workspace folder: ${workspaceFolder.uri.toString()}, git ignore patterns: ${JSON.stringify([...patterns])}`,
  );
  gitIgnorePatternsMap.set(workspaceFolder.uri.toString(), patterns);
}

workspace.onDidChangeTextDocument(async (event) => {
  const uri = event.document.uri;
  if (path.basename(uri.fsPath) === ".gitignore") {
    const workspaceFolder = workspace.getWorkspaceFolder(uri);
    if (workspaceFolder) {
      await buildGitIgnorePatterns(workspaceFolder);
    }
  }
});

export async function findFiles(
  pattern: GlobPattern,
  options?: {
    excludes?: string[];
    noUserSettings?: boolean; // User settings is used by default, set to true to skip user settings
    noIgnoreFiles?: boolean; // .gitignore files are used by default, set to true to skip .gitignore files
    maxResults?: number;
    token?: CancellationToken;
  },
): Promise<Uri[]> {
  const combinedExcludes = new Set<string>();
  if (options?.excludes) {
    for (const exclude of options.excludes) {
      combinedExcludes.add(exclude);
    }
  }
  if (!options?.noUserSettings) {
    const searchExclude: Record<string, boolean> =
      (await workspace.getConfiguration("search", null).get("exclude")) ?? {};
    const filesExclude: Record<string, boolean> =
      (await workspace.getConfiguration("files", null).get("exclude")) ?? {};
    for (const pattern in { ...searchExclude, ...filesExclude }) {
      if (filesExclude[pattern]) {
        combinedExcludes.add(pattern);
      }
    }
  }
  if (options?.noIgnoreFiles) {
    const excludesPattern = `{${[...combinedExcludes].join(",")}}`;
    logger.debug(
      `Executing search: ${JSON.stringify({ includePattern: pattern, excludesPattern, maxResults: options?.maxResults, token: options?.token })}`,
    );
    return await workspace.findFiles(pattern, excludesPattern, options.maxResults, options.token);
  } else {
    await Promise.all(
      workspace.workspaceFolders?.map(async (workspaceFolder) => {
        if (!gitIgnorePatternsMap.has(workspaceFolder.uri.toString())) {
          await buildGitIgnorePatterns(workspaceFolder);
        }
      }) ?? [],
    );

    return new Promise((resolve, reject) => {
      if (options?.token) {
        options?.token.onCancellationRequested((reason) => {
          reject(reason);
        });
      }

      const allSearches =
        workspace.workspaceFolders?.map(async (workspaceFolder) => {
          let includePattern: RelativePattern;
          if (typeof pattern === "string") {
            includePattern = new RelativePattern(workspaceFolder, pattern);
          } else {
            if (pattern.baseUri.toString().startsWith(workspaceFolder.uri.toString())) {
              includePattern = pattern;
            } else {
              return [];
            }
          }
          const allExcludes = new Set([
            ...combinedExcludes,
            ...(gitIgnorePatternsMap.get(workspaceFolder.uri.toString()) ?? []),
          ]);
          const excludesPattern = `{${[...allExcludes].slice(0, 1000).join(",")}}`; // Limit to 1000 patterns
          logger.debug(
            `Executing search: ${JSON.stringify({ includePattern, excludesPattern, maxResults: options?.maxResults, token: options?.token })}`,
          );
          return await workspace.findFiles(includePattern, excludesPattern, options?.maxResults, options?.token);
        }) ?? [];

      const results: Uri[] = [];
      Promise.all(
        allSearches.map(async (search) => {
          try {
            const result = await search;
            if (result.length > 0) {
              results.push(...result);
              if (options?.maxResults && results.length >= options.maxResults) {
                resolve(results.slice(0, options.maxResults));
              }
            }
          } catch (error) {
            // ignore
          }
        }),
      ).then(() => {
        resolve(results);
      });
    });
  }
}
