// We want to use the findFiles2 API, but it is still in the proposed state.
// Therefore, we implement a findFiles function here that can use the following rules in the exclude pattern:
// 1. .gitignore files
// 2. Settings from `files.exclude` and `search.exclude`
//
// See https://github.com/microsoft/vscode/blob/main/src/vscode-dts/vscode.proposed.findFiles2.d.ts

import {
  ExtensionContext,
  GlobPattern,
  RelativePattern,
  Uri,
  WorkspaceFolder,
  CancellationToken,
  CancellationTokenSource,
  workspace,
} from "vscode";
import path from "path";
import { wrapCancelableFunction } from "./cancelableFunction";
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

async function updateGitIgnorePatterns(workspaceFolder: WorkspaceFolder, token?: CancellationToken | undefined) {
  const patterns = new Set<string>();
  logger.debug(`Building gitignore patterns for workspace folder: ${workspaceFolder.uri.toString()}`);

  // Read parent gitignore files
  let current = workspaceFolder.uri;
  let parent = current.with({ path: path.dirname(current.path) });
  while (parent.path !== current.path) {
    if (token?.isCancellationRequested) {
      return;
    }

    const gitignore = parent.with({ path: path.join(parent.path, ".gitignore") });
    try {
      const content = new TextDecoder().decode(await workspace.fs.readFile(gitignore));
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

  if (token?.isCancellationRequested) {
    return;
  }
  // Read subdirectories gitignore files
  let ignoreFiles: Uri[] = [];
  try {
    ignoreFiles = await workspace.findFiles(
      new RelativePattern(workspaceFolder, "**/.gitignore"),
      undefined,
      undefined,
      token,
    );
  } catch (error) {
    // ignore
  }

  await Promise.all(
    ignoreFiles.map(async (ignoreFile) => {
      if (token?.isCancellationRequested) {
        return;
      }
      const prefix = path.relative(workspaceFolder.uri.path, path.dirname(ignoreFile.path));
      try {
        const content = new TextDecoder().decode(await workspace.fs.readFile(ignoreFile));
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

const updateGitIgnorePatternsMap = wrapCancelableFunction(async (token?: CancellationToken) => {
  await Promise.all(
    workspace.workspaceFolders?.map(async (workspaceFolder) => {
      await updateGitIgnorePatterns(workspaceFolder, token);
    }) ?? [],
  );
});

export async function init(context: ExtensionContext) {
  context.subscriptions.push(
    workspace.onDidChangeWorkspaceFolders(async () => {
      await updateGitIgnorePatternsMap();
    }),
  );

  context.subscriptions.push(
    workspace.onDidChangeTextDocument(async (event) => {
      const uri = event.document.uri;
      if (path.basename(uri.fsPath) === ".gitignore") {
        await updateGitIgnorePatternsMap();
      }
    }),
  );

  await updateGitIgnorePatternsMap();
}

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
    return new Promise((resolve, reject) => {
      if (options?.token?.isCancellationRequested) {
        reject(new Error("Operation canceled."));
        return;
      }

      const cancellationTokenSource = new CancellationTokenSource();
      if (options?.token) {
        options?.token.onCancellationRequested(() => {
          cancellationTokenSource.cancel();
          reject(new Error("Operation canceled."));
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
          const sortedExcludes = [...allExcludes].sort((a, b) => a.length - b.length).slice(0, 1000); // Limit to 1000 patterns
          const excludesPattern = `{${sortedExcludes.join(",")}}`;
          logger.debug(
            `Executing search: ${JSON.stringify({ includePattern, excludesPattern, maxResults: options?.maxResults })}`,
          );
          return await workspace.findFiles(
            includePattern,
            excludesPattern,
            options?.maxResults,
            cancellationTokenSource.token,
          );
        }) ?? [];

      const results: Uri[] = [];
      Promise.all(
        allSearches.map(async (search) => {
          try {
            const result = await search;
            if (result.length > 0) {
              results.push(...result);
              if (options?.maxResults && results.length >= options.maxResults) {
                cancellationTokenSource.cancel();
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

export function escapeGlobPattern(query: string): string {
  // escape special glob characters: * ? [ ] { } ( ) ! @
  return query.replace(/[*?[\]{}()!@]/g, "\\$&");
}

export function caseInsensitivePattern(query: string) {
  const caseInsensitivePattern = query
    .split("")
    .map((char) => {
      if (char.toLowerCase() !== char.toUpperCase()) {
        return `{${char.toLowerCase()},${char.toUpperCase()}}`;
      }
      return escapeGlobPattern(char);
    })
    .join("");

  return `**/${caseInsensitivePattern}*`;
}
