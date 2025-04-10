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
  TabInputText,
  WorkspaceFolder,
  CancellationToken,
  CancellationTokenSource,
  workspace,
  window,
} from "vscode";
import path from "path";
import { wrapCancelableFunction } from "./cancelableFunction";
import { getLogger } from "./logger";

const logger = getLogger("FindFiles");

// Map from workspace folder to gitignore patterns
const gitIgnorePatternsMap = new Map<string, string[]>();

function gitIgnoreItemToExcludePatterns(item: string, prefix?: string | undefined): string[] {
  let pattern = item.trim();
  if (pattern.startsWith("#") || pattern.startsWith("!") || pattern.length === 0) {
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

function joinPatterns(patterns: string[], maxLength = 32000 /* < 2 ^ 15 */): string {
  let result = "";
  for (const pattern of patterns) {
    if (result.length + pattern.length + 3 >= maxLength) {
      continue;
    }
    if (result.length > 0) {
      result += ",";
    }
    result += pattern;
  }
  return `{${result}}`;
}

function addUniqueItem(arr: string[], item: string) {
  if (!arr.includes(item)) {
    arr.push(item);
  }
}

async function updateGitIgnorePatterns(workspaceFolder: WorkspaceFolder, token?: CancellationToken | undefined) {
  const patterns: string[] = [];
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
        gitIgnoreItemToExcludePatterns(line).forEach((pattern) => addUniqueItem(patterns, pattern));
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
    ignoreFiles = (
      await workspace.findFiles(
        new RelativePattern(workspaceFolder, "**/.gitignore"),
        joinPatterns(patterns),
        undefined,
        token,
      )
    ).sort((a, b) => {
      const aDepth = a.path.split(path.sep).length;
      const bDepth = b.path.split(path.sep).length;
      if (aDepth != bDepth) {
        return aDepth - bDepth;
      }
      return a.path.localeCompare(b.path);
    });
  } catch (error) {
    // ignore
  }

  for (const ignoreFile of ignoreFiles) {
    if (token?.isCancellationRequested) {
      return;
    }
    const prefix = path.relative(workspaceFolder.uri.path, path.dirname(ignoreFile.path));
    try {
      const content = new TextDecoder().decode(await workspace.fs.readFile(ignoreFile));
      content.split(/\r?\n/).forEach((line) => {
        gitIgnoreItemToExcludePatterns(line, prefix).forEach((pattern) => addUniqueItem(patterns, pattern));
      });
    } catch (error) {
      // ignore
    }
  }
  // Update map
  logger.debug(
    `Completed building git ignore patterns for workspace folder: ${workspaceFolder.uri.toString()}, git ignore patterns: ${JSON.stringify(patterns)}`,
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
  const combinedExcludes: string[] = [];
  if (options?.excludes) {
    for (const exclude of options.excludes) {
      addUniqueItem(combinedExcludes, exclude);
    }
  }
  if (!options?.noUserSettings) {
    const searchExclude: Record<string, boolean> =
      (await workspace.getConfiguration("search", null).get("exclude")) ?? {};
    const filesExclude: Record<string, boolean> =
      (await workspace.getConfiguration("files", null).get("exclude")) ?? {};
    for (const pattern in { ...searchExclude, ...filesExclude }) {
      if (filesExclude[pattern]) {
        addUniqueItem(combinedExcludes, pattern);
      }
    }
  }
  if (options?.noIgnoreFiles) {
    const excludesPattern = joinPatterns(combinedExcludes);
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
          const excludesPattern = joinPatterns([
            ...combinedExcludes,
            ...(gitIgnorePatternsMap.get(workspaceFolder.uri.toString()) ?? []),
          ]);
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

export function sortFiles(files: Uri[], query: string): Uri[] {
  const matchString = query.toLowerCase().split("*").filter(Boolean)[0];
  if (!matchString) {
    return files.toSorted((uriA, uriB) => {
      const basenameA = path.basename(uriA.fsPath).toLowerCase();
      const basenameB = path.basename(uriB.fsPath).toLowerCase();
      return basenameA.localeCompare(basenameB);
    });
  }

  const getScore = (basename: string) => {
    if (basename == matchString) {
      return 4;
    }
    if (basename.split(".").includes(matchString)) {
      return 3;
    }
    if (basename.startsWith(matchString)) {
      return 2;
    }
    if (basename.includes(matchString)) {
      return 1;
    }
    return 0;
  };
  return files.toSorted((uriA, uriB) => {
    const basenameA = path.basename(uriA.fsPath).toLowerCase();
    const basenameB = path.basename(uriB.fsPath).toLowerCase();
    const scoreA = getScore(basenameA);
    const scoreB = getScore(basenameB);
    if (scoreA > scoreB) {
      return -1;
    }
    if (scoreA < scoreB) {
      return 1;
    }
    if (basenameA == basenameB) {
      const dirnameA = path.dirname(uriA.fsPath).toLowerCase();
      const dirnameB = path.dirname(uriB.fsPath).toLowerCase();
      return dirnameA.localeCompare(dirnameB);
    }
    return basenameA.localeCompare(basenameB);
  });
}

export function buildGlobPattern(query: string): GlobPattern {
  const caseInsensitivePattern = query
    .split("")
    .map((char) => {
      if (char.toLowerCase() !== char.toUpperCase()) {
        return `{${char.toLowerCase()},${char.toUpperCase()}}`;
      }
      // escape special glob characters: ? [ ] { } ( ) ! @
      return char.replace(/[?[\]{}()!@]/g, "\\$&");
    })
    .join("");

  return `**/*${caseInsensitivePattern}{*,*/*}`;
}

// `listFiles` will check the opened editors first, then use `findFiles` to search for files until the limit.
export async function listFiles(
  query: string,
  limit?: number | undefined,
  token?: CancellationToken | undefined,
): Promise<
  {
    uri: Uri;
    isOpenedInEditor: boolean;
  }[]
> {
  const maxResults = limit ?? 30;
  const queryString = query.trim().toLowerCase();

  const allEditorUris = window.tabGroups.all
    .flatMap((group) => group.tabs)
    .filter((tab) => tab.input && (tab.input as TabInputText).uri)
    .map((tab) => (tab.input as TabInputText).uri);

  const editorUris = sortFiles(
    allEditorUris
      // deduplicate
      .filter((uri, idx) => allEditorUris.findIndex((item) => item.fsPath === uri.fsPath) === idx)
      // filter by search query
      .filter((uri) => uri.fsPath.toLowerCase().includes(queryString)),
    queryString,
  )
    // move the active editor to the top
    .sort((uriA, uriB) => {
      const activeEditorUri = window.activeTextEditor?.document.uri;
      if (activeEditorUri) {
        if (uriA.fsPath === activeEditorUri.fsPath) return -1;
        if (uriB.fsPath === activeEditorUri.fsPath) return 1;
      }
      return 0;
    });

  const result = editorUris.map((uri) => {
    return {
      uri,
      isOpenedInEditor: true,
    };
  });
  if (result.length >= maxResults) {
    return result.slice(0, maxResults);
  }

  const globPattern = buildGlobPattern(queryString);
  try {
    const foundFiles = await findFiles(globPattern, {
      maxResults: maxResults - editorUris.length,
      excludes: editorUris.map((uri) => uri.fsPath),
      token,
    });
    const searchResult = sortFiles(
      foundFiles.filter(
        (uri, idx) =>
          foundFiles.findIndex((item) => item.fsPath === uri.fsPath) === idx &&
          !editorUris.some((exisingUri) => exisingUri.fsPath === uri.fsPath),
      ),
      queryString,
    );

    logger.debug(`Found ${searchResult.length} files matching pattern "${globPattern}"`);
    result.push(
      ...searchResult.map((uri) => {
        return {
          uri,
          isOpenedInEditor: false,
        };
      }),
    );
  } catch (error) {
    logger.debug("Failed to find files:", error);
  }

  return result;
}
