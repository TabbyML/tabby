import path, { basename, dirname } from "path";

import fuzzysort from "fuzzysort";
import { throttle } from "lodash";
import * as vscode from "vscode";

import { getOpenTabsUris, getWorkspaceSymbols } from ".";
import {
  ContextFile,
  ContextFileSource,
  ContextFileType,
  SymbolKind,
} from "../../codebase-context/messages";

const findWorkspaceFiles = async (
  cancellationToken: vscode.CancellationToken
): Promise<vscode.Uri[]> => {
  // TODO(toolmantim): Add support for the search.exclude option, e.g.
  // Object.keys(vscode.workspace.getConfiguration().get('search.exclude',
  // {}))
  const fileExcludesPattern =
    "**/{*.env,.git,out/,dist/,bin/,snap,node_modules,__pycache__}**";
  // TODO(toolmantim): Check this performs with remote workspaces (do we need a UI spinner etc?)
  return vscode.workspace.findFiles(
    "",
    fileExcludesPattern,
    undefined,
    cancellationToken
  );
};

// This is expensive for large repos (e.g. Chromium), so we only do it max once
// every 10 seconds. It also handily supports a cancellation callback to use
// with the cancellation token to discard old requests.
const throttledFindFiles = throttle(findWorkspaceFiles, 10000);

/**
 * Searches all workspaces for files matching the given string. VS Code doesn't
 * provide an API for fuzzy file searching, only precise globs, so we recreate
 * it by getting a list of all files across all workspaces and using fuzzysort.
 */
export async function getFileContextFiles(
  query: string,
  maxResults: number,
  token: vscode.CancellationToken
): Promise<ContextFile[]> {
  if (!query.trim()) {
    return [];
  }
  token.onCancellationRequested(() => {
    throttledFindFiles.cancel();
  });

  const uris = await throttledFindFiles(token);

  if (!uris) {
    return [];
  }

  // On Windows, if the user has typed forward slashes, map them to backslashes before
  // running the search so they match the real paths.
  query = query.replaceAll(path.posix.sep, path.sep);

  // Add on the relative URIs for search, so we only search the visible part
  // of the path and not the full FS path.
  const urisWithRelative = uris.map((uri) => ({
    uri,
    relative: asRelativePath(uri),
  }));
  const results = fuzzysort.go(query, urisWithRelative, {
    key: "relative",
    limit: maxResults,
    // We add a threshold for performance as per fuzzysort’s
    // recommendations. Testing with sg/sg path strings, somewhere over 10k
    // threshold is where it seems to return results that make no sense. VS
    // Code’s own fuzzy finder seems to cap out much higher. To be safer and
    // to account for longer paths from even deeper source trees we use
    // 100k. We may want to revisit this number if we get reports of missing
    // file results from very large repos.
    threshold: -100000,
  });

  // fuzzysort can return results in different order for the same query if
  // they have the same score :( So we do this hacky post-limit sorting (first
  // by score, then by path) to ensure the order stays the same
  const sortedResults = [...results].sort((a, b) => {
    return (
      b.score - a.score ||
      new Intl.Collator(undefined, { numeric: true }).compare(
        a.obj.uri.fsPath,
        b.obj.uri.fsPath
      )
    );
  });

  // TODO(toolmantim): Add fuzzysort.highlight data to the result so we can show it in the UI

  return sortedResults.map((result) =>
    createContextFileFromUri(result.obj.uri)
  );
}

export async function getSymbolContextFiles(
  query: string,
  maxResults = 20
): Promise<ContextFile[]> {
  if (!query.trim()) {
    return [];
  }

  const queryResults = await getWorkspaceSymbols(query); // doesn't support cancellation tokens :(

  const relevantQueryResults = queryResults?.filter(
    (symbol) =>
      (symbol.kind === vscode.SymbolKind.Function ||
        symbol.kind === vscode.SymbolKind.Method ||
        symbol.kind === vscode.SymbolKind.Class ||
        symbol.kind === vscode.SymbolKind.Interface ||
        symbol.kind === vscode.SymbolKind.Enum ||
        symbol.kind === vscode.SymbolKind.Struct ||
        symbol.kind === vscode.SymbolKind.Constant ||
        // in TS an export const is considered a variable
        symbol.kind === vscode.SymbolKind.Variable) &&
      // TODO(toolmantim): Remove once https://github.com/microsoft/vscode/pull/192798 is in use (test: do a symbol search and check no symbols exist from node_modules)
      !symbol.location?.uri?.fsPath.includes("node_modules/")
  );

  const results = fuzzysort.go(query, relevantQueryResults, {
    key: "name",
    limit: maxResults,
  });

  // TODO(toolmantim): Add fuzzysort.highlight data to the result so we can show it in the UI

  const symbols = results.map((result) => result.obj);

  if (!symbols.length) {
    return [];
  }

  const matches = [];
  for (const symbol of symbols) {
    // TODO(toolmantim): Update the kinds to match above
    const kind: SymbolKind =
      symbol.kind === vscode.SymbolKind.Class ? "class" : "function";
    const source: ContextFileSource = "user";
    const contextFile: ContextFile = createContextFileFromUri(
      symbol.location.uri,
      source,
      "symbol",
      symbol.location.range,
      kind
    );
    contextFile.fileName = symbol.name;
    matches.push(contextFile);
  }

  return matches;
}

export function getOpenTabsContextFile(): ContextFile[] {
  // de-dupe by fspath in case if they have a file open in two tabs
  const fsPaths = new Set();
  return getOpenTabsUris()
    .filter((uri) => {
      if (!fsPaths.has(uri.fsPath)) {
        fsPaths.add(uri.fsPath);
        return true;
      }
      return false;
    })
    .map((uri) =>
      createContextFileFromUri(uri, "user", "file", undefined, undefined)
    );
}

function createContextFileFromUri(
  uri: vscode.Uri,
  source: ContextFileSource = "user",
  type: ContextFileType = "file",
  selectionRange?: vscode.Range,
  kind?: SymbolKind
): ContextFile {
  const range = selectionRange
    ? createContextFileRange(selectionRange)
    : selectionRange;
  return {
    fileName: vscode.workspace.asRelativePath(uri.fsPath),
    uri,
    path: createContextFilePath(uri),
    range,
    type,
    source,
    kind,
  };
}

function createContextFileRange(
  selectionRange: vscode.Range
): ContextFile["range"] {
  return {
    start: {
      line: selectionRange.start.line,
      character: selectionRange.start.character,
    },
    end: {
      line: selectionRange.end.line,
      character: selectionRange.end.character,
    },
  };
}

function createContextFilePath(uri: vscode.Uri): ContextFile["path"] {
  return {
    basename: basename(uri.fsPath),
    dirname: dirname(uri.fsPath),
    relative: asRelativePath(uri),
  };
}

/**
 * Returns a relative path using the correct slash direction for the current platform.
 */
function asRelativePath(uri: vscode.Uri): string {
  let relativePath = vscode.workspace.asRelativePath(uri.fsPath);
  // asRelativePath returns forward slashes on Windows but we want to
  // render a native path like VS Code does in the file quick-pick.
  relativePath = relativePath.replaceAll(path.posix.sep, path.sep);

  return relativePath;
}
