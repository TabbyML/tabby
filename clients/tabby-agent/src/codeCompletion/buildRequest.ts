import type { Range } from "vscode-languageserver";
import type { components as TabbyApiComponents } from "tabby-openapi/compatible";
import { ConfigData } from "../config/type";
import { CompletionContext, CompletionExtraContexts } from "./contexts";
import path from "path";
import { intersectionRange } from "../utils/range";
import { cropTextToMaxChars } from "../utils/string";

export function buildRequest(params: {
  context: CompletionContext;
  extraContexts: CompletionExtraContexts;
  config: ConfigData["completion"]["prompt"];
}): TabbyApiComponents["schemas"]["Segments"] {
  const { context, extraContexts, config } = params;

  // prefix && suffix
  const prefix = context.prefixLines.slice(Math.max(context.prefixLines.length - config.maxPrefixLines, 0)).join("");
  const suffix = context.suffixLines.slice(0, config.maxSuffixLines).join("");

  // filepath && git_url
  let relativeRootUri: string | undefined = undefined;
  let gitUrl: string | undefined = undefined;
  if (extraContexts.git && extraContexts.git.repository) {
    // find remote url: origin > upstream > first
    const repo = extraContexts.git.repository;
    const remote =
      repo.remotes?.find((remote) => remote.name === "origin") ||
      repo.remotes?.find((remote) => remote.name === "upstream") ||
      repo.remotes?.[0];
    if (remote) {
      relativeRootUri = repo.root;
      gitUrl = remote.url;
    }
  }
  // if relativeFilepathRoot is not set by git context, use path relative to workspace
  if (!relativeRootUri && extraContexts.workspace) {
    relativeRootUri = extraContexts.workspace.uri;
  }
  const convertToRelativePath = (filepath: string): string => {
    if (relativeRootUri && filepath.startsWith(relativeRootUri)) {
      return path.relative(relativeRootUri, filepath);
    }
    return filepath;
  };

  const filepath = convertToRelativePath(context.document.uri);

  // snippets location for deduplication
  const snippetsLocations: { uri: string; range?: Range }[] = [];
  const isExists = (item: { uri: string; range?: Range }): boolean => {
    return !!snippetsLocations.find(
      (location) =>
        location.uri === item.uri && (!location.range || !item.range || intersectionRange(location.range, item.range)),
    );
  };

  // declarations
  const declarations: TabbyApiComponents["schemas"]["Declaration"][] = [];
  extraContexts.declarations?.forEach((item) => {
    if (declarations.length >= config.fillDeclarations.maxSnippets) {
      return;
    }
    declarations.push({
      filepath: convertToRelativePath(item.uri),
      body: cropTextToMaxChars(item.text, config.fillDeclarations.maxCharsPerSnippet),
    });
    snippetsLocations.push(item);
  });

  // snippets: recently changed code search
  const recentlyChangedCodeSearchResult: TabbyApiComponents["schemas"]["Snippet"][] = [];
  extraContexts.recentlyChangedCodeSearchResult?.forEach((item) => {
    if (
      recentlyChangedCodeSearchResult.length >= config.collectSnippetsFromRecentChangedFiles.maxSnippets ||
      isExists(item)
    ) {
      return;
    }
    recentlyChangedCodeSearchResult.push({
      filepath: convertToRelativePath(item.uri),
      body: item.text,
      score: item.score,
    });
    snippetsLocations.push(item);
  });

  // snippets: last viewed ranges
  const lastViewedSnippets: TabbyApiComponents["schemas"]["Snippet"][] = [];
  extraContexts.lastViewedSnippets?.forEach((item) => {
    if (lastViewedSnippets.length >= config.collectSnippetsFromRecentOpenedFiles.maxOpenedFiles || isExists(item)) {
      return;
    }
    lastViewedSnippets.push({
      filepath: convertToRelativePath(item.uri),
      body: cropTextToMaxChars(item.text, config.collectSnippetsFromRecentOpenedFiles.maxCharsPerOpenedFiles),
      score: 1,
    });
    snippetsLocations.push(item);
  });

  return {
    prefix,
    suffix,
    filepath,
    git_url: gitUrl,
    declarations: declarations.length > 0 ? declarations : undefined,
    relevant_snippets_from_changed_files:
      recentlyChangedCodeSearchResult.length > 0 ? recentlyChangedCodeSearchResult : undefined,
    relevant_snippets_from_recently_opened_files: lastViewedSnippets.length > 0 ? lastViewedSnippets : undefined,
  };
}
