import type { Range, ServerCapabilities } from "vscode-languageserver";
import type { TextDocument, TextDocumentContentChangeEvent } from "vscode-languageserver-textdocument";
import type { TextDocuments } from "../extensions/textDocuments";
import type { Feature } from "../feature";
import type { Configurations } from "../config";
import type { ConfigData } from "../config/type";
import type { DocumentRange } from "../utils/types";
import { CodeSearchEngine, CodeSearchResult } from "../codeSearch";
import deepEqual from "deep-equal";
import { unionRange, rangeInDocument } from "../utils/range";
import { getLogger } from "../logger";

export class RecentlyChangedCodeSearch implements Feature {
  private readonly logger = getLogger("RecentlyChangedCodeSearch");
  private codeSearchEngine: CodeSearchEngine | undefined = undefined;

  private indexingWorker: ReturnType<typeof setInterval> | undefined = undefined;

  private pendingDocumentRanges: DocumentRange[] = [];
  private didChangeEventDebouncingCache = new Map<
    string,
    { documentRange: DocumentRange; timer: ReturnType<typeof setTimeout> }
  >();

  constructor(
    private readonly configurations: Configurations,
    private readonly documents: TextDocuments<TextDocument>,
  ) {}

  initialize(): ServerCapabilities {
    this.setup();
    this.configurations.on("updated", (config: ConfigData, oldConfig: ConfigData) => {
      if (!deepEqual(pickConfig(config), pickConfig(oldConfig))) {
        this.shutdown();
        this.setup();
      }
    });

    this.documents.onDidChangeContent(async (params: unknown) => {
      if (!params || typeof params !== "object" || !("document" in params) || !("contentChanges" in params)) {
        return;
      }
      const event = params as { document: TextDocument; contentChanges: TextDocumentContentChangeEvent[] };
      this.handleDidChangeTextDocument(event);
    });

    return {};
  }

  shutdown() {
    if (this.indexingWorker) {
      clearInterval(this.indexingWorker);
    }
    this.codeSearchEngine = undefined;
  }

  private setup() {
    const config = pickConfig(this.configurations.getMergedConfig());
    if (!config.enabled) {
      this.logger.info("Recently changed code search is disabled.");
      return;
    }

    const engine = new CodeSearchEngine(config.indexing);
    this.codeSearchEngine = engine;

    this.indexingWorker = setInterval(async () => {
      let documentRange: DocumentRange | undefined = undefined;
      while ((documentRange = this.pendingDocumentRanges.shift()) != undefined) {
        this.logger.trace("Consuming indexing task.");
        await engine.index(documentRange);
      }
    }, config.indexing.checkingChangesInterval);
    this.logger.info("Created code search engine for recently changed files.");
    this.logger.trace("Created with config.", { config });
  }

  private handleDidChangeTextDocument(event: {
    document: TextDocument;
    contentChanges: TextDocumentContentChangeEvent[];
  }) {
    const config = pickConfig(this.configurations.getMergedConfig());

    const { document, contentChanges } = event;
    if (contentChanges.length < 1) {
      return;
    }
    let ranges = [];
    if (this.didChangeEventDebouncingCache.has(document.uri)) {
      const cache = this.didChangeEventDebouncingCache.get(document.uri);
      if (cache) {
        ranges.push(cache.documentRange.range);
        clearTimeout(cache.timer);
      }
    }
    ranges = ranges.concat(
      contentChanges
        .map((change) =>
          "range" in change
            ? {
                start: change.range.start,
                end: document.positionAt(document.offsetAt(change.range.start) + change.text.length),
              }
            : null,
        )
        .filter((range): range is Range => range !== null),
    );
    const mergedEditedRange = ranges.reduce((a, b) => unionRange(a, b));
    // Expand the range to cropping window
    const expand: Range = {
      start: {
        line: Math.max(0, mergedEditedRange.start.line - config.indexing.prefixLines),
        character: 0,
      },
      end: {
        line: Math.min(document.lineCount, mergedEditedRange.end.line + config.indexing.suffixLines + 1),
        character: 0,
      },
    };
    const targetRange = rangeInDocument(expand, document);
    if (targetRange === null) {
      return;
    }
    const documentRange = { document, range: targetRange };
    // A debouncing to avoid indexing the same document multiple times in a short time
    this.didChangeEventDebouncingCache.set(document.uri, {
      documentRange,
      timer: setTimeout(() => {
        this.pendingDocumentRanges.push(documentRange);
        this.didChangeEventDebouncingCache.delete(document.uri);
        this.logger.trace("Created indexing task:", {
          document: documentRange.document.uri,
          range: documentRange.range,
        });
      }, config.indexing.changesDebouncingInterval),
    });
  }

  async search(
    query: string,
    excludes: string[],
    language: string,
    limit?: number,
  ): Promise<CodeSearchResult | undefined> {
    const engine = this.codeSearchEngine;
    if (!engine) {
      return undefined;
    }
    const indexedDocumentRange = engine.getIndexedDocumentRange();

    const filepaths = indexedDocumentRange
      .map((documentRange) => documentRange.document.uri.toString())
      .filter((filepath) => !excludes.includes(filepath));
    if (filepaths.length < 1) {
      return [];
    }

    const options = {
      filepathsFilter: filepaths,
      languagesFilter: getLanguageFilter(language),
      limit,
    };
    this.logger.trace("Search in recently changed files", { query, options });
    const result = await engine.search(query, options);
    this.logger.trace("Search result", { result });
    return result;
  }
}

function pickConfig(configData: ConfigData) {
  return configData.completion.prompt.collectSnippetsFromRecentChangedFiles;
}

function getLanguageFilter(languageId: string): string[] {
  const tsx = ["javascript", "javascriptreact", "typescript", "typescriptreact"];
  if (tsx.includes(languageId)) {
    return tsx;
  } else {
    return [languageId];
  }
}
