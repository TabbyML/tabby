import type { Range, ServerCapabilities } from "vscode-languageserver";
import type { TextDocument, TextDocumentContentChangeEvent } from "vscode-languageserver-textdocument";
import type { TextDocuments } from "../lsp/textDocuments";
import type { Feature } from "../feature";
import type { Configurations } from "../config";
import type { ConfigData } from "../config/type";
import type { DocumentRange } from "./engine";
import { CodeSearchEngine } from "./engine";
import deepEqual from "deep-equal";
import { getLogger } from "../logger";
import { unionRange, rangeInDocument } from "../utils/range";

function pickConfig(configData: ConfigData) {
  return configData.completion.prompt.collectSnippetsFromRecentChangedFiles.indexing;
}

function getLanguageFilter(languageId: string): string[] {
  const tsx = ["javascript", "javascriptreact", "typescript", "typescriptreact"];
  if (tsx.includes(languageId)) {
    return tsx;
  } else {
    return [languageId];
  }
}

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

    const engine = new CodeSearchEngine(config);
    this.codeSearchEngine = engine;

    if (this.indexingWorker) {
      clearInterval(this.indexingWorker);
    }
    this.indexingWorker = setInterval(async () => {
      let documentRange: DocumentRange | undefined = undefined;
      while ((documentRange = this.pendingDocumentRanges.shift()) != undefined) {
        this.logger.trace("Consuming indexing task.");
        await engine.index(documentRange);
      }
    }, config.checkingChangesInterval);
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
        line: Math.max(0, mergedEditedRange.start.line - config.prefixLines),
        character: 0,
      },
      end: {
        line: Math.min(document.lineCount, mergedEditedRange.end.line + config.suffixLines + 1),
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
        this.logger.trace("Created indexing task:", { documentRange });
      }, config.changesDebouncingInterval),
    });
  }

  async collectRelevantSnippets(
    query: string,
    currentDocument: TextDocument,
    limit?: number,
  ): Promise<{ filepath: string; offset: number; text: string; score: number }[] | undefined> {
    const engine = this.codeSearchEngine;
    if (!engine) {
      return undefined;
    }
    const indexed = engine.getIndexed();
    // Exclude current document from search
    const filepaths = indexed
      .map((documentRange) => documentRange.document.uri.toString())
      .filter((filepath) => filepath !== currentDocument.uri.toString());
    if (filepaths.length < 1) {
      return [];
    }
    const options = {
      filepathsFilter: filepaths,
      languagesFilter: getLanguageFilter(currentDocument.languageId),
      limit,
    };
    this.logger.trace("Search in recently changed files", { query, options });
    const result = await engine.search(query, options);
    this.logger.trace("Search result", { result });
    return result.map((hit) => {
      return {
        filepath: hit.snippet.filepath,
        offset: hit.snippet.offset,
        text: hit.snippet.fullText,
        score: hit.score,
      };
    });
  }
}
