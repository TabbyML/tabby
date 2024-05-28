import { Range } from "vscode-languageserver";
import { TextDocument, TextDocumentContentChangeEvent } from "vscode-languageserver-textdocument";
import { CodeSearchEngine, ChunkingConfig, DocumentRange } from "./CodeSearchEngine";
import { getLogger } from "../logger";
import { unionRange, rangeInDocument } from "../utils/range";

interface TextChangeCroppingWindowConfig {
  prefixLines: number;
  suffixLines: number;
}

interface TextChangeListenerConfig {
  checkingChangesInterval: number;
  changesDebouncingInterval: number;
}

export class RecentlyChangedCodeSearch {
  private readonly logger = getLogger("CodeSearch");
  private codeSearchEngine: CodeSearchEngine;

  private pendingDocumentRanges: DocumentRange[] = [];
  /* @ts-expect-error noUnusedLocals */
  private indexingWorker: ReturnType<typeof setInterval>;

  private didChangeEventDebouncingCache = new Map<
    string,
    { documentRange: DocumentRange; timer: ReturnType<typeof setTimeout> }
  >();

  constructor(private config: TextChangeCroppingWindowConfig & TextChangeListenerConfig & ChunkingConfig) {
    this.codeSearchEngine = new CodeSearchEngine(config);
    this.indexingWorker = setInterval(async () => {
      let documentRange: DocumentRange | undefined = undefined;
      while ((documentRange = this.pendingDocumentRanges.shift()) != undefined) {
        this.logger.trace("Consuming indexing task.");
        await this.codeSearchEngine.index(documentRange);
      }
    }, config.checkingChangesInterval);
    this.logger.info("Created code search engine for recently changed files.");
    this.logger.trace("Created with config.", { config });
  }

  handleDidChangeTextDocument(event: { document: TextDocument; contentChanges: TextDocumentContentChangeEvent[] }) {
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
        line: Math.max(0, mergedEditedRange.start.line - this.config.prefixLines),
        character: 0,
      },
      end: {
        line: Math.min(document.lineCount, mergedEditedRange.end.line + this.config.suffixLines + 1),
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
      }, this.config.changesDebouncingInterval),
    });
  }

  async collectRelevantSnippets(
    query: string,
    currentDocument: TextDocument,
    limit?: number,
  ): Promise<{ filepath: string; offset: number; text: string; score: number }[] | undefined> {
    const indexed = this.codeSearchEngine.getIndexed();
    // Exclude current document from search
    const filepaths = indexed
      .map((documentRange) => documentRange.document.uri.toString())
      .filter((filepath) => filepath !== currentDocument.uri.toString());
    if (filepaths.length < 1) {
      return [];
    }
    const options = {
      filepathsFilter: filepaths,
      languagesFilter: this.getLanguageFilter(currentDocument.languageId),
      limit,
    };
    this.logger.trace("Search in recently changed files", { query, options });
    const result = await this.codeSearchEngine.search(query, options);
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

  private getLanguageFilter(languageId: string): string[] {
    const tsx = ["javascript", "javascriptreact", "typescript", "typescriptreact"];
    if (tsx.includes(languageId)) {
      return tsx;
    } else {
      return [languageId];
    }
  }
}
