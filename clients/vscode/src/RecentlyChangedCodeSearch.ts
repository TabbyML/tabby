import { Range, TextDocument, TextDocumentChangeEvent } from "vscode";
import { getLogger } from "./logger";
import { CodeSearchEngine, ChunkingConfig, DocumentRange } from "./CodeSearchEngine";

type TextChangeCroppingWindowConfig = {
  prefixLines: number;
  suffixLines: number;
};

type TextChangeListenerConfig = {
  checkingChangesInterval: number;
  changesDebouncingInterval: number;
};

export class RecentlyChangedCodeSearch {
  private readonly logger = getLogger("CodeSearch");
  private codeSearchEngine: CodeSearchEngine;

  private pendingDocumentRanges: DocumentRange[] = [];
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

  handleDidChangeTextDocument(event: TextDocumentChangeEvent) {
    const { document, contentChanges } = event;
    if (contentChanges.length < 1) {
      return;
    }
    const documentUriString = document.uri.toString();
    let ranges = [];
    if (this.didChangeEventDebouncingCache.has(documentUriString)) {
      const cache = this.didChangeEventDebouncingCache.get(documentUriString)!;
      ranges.push(cache.documentRange.range);
      clearTimeout(cache.timer);
    }
    ranges = ranges.concat(
      contentChanges.map(
        (change) =>
          new Range(
            document.positionAt(change.rangeOffset),
            document.positionAt(change.rangeOffset + change.text.length),
          ),
      ),
    );
    const mergedEditedRange = ranges.reduce((a, b) => a.union(b));
    // Expand the range to cropping window
    const targetRange = document.validateRange(
      new Range(
        Math.max(0, mergedEditedRange.start.line - this.config.prefixLines),
        0,
        Math.min(document.lineCount, mergedEditedRange.end.line + this.config.suffixLines + 1),
        0,
      ),
    );
    const documentRange = { document, range: targetRange };
    // A debouncing to avoid indexing the same document multiple times in a short time
    this.didChangeEventDebouncingCache.set(documentUriString, {
      documentRange,
      timer: setTimeout(() => {
        this.pendingDocumentRanges.push(documentRange);
        this.didChangeEventDebouncingCache.delete(documentUriString);
        this.logger.trace("Created indexing task:", { path: documentUriString, range: targetRange });
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
