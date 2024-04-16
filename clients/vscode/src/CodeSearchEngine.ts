import * as Engine from "@orama/orama";
import { Range, TextDocument } from "vscode";
import { extractSematicSymbols } from "./utils";

export type DocumentRange = {
  document: TextDocument;
  range: Range;
};

export type CodeSnippet = {
  // Which file does the snippet belongs to
  filepath: string;
  // (Not Indexed) The offset of the snippet in the file
  offset: number;
  // (Not Indexed) The full text of the snippet
  fullText: string;
  // The code language id of the snippet
  language: string;
  // The sematic symbols extracted from the snippet
  symbols: string;
};

export type ChunkingConfig = {
  // max count for chunks in memory
  maxChunks: number;
  // chars count per code chunk
  chunkSize: number;
  // lines count overlap between neighbor chunks
  overlapLines: number;
};

export type CodeSearchHit = {
  snippet: CodeSnippet;
  score: number;
};

export class CodeSearchEngine {
  constructor(private config: ChunkingConfig) {}

  private db: any | undefined = undefined;
  private indexedDocumentRanges: (DocumentRange & { indexIds: string[] })[] = [];

  private async init() {
    this.db = await Engine.create({
      schema: {
        filepath: "string",
        language: "string",
        symbols: "string",
      },
    });
  }

  private async count(): Promise<number> {
    if (!this.db) {
      return 0;
    }
    return await Engine.count(this.db);
  }

  private async insert(snippets: CodeSnippet[]): Promise<string[]> {
    if (!this.db) {
      await this.init();
    }
    return await Engine.insertMultiple(this.db, snippets);
  }

  private async remove(ids: string[]): Promise<number> {
    if (!this.db) {
      return 0;
    }
    return await Engine.removeMultiple(this.db, ids);
  }

  private async chunk(documentRange: DocumentRange): Promise<CodeSnippet[]> {
    const chunks: CodeSnippet[] = [];
    const { document, range } = documentRange;
    const documentUriString = document.uri.toString();
    let positionStart = range.start;
    while (positionStart.isBefore(range.end)) {
      const offset = document.offsetAt(positionStart);

      // move forward chunk size
      let positionEnd = document.positionAt(offset + this.config.chunkSize);
      if (positionEnd.isBefore(range.end)) {
        // If have not moved to the end, back to the last newline instead
        positionEnd = positionEnd.with({ character: 0 });
      }
      if (positionEnd.line <= positionStart.line + this.config.overlapLines) {
        // In case of forward chunk size does not moved enough lines for overlap, force move that much lines
        positionEnd = positionEnd.with(Math.min(document.lineCount, positionEnd.line + this.config.overlapLines + 1));
      }
      if (positionEnd.isAfter(range.end)) {
        // If have passed the end, back to the end
        positionEnd = range.end;
      }

      const text = document.getText(new Range(positionStart, positionEnd));
      if (text.trim().length > 0) {
        chunks.push({
          filepath: documentUriString,
          offset,
          fullText: text,
          language: document.languageId,
          symbols: extractSematicSymbols(text),
        });
      }
      if (
        chunks.length > this.config.maxChunks ||
        positionEnd.isAfterOrEqual(range.end) ||
        positionEnd.line > document.lineCount
      ) {
        break;
      }
      positionStart = positionEnd.with(Math.max(0, positionEnd.line - this.config.overlapLines));
    }
    return chunks;
  }

  getIndexed(): DocumentRange[] {
    return this.indexedDocumentRanges;
  }

  /**
   * Index the range of the document.
   *
   * When invoked multiple times with the same document but different ranges,
   * the ranges will be merged and re-chunked.
   *
   * If the indexed chunks in memory is too many, the oldest document will be removed.
   * The removal is by document, all chunks from the document will be removed.
   *
   * @param documentRange The document and specific range to index.
   */
  async index(documentRange: DocumentRange): Promise<void> {
    const { document, range } = documentRange;
    const documentUriString = document.uri.toString();
    let targetRange = range;
    const indexToUpdate = this.indexedDocumentRanges.findIndex(
      (item) => item.document.uri.toString() === documentUriString,
    );
    if (indexToUpdate >= 0) {
      // FIXME: union is not very good for merging two ranges have large distance between them
      targetRange = targetRange.union(this.indexedDocumentRanges[indexToUpdate]!.range);
    }
    const chunks = await this.chunk({ document, range: targetRange });
    if (indexToUpdate >= 0) {
      await this.remove(this.indexedDocumentRanges[indexToUpdate]!.indexIds);
      this.indexedDocumentRanges.splice(indexToUpdate);
    }
    const indexIds = await this.insert(chunks);
    this.indexedDocumentRanges.push({
      document,
      range: targetRange,
      indexIds,
    });

    // Check chunks count and evict if needed.
    while ((await this.count()) > this.config.maxChunks) {
      const toRemove = this.indexedDocumentRanges.shift();
      if (toRemove) {
        await this.remove(toRemove.indexIds);
      } else {
        break;
      }
    }
  }

  /**
   * Search relevant code snippets that has been indexed.
   * @param query contains words to search.
   * @param options
   * @param options.filepathsFilter only search in these filepaths.
   * @param options.languagesFilter only search in these languages.
   * @param options.limit max number of hits to return.
   * @returns A list of hit results, contains the snippet and score.
   */
  async search(
    query: string,
    options?: {
      filepathsFilter?: string[];
      languagesFilter?: string[];
      limit?: number;
    },
  ): Promise<CodeSearchHit[]> {
    if (!this.db) {
      return [];
    }
    const searchResult = await Engine.search(this.db, {
      term: query,
      properties: ["symbols"],
      where: {
        // FIXME: It seems this cannot exactly filtering using the filepaths
        // So we do a manual filtering later
        filepath: options?.filepathsFilter,
        language: options?.languagesFilter,
      },
      limit: options?.limit,
    });
    return (
      searchResult.hits
        // manual filtering
        .filter((hit) => {
          if (options?.filepathsFilter && !options?.filepathsFilter.includes(hit.document.filepath)) {
            return false;
          }
          if (options?.languagesFilter && !options?.languagesFilter.includes(hit.document.language)) {
            return false;
          }
          return true;
        })
        .map((hit) => {
          return {
            snippet: hit.document,
            // FIXME: Why there are many scores NaN?
            score: hit.score ?? 0,
          };
        })
    );
  }
}
