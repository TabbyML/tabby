import path from "path";

import { debounce } from "lodash";
import { LRUCache } from "lru-cache";
import * as vscode from "vscode";
import { URI } from "vscode-uri";

import { getGraphContextFromRange as defaultGetGraphContextFromRange } from "../../../../graph/lsp/graph";
import { ContextRetriever, ContextRetrieverOptions, ContextSnippet, SymbolContextSnippet } from "../../../types";
import { CustomAbortController, CustomAbortSignal } from "../../utils";
import { HoverContext } from "../../../../codebase-context/messages";
import { dedupeWith } from "../../../../utils";

export class LspLightRetriever implements ContextRetriever {
  public identifier = "lsp-light";
  private disposables: vscode.Disposable[] = [];
  private cache: GraphCache = new GraphCache();

  private lastRequestKey: string | null = null;
  private abortLastRequest: () => void = () => {};

  constructor(
    // All arguments are optional, because they are only used in tests.
    private window: Pick<typeof vscode.window, "onDidChangeTextEditorSelection"> = vscode.window,
    private workspace: Pick<typeof vscode.workspace, "onDidChangeTextDocument"> = vscode.workspace,
    private getGraphContextFromRange = defaultGetGraphContextFromRange,
  ) {
    this.onDidChangeTextEditorSelection = debounce(this.onDidChangeTextEditorSelection.bind(this), 100);
    this.disposables.push(
      this.window.onDidChangeTextEditorSelection(this.onDidChangeTextEditorSelection.bind(this)),
      this.workspace.onDidChangeTextDocument(this.onDidChangeTextDocument.bind(this)),
    );
  }

  public async retrieve({
    document,
    position,
    hints: { maxChars },
  }: {
    document: ContextRetrieverOptions["document"];
    position: ContextRetrieverOptions["position"];
    hints: {
      maxChars: ContextRetrieverOptions["hints"]["maxChars"];
    };
  }): Promise<ContextSnippet[]> {
    const key = `${document.uri.toString()}█${position.line}█${document.lineAt(position.line).text}`;
    if (this.lastRequestKey !== key) {
      this.abortLastRequest();
    }

    const abortController = new CustomAbortController();

    this.lastRequestKey = key;
    this.abortLastRequest = () => abortController.abort();

    const prevLine = Math.max(position.line - 1, 0);
    const currentLine = position.line;

    const [prevLineContext, currentLineContext] = await Promise.all([
      this.getLspContextForLine({
        document,
        line: prevLine,
        abortSignal: abortController.signal,
      }),
      this.getLspContextForLine({
        document,
        line: currentLine,
        abortSignal: abortController.signal,
      }),
    ]);

    const sectionGraphContext = [...prevLineContext, ...currentLineContext];

    if (maxChars === 0) {
      // This is likely just a preloading request, so we don't need to prepare the actual
      // context
      return [];
    }

    return hoverContextsToSnippets(sectionGraphContext);
  }

  public isSupportedForLanguageId(languageId: string): boolean {
    return supportedLanguageId(languageId);
  }

  private getLspContextForLine({
    document,
    line,
    abortSignal,
  }: {
    document: vscode.TextDocument;
    line: number;
    abortSignal: CustomAbortSignal;
  }): Promise<HoverContext[]> {
    const request = {
      document,
      line,
    };

    const res = this.cache.get(request);
    if (res) {
      return res;
    }

    const range = document.lineAt(line).range;

    let finished = false;

    const promise = this.getGraphContextFromRange(document, range, abortSignal).then((response) => {
      finished = true;
      return response;
    });

    // Remove the aborted promise from the cache
    abortSignal.addEventListener("abort", () => {
      if (!finished) {
        this.cache.delete(request);
      }
    });

    this.cache.set(request, promise);

    return promise;
  }

  public dispose(): void {
    this.abortLastRequest();
    for (const disposable of this.disposables) {
      disposable.dispose();
    }
  }

  /**
   * When the cursor is moving into a new line, we want to fetch the context for the new line.
   */
  private onDidChangeTextEditorSelection(event: vscode.TextEditorSelectionChangeEvent): void {
    if (!supportedLanguageId(event.textEditor.document.languageId)) {
      return;
    }

    // Start a preloading requests as identifier by setting the maxChars to 0
    void this.retrieve({
      document: event.textEditor.document,
      position: event.selections[0].active,
      hints: { maxChars: 0 },
    });
  }

  /**
   * Whenever there are changes to a document, all cached contexts for other documents must be
   * evicted
   */
  private onDidChangeTextDocument(event: vscode.TextDocumentChangeEvent): void {
    this.cache.evictForOtherDocuments(event.document.uri);
  }
}

interface GraphCacheParams {
  document: vscode.TextDocument;
  line: number;
}
const MAX_CACHED_DOCUMENTS = 10;
const MAX_CACHED_LINES = 100;
class GraphCache {
  // This is a nested cache. The first level is the file uri, the second level is the line inside
  // the file.
  private cache = new LRUCache<string, LRUCache<string, Promise<HoverContext[]>>>({ max: MAX_CACHED_DOCUMENTS });

  private toCacheKeys(key: GraphCacheParams): [string, string] {
    return [key.document.uri.toString(), `${key.line}█${key.document.lineAt(key.line).text}`];
  }

  public get(key: GraphCacheParams): Promise<HoverContext[]> | undefined {
    const [docKey, lineKey] = this.toCacheKeys(key);

    const docCache = this.cache.get(docKey);
    if (!docCache) {
      return undefined;
    }

    return docCache.get(lineKey);
  }

  public set(key: GraphCacheParams, entry: Promise<HoverContext[]>): void {
    const [docKey, lineKey] = this.toCacheKeys(key);

    let docCache = this.cache.get(docKey);
    if (!docCache) {
      docCache = new LRUCache<string, Promise<HoverContext[]>>({
        max: MAX_CACHED_LINES,
      });
      this.cache.set(docKey, docCache);
    }
    docCache.set(lineKey, entry);
  }

  public delete(key: GraphCacheParams): void {
    const [docKey, lineKey] = this.toCacheKeys(key);

    const docCache = this.cache.get(docKey);
    if (!docCache) {
      return undefined;
    }
    docCache.delete(lineKey);
  }

  public evictForOtherDocuments(uri: vscode.Uri): void {
    const keysToDelete: string[] = [];
    this.cache.forEach((_, otherUri) => {
      if (otherUri === uri.toString()) {
        return;
      }
      keysToDelete.push(otherUri);
    });
    for (const key of keysToDelete) {
      this.cache.delete(key);
    }
  }
}

function hoverContextsToSnippets(contexts: HoverContext[]): SymbolContextSnippet[] {
  return dedupeWith(contexts.map(hoverContextToSnippets), (context) =>
    [context.symbol, context.fileName, context.content].join("\n"),
  );
}

function hoverContextToSnippets(context: HoverContext): SymbolContextSnippet {
  return {
    fileName: path.normalize(vscode.workspace.asRelativePath(URI.parse(context.uri).fsPath)),
    symbol: context.symbolName,
    content: context.content.join("\n").trim(),
  };
}
export function supportedLanguageId(languageId: string): boolean {
  switch (languageId) {
    case "python":
    case "go":
    case "javascript":
    case "javascriptreact":
    case "typescript":
    case "typescriptreact":
      return true;
    default:
      return false;
  }
}
