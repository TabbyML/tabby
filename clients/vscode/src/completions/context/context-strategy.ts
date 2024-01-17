import * as vscode from "vscode";

import { ContextRetriever } from "../types";

import { JaccardSimilarityRetriever } from "./retrievers/jaccard-similarity/jaccard-similarity-retriever";
import { LspLightRetriever } from "./retrievers/lsp-light/lsp-light-retriever";
import { SectionHistoryRetriever } from "./retrievers/section-history/section-history-retriever";

export type ContextStrategy =
  | "bfg"
  | "bfg-mixed"
  | "lsp-light"
  | "bfg"
  | "jaccard-similarity"
  | "bfg-mixed"
  | "local-mixed"
  | "section-history"
  | "none";

export interface ContextStrategyFactory extends vscode.Disposable {
  getStrategy(document: vscode.TextDocument): {
    name: ContextStrategy;
    retrievers: ContextRetriever[];
  };
}

export class DefaultContextStrategyFactory implements ContextStrategyFactory {
  private disposables: vscode.Disposable[] = [];

  private localRetriever: ContextRetriever | undefined;
  private graphRetriever: ContextRetriever | undefined;

  constructor(
    private contextStrategy: ContextStrategy,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    _context: vscode.ExtensionContext,
  ) {
    switch (contextStrategy) {
      case "none":
        break;
      case "bfg-mixed":
      case "bfg":
        throw new Error("not implemented");

        break;
      case "lsp-light":
        this.localRetriever = new JaccardSimilarityRetriever();
        this.graphRetriever = new LspLightRetriever();
        this.disposables.push(this.localRetriever, this.graphRetriever);
        break;
      case "jaccard-similarity":
        this.localRetriever = new JaccardSimilarityRetriever();
        this.disposables.push(this.localRetriever);
        break;
      case "section-history":
        this.localRetriever = SectionHistoryRetriever.createInstance();
        this.disposables.push(this.localRetriever);
        break;
      case "local-mixed":
        this.localRetriever = new JaccardSimilarityRetriever();
        // Filling the graphRetriever field with another local retriever but that's alright
        // we simply mix them later anyways.
        this.graphRetriever = SectionHistoryRetriever.createInstance();
        this.disposables.push(this.localRetriever, this.graphRetriever);
    }
  }

  public getStrategy(document: vscode.TextDocument): {
    name: ContextStrategy;
    retrievers: ContextRetriever[];
  } {
    const retrievers: ContextRetriever[] = [];

    switch (this.contextStrategy) {
      case "none": {
        break;
      }

      // The lsp-light strategy mixes local and graph based retrievers
      case "lsp-light": {
        if (this.graphRetriever && this.graphRetriever.isSupportedForLanguageId(document.languageId)) {
          retrievers.push(this.graphRetriever);
        }
        if (this.localRetriever) {
          retrievers.push(this.localRetriever);
        }
        break;
      }

      // The bfg strategy exclusively uses bfg strategy when the language is supported
      case "bfg":
        if (this.graphRetriever && this.graphRetriever.isSupportedForLanguageId(document.languageId)) {
          retrievers.push(this.graphRetriever);
        } else if (this.localRetriever) {
          retrievers.push(this.localRetriever);
        }
        break;

      // The bfg mixed strategy mixes local and graph based retrievers
      case "bfg-mixed":
        if (this.graphRetriever && this.graphRetriever.isSupportedForLanguageId(document.languageId)) {
          retrievers.push(this.graphRetriever);
        }
        if (this.localRetriever) {
          retrievers.push(this.localRetriever);
        }
        break;

      // The local mixed strategy combines two local retrievers
      case "local-mixed":
        if (this.localRetriever) {
          retrievers.push(this.localRetriever);
        }
        if (this.graphRetriever) {
          retrievers.push(this.graphRetriever);
        }
        break;

      // The jaccard similarity strategy only uses the local retriever
      case "jaccard-similarity": {
        if (this.localRetriever) {
          retrievers.push(this.localRetriever);
        }
        break;
      }

      case "section-history": {
        if (this.localRetriever) {
          retrievers.push(this.localRetriever);
        }
        break;
      }
    }
    return { name: this.contextStrategy, retrievers };
  }

  public dispose(): void {
    this.disposables.forEach((disposable) => disposable.dispose());
  }
}
