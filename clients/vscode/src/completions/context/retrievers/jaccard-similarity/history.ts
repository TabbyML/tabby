import * as vscode from "vscode";
import { baseLanguageId } from "../../utils";

export interface HistoryItem {
  document: Pick<vscode.TextDocument, "uri" | "languageId">;
}

export interface DocumentHistory {
  addItem(newItem: HistoryItem): void;
  lastN(n: number, languageId?: string, ignoreUris?: vscode.Uri[]): HistoryItem[];
}

export class VSCodeDocumentHistory implements DocumentHistory, vscode.Disposable {
  private window = 50;

  // tracks history in chronological order (latest at the end of the array)
  private history: HistoryItem[];

  private subscriptions: vscode.Disposable[] = [];

  constructor(
    register: () => vscode.Disposable | null = () =>
      vscode.window.onDidChangeActiveTextEditor((event) => {
        if (!event?.document.uri) {
          return;
        }
        this.addItem({
          document: event.document,
        });
      }),
  ) {
    this.history = [];
    if (register) {
      const disposable = register();
      if (disposable) {
        this.subscriptions.push(disposable);
      }
    }
  }

  public dispose(): void {
    vscode.Disposable.from(...this.subscriptions).dispose();
  }

  public addItem(newItem: HistoryItem): void {
    if (newItem.document.uri.scheme === "codegen") {
      return;
    }
    const foundIndex = this.history.findIndex(
      (item) => item.document.uri.toString() === newItem.document.uri.toString(),
    );
    if (foundIndex >= 0) {
      this.history = [...this.history.slice(0, foundIndex), ...this.history.slice(foundIndex + 1)];
    }
    this.history.push(newItem);
    if (this.history.length > this.window) {
      this.history.shift();
    }
  }

  /**
   * Returns the last n items of history in reverse chronological order (latest item at the front)
   */
  public lastN(n: number, languageId?: string, ignoreUris?: vscode.Uri[]): HistoryItem[] {
    const ret: HistoryItem[] = [];
    const ignoreSet = new Set(ignoreUris || []);
    for (let i = this.history.length - 1; i >= 0; i--) {
      const item = this.history[i];
      if (ret.length > n) {
        break;
      }
      if (ignoreSet.has(item.document.uri)) {
        continue;
      }
      if (languageId && baseLanguageId(languageId) !== baseLanguageId(item.document.languageId)) {
        continue;
      }
      ret.push(item);
    }
    return ret;
  }
}
