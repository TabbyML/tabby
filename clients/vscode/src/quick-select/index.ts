import { window } from "vscode";
import type { QuickPick, QuickPickItem } from "vscode";
import { Client } from "../lsp/Client";
import { Config } from "../Config";

export class QuickChatController {
  private root = process.cwd();
  private quickPick: QuickPick<QuickPickItem> = window.createQuickPick<QuickPickItem>();

  constructor(
    private readonly client: Client,
    private readonly config: Config,
  ) {
    this.quickPick.placeholder = "Select file to chat";
  }

  start(dir?: string) {
    // Todo
    const path = dir || this.root;
  }

  stop() {
    // Todo
  }

  private showFileSelect() {}

  private fixPath(path: string): string {
    return ''
  }
}
