import fs from "fs/promises";
import path from "path";
import { QuickInputButtons, QuickPickItemKind, Uri, ThemeIcon, window } from "vscode";
import type { QuickPick, QuickPickItem } from "vscode";
import type { Context } from "tabby-chat-panel";
import type { ChatSideViewProvider } from "../chat/ChatSideViewProvider";
import { WebviewHelper } from "../chat/WebviewHelper";
import type { GitProvider } from "../git/GitProvider";

const folderSeparator = {
  label: "Folders",
  kind: QuickPickItemKind.Separator,
};

const fileSeparator = {
  label: "Files",
  kind: QuickPickItemKind.Separator,
};

export class QuickFileAttach {
  private base = process.cwd();
  private quickPick: QuickPick<QuickPickItem> = window.createQuickPick<QuickPickItem>();
  private current: string;

  constructor(
    private readonly chatViewProvider: ChatSideViewProvider,
    private readonly gitProvider: GitProvider,
  ) {
    this.current = this.base;

    this.quickPick.placeholder = "Select or search a file to add to the chat...";
    this.quickPick.canSelectMany = false;
    this.quickPick.matchOnDescription = true;
    this.quickPick.buttons = [QuickInputButtons.Back];

    this.quickPick.onDidAccept(this.onDidAccept, this);
    this.quickPick.onDidTriggerButton(this.onDidTriggerButton, this);
    this.quickPick.onDidHide(this.quickPick.dispose);
  }

  get root() {
    return this.base;
  }

  set currentBase(p: string) {
    this.current = p;
  }

  get currentBase() {
    return this.current;
  }

  public async start() {
    this.quickPick.items = await this.listFiles();
    this.quickPick.show();
  }

  private async readDir(d: string) {
    return await fs.readdir(d);
  }

  private async onDidAccept() {
    const selected = this.quickPick.selectedItems[0];
    const root = selected?.detail;
    if (root) {
      const s = await fs.stat(root);
      if (s.isFile()) {
        await this.addFileToChat(root);
        this.quickPick.hide();
      } else {
        this.currentBase = root;
        this.quickPick.items = await this.listFiles(root);
      }
    }
  }

  private async onDidTriggerButton() {
    if (this.currentBase === this.root) {
      return;
    }

    const prev = this.currentBase.split("/");
    const root = prev.slice(0, prev.length - 1).join("/");
    const files = await this.listFiles(root);
    this.quickPick.items = files;
  }

  private async listFiles(p?: string) {
    const root = p || this.root;
    const currentDir = await this.readDir(root);
    const files: QuickPickItem[] = [];
    const dirs: QuickPickItem[] = [];
    const result = [];

    for (const dir of currentDir) {
      const p = path.join(root, dir);
      const s = await fs.stat(p);
      if (s.isFile()) {
        files.push({
          detail: p,
          label: dir,
          iconPath: new ThemeIcon("file"),
        });
      } else if (s.isDirectory()) {
        dirs.push({
          detail: p,
          label: dir,
          iconPath: new ThemeIcon("file-directory"),
        });
      }
    }

    if (files.length) {
      result.unshift(fileSeparator);
      result.push(...files);
    }

    if (dirs.length) {
      result.push(folderSeparator);
      result.push(...dirs);
    }

    return result;
  }

  private async addFileToChat(path: string) {
    const uri = Uri.file(path);
    const content = await fs.readFile(path, "utf8");
    const lines = content.split("\n").length;
    const { filepath, git_url } = WebviewHelper.resolveFilePathAndGitUrl(uri, this.gitProvider);
    const fileContext: Context = {
      kind: "file",
      content,
      range: {
        start: 1,
        end: lines,
      },
      filepath,
      git_url,
    };

    this.chatViewProvider.addRelevantContext(fileContext);
  }
}
