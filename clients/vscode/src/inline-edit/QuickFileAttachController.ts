import fuzzysort from "fuzzysort";
import fs from "fs/promises";
import throttle from "lodash/throttle";
import path from "path";
import {
  CancellationToken,
  QuickInputButtons,
  QuickPickItemKind,
  RelativePattern,
  TabInputText,
  Uri,
  ThemeIcon,
  window,
  workspace,
} from "vscode";
import type { QuickPick, QuickPickItem } from "vscode";
import type { Context } from "tabby-chat-panel";
import { WebviewHelper } from "../chat/WebviewHelper";
import type { GitProvider } from "../git/GitProvider";

const activeSeparator = {
  label: "Active",
  kind: QuickPickItemKind.Separator,
  value: "",
};

const folderSeparator = {
  label: "Folders",
  kind: QuickPickItemKind.Separator,
  value: "",
};

const fileSeparator = {
  label: "Files",
  kind: QuickPickItemKind.Separator,
  value: "",
};

export class QuickFileAttachController {
  private quickPick: QuickPick<FilePickItem> = window.createQuickPick<FilePickItem>();
  private current: string;
  private fileContext: Context | undefined = undefined;

  constructor(
    private readonly gitProvider: GitProvider,
    private selectCallback: (file: string) => void,
    private onBackTrigger: () => void,
  ) {
    this.current = this.root;

    this.quickPick.placeholder = "Select a file to attach";
    this.quickPick.canSelectMany = false;
    this.quickPick.matchOnDescription = false;
    this.quickPick.buttons = [QuickInputButtons.Back];

    this.quickPick.onDidAccept(this.onDidAccept, this);
    this.quickPick.onDidChangeValue(this.onDidChangeValue, this);
    this.quickPick.onDidTriggerButton(this.onDidTriggerButton, this);
  }

  get root() {
    const rootPath = workspace.workspaceFolders?.[0]?.uri.path || "";
    return rootPath;
  }

  set currentBase(p: string) {
    this.current = p;
  }

  get currentBase() {
    return this.current;
  }

  set selectedFileContext(ctx: Context) {
    this.fileContext = ctx;
  }

  get selectedFileContext(): Context | undefined {
    return this.fileContext;
  }

  public async start() {
    this.quickPick.items = await this.listFiles();
    this.quickPick.show();
  }

  public clear() {
    this.fileContext = undefined;
  }

  /**
   * This action is too expensive, so it will be only trigged in every 20 seconds;
   */
  public findWorkspaceFiles = throttle(() => this.getWorkspaceFiles(), 20 * 1000);

  private async onDidAccept() {
    const selected = this.quickPick.selectedItems[0]?.value;
    if (selected) {
      const stat = await fs.stat(selected);
      if (stat.isFile()) {
        const fileContext = await this.getSelectedFileContext(selected);
        this.fileContext = fileContext;
        this.quickPick.value = "";
        this.quickPick.hide();

        if (typeof this.selectCallback === "function") {
          this.selectCallback(path.basename(selected));
        }
      } else {
        this.currentBase = selected;
        this.quickPick.items = await this.listFiles(selected);
      }
    }
  }

  private async onDidChangeValue(value: string) {
    const results = await this.search(value);

    if (results.length) {
      this.quickPick.items = results.map((result) => ({
        label: result.target,
        value: result.obj.file.path,
        iconPath: new ThemeIcon("file"),
      }));
    } else {
      this.quickPick.items = await this.listFiles();
    }
  }

  private async onDidTriggerButton() {
    if (this.currentBase === this.root) {
      this.onBackTrigger();
      return;
    }

    const pos = this.currentBase.lastIndexOf("/");
    const root = pos > 0 ? this.currentBase.slice(0, pos) : this.root;
    const files = await this.listFiles(root === this.root ? undefined : root);
    this.currentBase = root;
    this.quickPick.items = files;
  }

  private async listFiles(p?: string) {
    const root = p || this.root;
    const ignorePatterns = this.getExcludedConfig().split(",");
    const currentDir = await fs.readdir(root);
    const files: FilePickItem[] = [];
    const dirs: FilePickItem[] = [];
    const activeFiles = this.getFilesFromOpenTabs();
    const result = [];

    if (activeFiles.length && typeof p === "undefined") {
      result.push(activeSeparator);
      result.push(
        ...(activeFiles.map((file) => ({
          value: file,
          label: this.handleQuickPickItemLabel(file),
          iconPath: new ThemeIcon("file"),
        })) as FilePickItem[]),
      );
    }

    for (const dir of currentDir) {
      const p = path.join(root, dir);

      if (ignorePatterns.includes(dir)) {
        continue;
      }

      const s = await fs.stat(p);
      if (s.isFile()) {
        files.push({
          value: p,
          label: dir,
          iconPath: new ThemeIcon("file"),
        });
      } else if (s.isDirectory()) {
        dirs.push({
          value: p,
          label: dir,
          iconPath: new ThemeIcon("file-directory"),
        });
      }
    }

    if (files.length) {
      result.push(fileSeparator);
      result.push(...files);
    }

    if (dirs.length) {
      result.push(folderSeparator);
      result.push(...dirs);
    }

    return result;
  }

  private getExcludedConfig() {
    // FIXME(@eryue0220): support custom exclude patterns
    const excluded = ".git,.svn,.hg,CVS,.DS_Store,Thumbs.db,node_modules,bower_components,*.code-search";
    return excluded;
  }

  private async getSelectedFileContext(path: string): Promise<Context> {
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
    return fileContext;
  }

  private async getWorkspaceFiles(cancellationToken?: CancellationToken) {
    return (
      await Promise.all(
        (workspace.workspaceFolders ?? [null]).map(async (workspaceFolder) =>
          workspace.findFiles(
            workspaceFolder ? new RelativePattern(workspaceFolder, "**") : "",
            this.getExcludedConfig(),
            undefined,
            cancellationToken,
          ),
        ),
      )
    ).flat();
  }

  private getFilesFromOpenTabs(): string[] {
    const tabGroups = window.tabGroups.all;
    const openTabs = tabGroups.flatMap((group) => group.tabs.map((tab) => tab.input)) as TabInputText[];

    return openTabs
      .map((tab) => {
        if (!tab.uri || tab.uri.scheme !== "file" || !workspace.getWorkspaceFolder(tab.uri)) {
          return undefined;
        }

        return tab.uri.path;
      })
      .filter((path): path is string => path !== undefined);
  }

  private handleQuickPickItemLabel(path: string) {
    return path.replace(`${this.root}/`, "");
  }

  private async search(query: string) {
    const files = await this.findWorkspaceFiles();
    const ranges = files.map((file) => ({ file, key: this.handleQuickPickItemLabel(file.path) }));
    const results = fuzzysort.go(query, ranges, { key: "key", limit: 20 });

    return results.map((item) => ({ ...item, score: item.score })).sort((a, b) => b.score - a.score);
  }
}

interface FilePickItem extends QuickPickItem {
  value: string;
}
