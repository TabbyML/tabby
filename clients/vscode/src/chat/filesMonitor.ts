import { workspace, ExtensionContext, FileSystemWatcher, RelativePattern, Uri } from "vscode";
import EventEmitter from "events";
import path from "path";

/**
 * WorkspaceFile is a file in the workspace
 */
export interface WorkspaceFile {
  filename: string;
  fsPath: string;
}
/**
 * Keep Tracking all files in workspace with cache set
 */
export class FilesMonitor extends EventEmitter {
  private fileCache: Set<string>;
  private watcher: FileSystemWatcher | undefined;

  constructor(private readonly context: ExtensionContext) {
    super();
    this.fileCache = new Set();
    this.initialize();
  }

  private initialize(): void {
    // Initial file scan
    this.scanWorkspace();

    // Setup file watchers
    const workspaceFolders = workspace.workspaceFolders;
    if (workspaceFolders && workspaceFolders[0]) {
      this.watcher = workspace.createFileSystemWatcher(new RelativePattern(workspaceFolders[0], "**/*"));

      this.watcher.onDidCreate((uri) => this.handleFileCreated(uri));
      this.watcher.onDidDelete((uri) => this.handleFileDeleted(uri));
      this.watcher.onDidChange((uri) => this.handleFileChanged(uri));

      this.context.subscriptions.push(this.watcher);
    }
  }

  private async scanWorkspace() {
    const workspaceFolders = workspace.workspaceFolders;
    if (!workspaceFolders) return;

    for (const folder of workspaceFolders) {
      const files = await workspace.findFiles(new RelativePattern(folder, "**/*"), "**/node_modules/**");
      files.forEach((uri) => {
        this.fileCache.add(uri.fsPath);
      });
    }
  }

  private handleFileCreated(uri: Uri) {
    this.fileCache.add(uri.fsPath);
  }

  private handleFileDeleted(uri: Uri) {
    this.fileCache.delete(uri.fsPath);
  }

  private handleFileChanged(uri: Uri) {
    if (!this.fileCache.has(uri.fsPath)) {
      this.fileCache.add(uri.fsPath);
    }
  }

  /**
   * Get all files in the workspace
   * @returns WorkspaceFile is a file struct represent filename and fsPath in current workspace
   */
  public getWorkspaceFiles(): WorkspaceFile[] {
    return Array.from(this.fileCache).map((fsPath) => {
      const relativePath = workspace.asRelativePath(fsPath);
      const parsed = path.parse(relativePath);
      return {
        filename: parsed.base,
        fsPath: fsPath,
      } as WorkspaceFile;
    });
  }

  public dispose(): void {
    this.fileCache.clear();
    this.watcher?.dispose();
  }
}
