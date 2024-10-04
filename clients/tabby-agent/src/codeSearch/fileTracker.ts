import { Connection, Range } from "vscode-languageserver";
import { Feature } from "../feature";
import { OpenedFileParams, OpenedFileRequest, ServerCapabilities } from "../protocol";
import { getLogger } from "../logger";
import { Configurations } from "../config";

interface OpenedFile {
  uri: string;
  //order by range, the left most is the most recent one
  lastVisibleRange: Range[];
  invisible: boolean;
  isActive: boolean;
}

export class LRUList {
  private list: OpenedFile[] = [];

  constructor(private readonly maxSize: number) {}

  insert(file: OpenedFile): void {
    const existingIndex = this.list.findIndex((item) => item.uri === file.uri);
    if (existingIndex !== -1) {
      const existingFile = this.list[existingIndex];
      if (!existingFile) {
        return;
      }
      existingFile.invisible = file.invisible;
      this.list.splice(existingIndex, 1);
      this.list.unshift(existingFile);
    } else {
      this.list.unshift(file);
    }

    if (this.list.length > this.maxSize) {
      this.removeLast();
    }
  }

  removeLast(): OpenedFile | undefined {
    return this.list.pop();
  }

  get(filename: string): OpenedFile | undefined {
    const index = this.list.findIndex((item) => item.uri === filename);
    if (index === -1) return undefined;
    const item = this.list.splice(index, 1)[0];
    if (!item) return undefined;
    this.list.unshift(item);
    return item;
  }

  contains(filename: string): boolean {
    return this.list.some((item) => item.uri === filename);
  }

  update(filename: string, updates: Partial<OpenedFile>): boolean {
    const file = this.get(filename);
    if (!file) return false;
    if (updates.lastVisibleRange) {
      file.lastVisibleRange = updates.lastVisibleRange;
    }
    if (updates.invisible !== undefined) {
      file.invisible = updates.invisible;
    }
    return true;
  }

  getAll(): OpenedFile[] {
    return [...this.list];
  }

  remove(filename: string): OpenedFile | undefined {
    const index = this.list.findIndex((item) => item.uri === filename);
    if (index === -1) return undefined;
    return this.list.splice(index, 1)[0];
  }

  clear(): void {
    this.list = [];
  }

  size(): number {
    return this.list.length;
  }

  toString(): string {
    return JSON.stringify(this.list);
  }
}
export class FileTracker implements Feature {
  private readonly logger = getLogger("FileTracker");
  private fileList: LRUList = new LRUList(
    this.configurations.getMergedConfig().completion.prompt.collectSnippetsFromRecentOpenedFiles.maxOpenedFiles,
  );

  constructor(private readonly configurations: Configurations) {}
  initialize(connection: Connection): ServerCapabilities | Promise<ServerCapabilities> {
    connection.onNotification(OpenedFileRequest.type, (param: OpenedFileParams) => {
      console.log("Received opened file request:" + param.action);
      this.resolveOpenedFileRequest(param);
    });

    return {};
  }

  resolveOpenedFileRequest(param: OpenedFileParams): void {
    switch (param.action) {
      case "change":
        this.changeFile(param);
        break;
      //TODO(Sma1lboy): This feature may change in the feature, remain this for testing
      case "test":
        this.logger.info("test lru list: " + this.fileList.toString());
        this.logger.info("get all gonna return: " + JSON.stringify(this.getAllFilesWithoutActive()));
        break;
      default:
        this.logger.warn(`Unhandled action: ${param.action}`);
        break;
    }
  }

  changeFile(param: OpenedFileParams) {
    const { activeEditor, visibleEditors } = param;

    const visitedPaths = new Set<string>();

    //get all visible editors
    if (visibleEditors) {
      visibleEditors.forEach((editor) => {
        const visibleFile = this.fileList.get(editor.uri);
        if (visibleFile) {
          visibleFile.lastVisibleRange = [];
        }
      });

      visibleEditors.forEach((editor) => {
        this.logger.info("visible editor name with range:" + editor.uri + editor.visibleRange);
        let visibleFile = this.fileList.get(editor.uri);
        if (!visibleFile) {
          visibleFile = {
            uri: editor.uri,
            lastVisibleRange: [editor.visibleRange],
            invisible: false,
            isActive: false,
          };
          this.fileList.insert(visibleFile);
        } else {
          if (visitedPaths.has(visibleFile.uri)) {
            const idx = visibleFile.lastVisibleRange.findIndex((range) => this.rangesEqual(range, editor.visibleRange));
            if (idx === -1) {
              visibleFile.lastVisibleRange = [editor.visibleRange, ...visibleFile.lastVisibleRange];
            }
            visibleFile.invisible = false;
          } else {
            visibleFile.invisible = false;
            visibleFile.lastVisibleRange = [editor.visibleRange];
          }
        }
        visitedPaths.add(visibleFile.uri);
      });
    }

    // //get active editor
    let file = this.fileList.get(activeEditor.uri);
    if (!file) {
      file = {
        uri: activeEditor.uri,
        lastVisibleRange: [activeEditor.visibleRange],
        invisible: false,
        isActive: true,
      };
      this.fileList.insert(file);
    } else {
      if (visitedPaths.has(file.uri)) {
        const idx = file.lastVisibleRange.findIndex((range) => this.rangesEqual(range, activeEditor.visibleRange));
        if (idx === -1) {
          file.lastVisibleRange = [activeEditor.visibleRange, ...file.lastVisibleRange];
        }
      } else {
        file.lastVisibleRange = [activeEditor.visibleRange];
      }
      file.invisible = false;
      file.isActive = true;
    }
    visitedPaths.add(file.uri);

    //set invisible flag for all files that are not in the current file list
    this.fileList.getAll().forEach((file) => {
      if (!visitedPaths.has(file.uri)) {
        file.invisible = true;
      }
      if (file.uri !== activeEditor.uri) {
        file.isActive = false;
      }
    });
  }

  /**
   * Return All recently opened files by order. [recently opened, ..., oldest] without active file
   * @returns return all recently opened files by order
   */
  getAllFilesWithoutActive(): OpenedFile[] {
    return this.fileList.getAll().filter((f) => !f.isActive);
  }

  private rangesEqual(range1: Range, range2: Range): boolean {
    return (
      range1.start.line === range2.start.line &&
      range1.start.character === range2.start.character &&
      range1.end.line === range2.end.line &&
      range1.end.character === range2.end.character
    );
  }
}
