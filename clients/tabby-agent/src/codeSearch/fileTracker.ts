import { Connection, Range } from "vscode-languageserver";
import { Feature } from "../feature";
import { DidChangeActiveEditorNotification, DidChangeActiveEditorParams, ServerCapabilities } from "../protocol";
import { Configurations } from "../config";
import { LRUCache } from "lru-cache";
import { isRangeEqual } from "../utils/range";

interface OpenedFile {
  uri: string;
  //order by range, the left most is the most recent one
  lastVisibleRange: Range[];
  invisible: boolean;
  isActive: boolean;
}

export class FileTracker implements Feature {
  private fileList = new LRUCache<string, OpenedFile>({
    max: this.configurations.getMergedConfig().completion.prompt.collectSnippetsFromRecentOpenedFiles.maxOpenedFiles,
  });

  constructor(private readonly configurations: Configurations) {}
  initialize(connection: Connection): ServerCapabilities | Promise<ServerCapabilities> {
    connection.onNotification(DidChangeActiveEditorNotification.type, (param: DidChangeActiveEditorParams) => {
      this.resolveChangedFile(param);
    });
    return {};
  }

  resolveChangedFile(param: DidChangeActiveEditorParams) {
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
        let visibleFile = this.fileList.get(editor.uri);
        if (!visibleFile) {
          visibleFile = {
            uri: editor.uri,
            lastVisibleRange: [editor.range],
            invisible: false,
            isActive: false,
          };
          this.fileList.set(editor.uri, visibleFile);
        } else {
          if (visitedPaths.has(visibleFile.uri)) {
            const idx = visibleFile.lastVisibleRange.findIndex((range) => isRangeEqual(range, editor.range));
            if (idx === -1) {
              visibleFile.lastVisibleRange = [editor.range, ...visibleFile.lastVisibleRange];
            }
            visibleFile.invisible = false;
          } else {
            visibleFile.invisible = false;
            visibleFile.lastVisibleRange = [editor.range];
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
        lastVisibleRange: [activeEditor.range],
        invisible: false,
        isActive: true,
      };
      this.fileList.set(activeEditor.uri, file);
    } else {
      if (visitedPaths.has(file.uri)) {
        const idx = file.lastVisibleRange.findIndex((range) => isRangeEqual(range, activeEditor.range));
        if (idx === -1) {
          file.lastVisibleRange = [activeEditor.range, ...file.lastVisibleRange];
        }
      } else {
        file.lastVisibleRange = [activeEditor.range];
      }
      file.invisible = false;
      file.isActive = true;
    }
    visitedPaths.add(file.uri);

    //set invisible flag for all files that are not in the current file list
    Array.from(this.fileList.values())
      .filter(this.isOpenedFile)
      .forEach((file) => {
        if (!visitedPaths.has(file.uri)) {
          file.invisible = true;
        }
        if (file.uri !== activeEditor.uri) {
          file.isActive = false;
        }
      });
  }
  private isOpenedFile(file: unknown): file is OpenedFile {
    return (file as OpenedFile).uri !== undefined;
  }

  /**
   * Return All recently opened files by order. [recently opened, ..., oldest] without active file
   * @returns return all recently opened files by order
   */
  getAllFilesWithoutActive(): OpenedFile[] {
    return Array.from(this.fileList.values())
      .filter(this.isOpenedFile)
      .filter((f) => !f.isActive);
  }
}
