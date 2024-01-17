/* eslint-disable @typescript-eslint/no-unused-vars */
import { URI } from "vscode-uri";

import { ContextFile } from "../codebase-context/messages";

export interface ActiveTextEditor {
  content: string;
  filePath: string;
  fileUri?: URI;
  repoName?: string;
  revision?: string;
  selectionRange?: ActiveTextEditorSelectionRange;
}

export interface ActiveTextEditorSelectionRange {
  start: {
    line: number;
    character: number;
  };
  end: {
    line: number;
    character: number;
  };
}

export interface ActiveTextEditorSelection {
  fileName: string;
  fileUri?: URI;
  repoName?: string;
  revision?: string;
  precedingText: string;
  selectedText: string;
  followingText: string;
  selectionRange?: ActiveTextEditorSelectionRange | null;
}

export type ActiveTextEditorDiagnosticType = "error" | "warning" | "information" | "hint";

export interface ActiveTextEditorDiagnostic {
  type: ActiveTextEditorDiagnosticType;
  range: ActiveTextEditorSelectionRange;
  text: string;
  message: string;
}

export interface ActiveTextEditorVisibleContent {
  content: string;
  fileName: string;
  fileUri?: URI;
  repoName?: string;
  revision?: string;
}

export interface TextDocumentContent {
  content: string;
  fileName: string;
  fileUri?: URI;
  repoName?: string;
  revision?: string;
}

/**
 * The intent classification for the fixup.
 * Manually determined depending on how the fixup is triggered.
 */
export type FixupIntent = "add" | "edit" | "fix" | "doc";

export interface VsCodeFixupTaskRecipeData {
  instruction: string;
  intent: FixupIntent;
  fileName: string;
  precedingText: string;
  selectedText: string;
  followingText: string;
  selectionRange: ActiveTextEditorSelectionRange;
}

export interface VsCodeFixupController {
  getTaskRecipeData(taskId: string): Promise<VsCodeFixupTaskRecipeData | undefined>;
}

export interface VsCodeCommandsController {
  addCommand(key: string, input?: string, contextFiles?: ContextFile[], addEnhancedContext?: boolean): Promise<string>;
  menu(type: "custom" | "config" | "default", showDesc?: boolean): Promise<void>;
}

export interface ActiveTextEditorViewControllers<
  F extends VsCodeFixupController = VsCodeFixupController,
  C extends VsCodeCommandsController = VsCodeCommandsController,
> {
  readonly fixups?: F;
  readonly command?: C;
}

export interface Editor<
  F extends VsCodeFixupController = VsCodeFixupController,
  P extends VsCodeCommandsController = VsCodeCommandsController,
> {
  controllers?: ActiveTextEditorViewControllers<F, P>;

  /**
   * The path of the workspace root if on the file system, otherwise `null`.
   * @deprecated Use {@link Editor.getWorkspaceRootUri} instead.
   */
  getWorkspaceRootPath(): string | null;

  /** The URI of the workspace root. */
  getWorkspaceRootUri(): URI | null;

  getActiveTextEditor(): ActiveTextEditor | null;
  getActiveTextEditorSelection(): ActiveTextEditorSelection | null;
  getActiveTextEditorSmartSelection(): Promise<ActiveTextEditorSelection | null>;

  /**
   * Gets the active text editor's selection, or the entire file if the selected range is empty.
   */
  getActiveTextEditorSelectionOrEntireFile(): ActiveTextEditorSelection | null;
  /**
   * Gets the active text editor's selection, or the visible content if the selected range is empty.
   */
  getActiveTextEditorSelectionOrVisibleContent(): ActiveTextEditorSelection | null;
  /**
   * Get diagnostics (errors, warnings, hints) for a range within the active text editor.
   */
  getActiveTextEditorDiagnosticsForRange(range: ActiveTextEditorSelectionRange): ActiveTextEditorDiagnostic[] | null;

  getActiveTextEditorVisibleContent(): ActiveTextEditorVisibleContent | null;

  getTextEditorContentForFile(uri: URI, range?: ActiveTextEditorSelectionRange): Promise<string | undefined>;

  replaceSelection(fileName: string, selectedText: string, replacement: string): Promise<void>;
  showQuickPick(labels: string[]): Promise<string | undefined>;
  showWarningMessage(message: string): Promise<void>;
  showInputBox(prompt?: string): Promise<string | undefined>;

  // TODO: When Non-Stop Fixup doesn't depend directly on the chat view,
  // move the recipe to vscode and remove this entrypoint.
  didReceiveFixupText(id: string, text: string, state: "streaming" | "complete"): Promise<void>;
}

export class NoopEditor implements Editor {
  public controllers?: ActiveTextEditorViewControllers<VsCodeFixupController, VsCodeCommandsController> | undefined;

  public getWorkspaceRootPath(): string | null {
    return null;
  }

  public getWorkspaceRootUri(): URI | null {
    return null;
  }

  public getActiveTextEditor(): ActiveTextEditor | null {
    return null;
  }

  public getActiveTextEditorSelection(): ActiveTextEditorSelection | null {
    return null;
  }

  public getActiveTextEditorSmartSelection(): Promise<ActiveTextEditorSelection | null> {
    return Promise.resolve(null);
  }

  public getActiveTextEditorSelectionOrEntireFile(): ActiveTextEditorSelection | null {
    return null;
  }

  public getActiveTextEditorSelectionOrVisibleContent(): ActiveTextEditorSelection | null {
    return null;
  }

  public getActiveTextEditorDiagnosticsForRange(): ActiveTextEditorDiagnostic[] | null {
    return null;
  }

  public getActiveTextEditorVisibleContent(): ActiveTextEditorVisibleContent | null {
    return null;
  }

  public getTextEditorContentForFile(_uri: URI, _range?: ActiveTextEditorSelectionRange): Promise<string | undefined> {
    return Promise.resolve(undefined);
  }

  public replaceSelection(_fileName: string, _selectedText: string, _replacement: string): Promise<void> {
    return Promise.resolve();
  }

  public showQuickPick(_labels: string[]): Promise<string | undefined> {
    return Promise.resolve(undefined);
  }

  public showWarningMessage(_message: string): Promise<void> {
    return Promise.resolve();
  }

  public showInputBox(_prompt?: string): Promise<string | undefined> {
    return Promise.resolve(undefined);
  }

  public didReceiveFixupText(_id: string, _text: string, _state: "streaming" | "complete"): Promise<void> {
    return Promise.resolve();
  }
}
