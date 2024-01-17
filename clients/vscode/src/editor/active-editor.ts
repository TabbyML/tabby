import * as vscode from "vscode";

/**
 * Gets the currently active text editor instance if available.
 * Returns undefined if no editor is active.
 *
 * NOTE: This handles edge case where activeTextEditor API returns undefined when webview panel has focus.
 */
let lastTrackedTextEditor: vscode.TextEditor | undefined;

// Support file, untitled, and notebooks
const validFileSchemes = new Set([
  "file",
  "untitled",
  "vscode-notebook",
  "vscode-notebook-cell",
]);

export function getActiveEditor(): vscode.TextEditor | undefined {
  // When there is no active editor, reset lastTrackedTextEditor
  const activeEditors = vscode.window.visibleTextEditors;
  if (!activeEditors.length) {
    lastTrackedTextEditor = undefined;
    return undefined;
  }

  // When the webview panel is focused, calling activeTextEditor will return undefined.
  // This allows us to get the active editor before the webview panel is focused.
  const get = (): vscode.TextEditor | undefined => {
    const activeEditor =
      vscode.window.activeTextEditor || vscode.window.visibleTextEditors[0];
    if (activeEditor?.document.uri.scheme) {
      if (validFileSchemes.has(activeEditor.document.uri.scheme)) {
        lastTrackedTextEditor = activeEditor;
      }
    }
    return lastTrackedTextEditor;
  };

  return get();
}
