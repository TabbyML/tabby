import { window, TextEditor, Uri } from "vscode";

export function findTextEditor(uri: Uri): TextEditor | undefined {
  if (window.activeTextEditor?.document.uri.toString() === uri.toString()) {
    return window.activeTextEditor;
  }
  return window.visibleTextEditors.find((editor) => editor.document.uri.toString() === uri.toString());
}
