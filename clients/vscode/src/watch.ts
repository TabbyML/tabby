import { parseAllVisibleDocuments, updateParseTreeOnEdit } from "./tree-sitter/parse-tree-cache";
import vscode from "vscode";

export const watchVisibleDocuments = () => {
  parseAllVisibleDocuments();

  return [
    vscode.window.onDidChangeVisibleTextEditors(parseAllVisibleDocuments),
    vscode.workspace.onDidChangeTextDocument(updateParseTreeOnEdit),
  ];
};
