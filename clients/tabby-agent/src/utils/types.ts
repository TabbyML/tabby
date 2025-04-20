import { Range } from "vscode-languageserver";
import { TextDocument } from "vscode-languageserver-textdocument";

export interface DocumentRange {
  document: TextDocument;
  range: Range;
}
