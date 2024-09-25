import type { Feature } from "./feature";
import {
  Range,
  Location,
  Connection,
  CancellationToken,
  WorkDoneProgressReporter,
  ResultProgressReporter,
  CodeLensParams,
} from "vscode-languageserver";
import { ClientCapabilities, ServerCapabilities, CodeLens, CodeLensType, ChangesPreviewLineType } from "./protocol";
import { TextDocuments } from "./lsp/textDocuments";
import { TextDocument } from "vscode-languageserver-textdocument";

const codeLensType: CodeLensType = "previewChanges";
const changesPreviewLineType = {
  header: "header" as ChangesPreviewLineType,
  footer: "footer" as ChangesPreviewLineType,
  commentsFirstLine: "commentsFirstLine" as ChangesPreviewLineType,
  comments: "comments" as ChangesPreviewLineType,
  waiting: "waiting" as ChangesPreviewLineType,
  inProgress: "inProgress" as ChangesPreviewLineType,
  unchanged: "unchanged" as ChangesPreviewLineType,
  inserted: "inserted" as ChangesPreviewLineType,
  deleted: "deleted" as ChangesPreviewLineType,
};

export class CodeLensProvider implements Feature {
  constructor(private readonly documents: TextDocuments<TextDocument>) {}

  initialize(connection: Connection, clientCapabilities: ClientCapabilities): ServerCapabilities {
    if (clientCapabilities.textDocument?.codeLens) {
      connection.onCodeLens(async (params, token, workDoneProgress, resultProgress) => {
        return this.provideCodeLens(params, token, workDoneProgress, resultProgress);
      });

      return {
        codeLensProvider: {
          resolveProvider: false,
        },
      };
    }
    return {};
  }

  async provideCodeLens(
    params: CodeLensParams,
    token: CancellationToken,
    workDoneProgress?: WorkDoneProgressReporter | undefined,
    resultProgress?: ResultProgressReporter<CodeLens[]> | undefined,
  ): Promise<CodeLens[] | null> {
    const uri = params.textDocument.uri;
    const textDocument = this.documents.get(uri);
    if (!textDocument) {
      return null;
    }
    const codeLenses: CodeLens[] = [];
    let lineInPreviewBlock = -1;
    let previewBlockMarkers = "";
    for (let line = textDocument.lineCount - 1; line >= 0; line = line - 1) {
      if (token.isCancellationRequested) {
        return null;
      }
      const lineRange = { start: { line: line, character: 0 }, end: { line: line + 1, character: 0 } };
      const text = textDocument.getText(lineRange);
      const codeLensRange: Range = {
        start: { line: line, character: 0 },
        end: { line: line, character: text.length - 1 },
      };

      const codeLensLocation: Location = { uri: uri, range: codeLensRange };
      const lineCodeLenses: CodeLens[] = [];
      if (lineInPreviewBlock < 0) {
        const match = /^>>>>>>> (tabby-[0-9|a-z|A-Z]{6}) (\[.*\])/g.exec(text);
        const editId = match?.[1];
        const markers = match?.[2];
        if (match && markers && editId) {
          previewBlockMarkers = markers;
          lineInPreviewBlock = 0;
          lineCodeLenses.push({
            range: codeLensRange,
            data: {
              type: codeLensType,
              line: changesPreviewLineType.footer,
            },
          });
        }
      } else {
        const match = /^<<<<<<< (tabby-[0-9|a-z|A-Z]{6})/g.exec(text);
        const editId = match?.[1];
        if (match && editId) {
          lineInPreviewBlock = -1;

          if (previewBlockMarkers.includes(".")) {
            lineCodeLenses.push({
              range: codeLensRange,
              command: {
                title: "$(sync~spin) Tabby is working...",
                command: " ",
              },
              data: {
                type: codeLensType,
                line: changesPreviewLineType.header,
              },
            });
            lineCodeLenses.push({
              range: codeLensRange,
              command: {
                title: "Cancel",
                command: "tabby/chat/edit/resolve",
                arguments: [{ location: codeLensLocation, action: "cancel" }],
              },
              data: {
                type: codeLensType,
                line: changesPreviewLineType.header,
              },
            });
          } else if (!previewBlockMarkers.includes("x")) {
            lineCodeLenses.push({
              range: codeLensRange,
              command: {
                title: "$(check)Accept",
                command: "tabby/chat/edit/resolve",
                arguments: [{ location: codeLensLocation, action: "accept" }],
              },
              data: {
                type: codeLensType,
                line: changesPreviewLineType.header,
              },
            });
            lineCodeLenses.push({
              range: codeLensRange,
              command: {
                title: "$(remove-close)Discard",
                command: "tabby/chat/edit/resolve",
                arguments: [{ location: codeLensLocation, action: "discard" }],
              },
              data: {
                type: codeLensType,
                line: changesPreviewLineType.header,
              },
            });
          }
        } else {
          lineInPreviewBlock++;
          const marker = previewBlockMarkers[previewBlockMarkers.length - lineInPreviewBlock - 1];
          let codeLens: CodeLens | undefined = undefined;
          switch (marker) {
            case "#":
              codeLens = {
                range: codeLensRange,
                data: {
                  type: codeLensType,
                  line:
                    previewBlockMarkers.indexOf("#") === lineInPreviewBlock
                      ? changesPreviewLineType.commentsFirstLine
                      : changesPreviewLineType.comments,
                },
              };
              break;
            case ".":
              codeLens = {
                range: codeLensRange,
                data: {
                  type: codeLensType,
                  line: changesPreviewLineType.waiting,
                },
              };
              break;
            case "|":
              codeLens = {
                range: codeLensRange,
                data: {
                  type: codeLensType,
                  line: changesPreviewLineType.inProgress,
                },
              };
              break;
            case "=":
              codeLens = {
                range: codeLensRange,
                data: {
                  type: codeLensType,
                  line: changesPreviewLineType.unchanged,
                },
              };
              break;
            case "+":
              codeLens = {
                range: codeLensRange,
                data: {
                  type: codeLensType,
                  line: changesPreviewLineType.inserted,
                },
              };
              break;
            case "-":
              codeLens = {
                range: codeLensRange,
                data: {
                  type: codeLensType,
                  line: changesPreviewLineType.deleted,
                },
              };
              break;
            default:
              break;
          }
          if (codeLens) {
            codeLenses.push(codeLens);
          }
        }
      }
      if (lineCodeLenses.length > 0) {
        if (resultProgress) {
          resultProgress.report(lineCodeLenses);
        } else {
          codeLenses.push(...lineCodeLenses);
        }
      }
    }
    workDoneProgress?.done();
    if (resultProgress) {
      return null;
    } else {
      return codeLenses;
    }
  }
}
