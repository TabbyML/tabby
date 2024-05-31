import {
  Range,
  Location,
  Connection,
  CancellationToken,
  WorkDoneProgressReporter,
  ResultProgressReporter,
  CodeLensParams,
} from "vscode-languageserver";
import { ServerCapabilities, CodeLens, CodeLensType, ChangesPreviewLineType } from "./protocol";
import { TextDocuments } from "./TextDocuments";
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

export class CodeLensProvider {
  constructor(
    private readonly connection: Connection,
    private readonly documents: TextDocuments<TextDocument>,
  ) {
    this.connection.onCodeLens(async (params, token, workDoneProgress, resultProgress) => {
      return this.provideCodeLens(params, token, workDoneProgress, resultProgress);
    });
  }

  fillServerCapabilities(capabilities: ServerCapabilities): void {
    capabilities.codeLensProvider = {
      resolveProvider: false,
    };
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
    for (let line = 0; line < textDocument.lineCount; line++) {
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
        const match = /^<<<<<<<.+(<.*>)\[(tabby-[0-9|a-z|A-Z]{6})\]/g.exec(text);
        const markers = match?.[1];
        const editId = match?.[2];
        if (match && markers && editId) {
          lineInPreviewBlock = 0;
          previewBlockMarkers = markers;

          lineCodeLenses.push({
            range: codeLensRange,
            command: {
              title: "Accept",
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
              title: "Discard",
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
        const match = /^>>>>>>>.+(<.*>)\[(tabby-[0-9|a-z|A-Z]{6})\]/g.exec(text);
        const editId = match?.[2];
        if (match && editId) {
          lineInPreviewBlock = -1;
          lineCodeLenses.push({
            range: codeLensRange,
            data: {
              type: codeLensType,
              line: changesPreviewLineType.footer,
            },
          });
        } else {
          lineInPreviewBlock++;
          const marker = previewBlockMarkers[lineInPreviewBlock];
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
