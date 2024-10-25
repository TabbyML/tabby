import EventEmitter from "events";
import { ApplyWorkspaceEditParams, ApplyWorkspaceEditRequest } from "tabby-agent";
import { BaseLanguageClient, FeatureState, StaticFeature, TextEdit } from "vscode-languageclient";
import { Disposable, Position, Range, TextDocument, TextEditorEdit, window, workspace } from "vscode";
import { diffLines } from "diff";

export class WorkSpaceFeature extends EventEmitter implements StaticFeature {
  private disposables: Disposable[] = [];

  constructor(private readonly client: BaseLanguageClient) {
    super();
  }
  getState(): FeatureState {
    return { kind: "static" };
  }

  fillInitializeParams() {
    // nothing
  }

  fillClientCapabilities(): void {
    // nothing
  }

  preInitialize(): void {
    // nothing
  }

  initialize(): void {
    this.disposables.push(
      this.client.onRequest(ApplyWorkspaceEditRequest.type, (params: ApplyWorkspaceEditParams) => {
        return this.handleApplyWorkspaceEdit(params);
      }),
    );
  }

  clear(): void {
    this.disposables.forEach((disposable) => disposable.dispose());
    this.disposables = [];
  }

  private async handleApplyWorkspaceEdit(params: ApplyWorkspaceEditParams): Promise<boolean> {
    const { edit, options } = params;
    const activeEditor = window.activeTextEditor;
    if (!activeEditor) {
      return false;
    }

    try {
      const success = await activeEditor.edit(
        (editBuilder: TextEditorEdit) => {
          Object.entries(edit.changes || {}).forEach(([uri, textEdits]) => {
            const document = workspace.textDocuments.find((doc) => doc.uri.toString() === uri);
            if (document && document === activeEditor.document) {
              const textEdit = textEdits[0];
              if (textEdits.length === 1 && textEdit) {
                applyTextEditMinimalLineChange(editBuilder, textEdit, document);
              } else {
                textEdits.forEach((textEdit) => {
                  const range = new Range(
                    new Position(textEdit.range.start.line, textEdit.range.start.character),
                    new Position(textEdit.range.end.line, textEdit.range.end.character),
                  );
                  editBuilder.replace(range, textEdit.newText);
                });
              }
            }
          });
        },
        {
          undoStopBefore: options?.undoStopBefore ?? false,
          undoStopAfter: options?.undoStopAfter ?? false,
        },
      );

      return success;
    } catch (error) {
      return false;
    }
  }
}

function applyTextEditMinimalLineChange(editBuilder: TextEditorEdit, textEdit: TextEdit, document: TextDocument) {
  const documentRange = new Range(
    new Position(textEdit.range.start.line, textEdit.range.start.character),
    new Position(textEdit.range.end.line, textEdit.range.end.character),
  );

  const text = document.getText(documentRange);
  const newText = textEdit.newText;
  const diffs = diffLines(text, newText);

  let line = documentRange.start.line;
  for (const diff of diffs) {
    if (!diff.count) {
      continue;
    }

    if (diff.added) {
      editBuilder.insert(new Position(line, 0), diff.value);
    } else if (diff.removed) {
      const range = new Range(new Position(line + 0, 0), new Position(line + diff.count, 0));
      editBuilder.delete(range);
      line += diff.count;
    } else {
      line += diff.count;
    }
  }
}
