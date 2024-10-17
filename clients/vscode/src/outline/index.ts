import {
  window,
  TextEditor,
  TextDocument,
  Range,
  Position,
  CancellationTokenSource,
  ProgressLocation,
  Uri,
} from "vscode";
import { ContextVariables } from "../ContextVariables";
import { OutlinesProvider } from "./OutlinesProvider";
import { getLogger } from "../logger";

export class OutlinesGenerator {
  private outlinesCancellationTokenSource: CancellationTokenSource | null = null;

  constructor(
    private contextVariables: ContextVariables,
    private outlinesProvider: OutlinesProvider,
  ) {}

  async generate() {
    const editor = window.activeTextEditor;
    if (!editor) {
      return;
    }

    const editLocation = this.getEditLocation(editor);

    await window.withProgress(
      {
        location: ProgressLocation.Notification,
        title: "Generating natural language outlines...",
        cancellable: true,
      },
      async (_, token) => {
        this.contextVariables.outlinesGenerationInProgress = true;
        if (token.isCancellationRequested) {
          return;
        }

        this.outlinesCancellationTokenSource = new CancellationTokenSource();
        token.onCancellationRequested(() => {
          this.outlinesCancellationTokenSource?.cancel();
        });

        try {
          await this.outlinesProvider.provideOutlinesGenerate({
            location: editLocation,
            editor: editor,
          });
        } catch (error) {
          if (typeof error === "object" && error && "message" in error && typeof error.message === "string") {
            window.showErrorMessage(`Error generating outlines: ${error.message}`);
          }
        } finally {
          this.outlinesCancellationTokenSource?.dispose();
          this.outlinesCancellationTokenSource = null;
          this.contextVariables.outlinesGenerationInProgress = false;
        }
      },
    );
  }

  async editOutline(uri?: Uri, startLine?: number) {
    const editor = window.activeTextEditor;
    if (!editor) return;

    let documentUri: string;
    let line: number;
    if (uri && startLine !== undefined) {
      documentUri = uri.toString();
      line = startLine;
    } else {
      documentUri = editor.document.uri.toString();
      line = editor.selection.active.line;
    }

    const content = this.outlinesProvider.getOutline(documentUri, line);
    if (!content) return;

    const quickPick = window.createQuickPick();
    quickPick.items = [{ label: content }];
    quickPick.placeholder = "Edit NL Outline content";
    quickPick.value = content;

    quickPick.onDidAccept(async () => {
      const newContent = quickPick.value;
      quickPick.hide();
      await window.withProgress(
        {
          location: ProgressLocation.Notification,
          title: "Updating NL Outline",
          cancellable: false,
        },
        async (progress) => {
          progress.report({ increment: 0 });
          try {
            await this.outlinesProvider.updateOutline(documentUri, line, newContent);
            progress.report({ increment: 100 });
            window.showInformationMessage(`Updated NL Outline: ${newContent}`);
          } catch (error) {
            getLogger().error("Error updating NL Outline:", error);
            window.showErrorMessage(
              `Error updating NL Outline: ${error instanceof Error ? error.message : String(error)}`,
            );
          }
        },
      );
    });

    quickPick.show();
  }

  async acceptOutline() {
    const editor = window.activeTextEditor;
    if (!editor) {
      return;
    }
    await this.outlinesProvider.resolveOutline("accept");
  }

  async discardOutline() {
    const editor = window.activeTextEditor;
    if (!editor) {
      return;
    }
    await this.outlinesProvider.resolveOutline("discard");
  }

  private getEditLocation(editor: TextEditor): { uri: Uri; range: Range } {
    if (!editor.selection.isEmpty) {
      return {
        uri: editor.document.uri,
        range: new Range(editor.selection.start, editor.selection.end),
      };
    }

    const visibleRanges = editor.visibleRanges;
    if (visibleRanges.length > 0) {
      const firstVisibleLine = visibleRanges[0]?.start.line;
      const lastVisibleLine = visibleRanges[visibleRanges.length - 1]?.end.line;
      if (firstVisibleLine === undefined || lastVisibleLine === undefined) {
        throw new Error("Unable to determine visible range");
      }
      const offsetRange = this.getOffsetRange(editor.document, firstVisibleLine, lastVisibleLine, 20);
      return {
        uri: editor.document.uri,
        range: offsetRange,
      };
    }

    const currentLine = editor.selection.active.line;
    const offsetRange = this.getOffsetRange(editor.document, currentLine, currentLine, 20);
    return {
      uri: editor.document.uri,
      range: offsetRange,
    };
  }

  private getOffsetRange(document: TextDocument, start: number, end: number, offset: number): Range {
    const offsetStart = Math.max(0, start - offset);
    const offsetEnd = Math.min(document.lineCount - 1, end + offset);
    return new Range(new Position(offsetStart, 0), document.lineAt(offsetEnd).range.end);
  }
}
