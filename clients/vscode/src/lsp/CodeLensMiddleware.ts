import {
  window,
  Range,
  CodeLens as VscodeCodeLens,
  TextEditor,
  TextEditorDecorationType,
  TextDocument,
  CancellationToken,
  ThemeColor,
  DecorationRangeBehavior,
  OverviewRulerLane,
} from "vscode";
import { CodeLensMiddleware as VscodeLspCodeLensMiddleware, ProvideCodeLensesSignature } from "vscode-languageclient";
import { CodeLens as TabbyCodeLens } from "tabby-agent";
import { findTextEditor } from "./vscodeWindowUtils";

type CodeLens = VscodeCodeLens & TabbyCodeLens;

const decorationTypeHeader = window.createTextEditorDecorationType({
  backgroundColor: new ThemeColor("merge.incomingHeaderBackground"),
  isWholeLine: true,
  rangeBehavior: DecorationRangeBehavior.ClosedClosed,
  overviewRulerLane: OverviewRulerLane.Right,
});
const decorationTypeFooter = window.createTextEditorDecorationType({
  backgroundColor: new ThemeColor("merge.incomingHeaderBackground"),
  isWholeLine: true,
  rangeBehavior: DecorationRangeBehavior.ClosedClosed,
  overviewRulerLane: OverviewRulerLane.Right,
});
const decorationTypeComments = window.createTextEditorDecorationType({
  color: new ThemeColor("editorInlayHint.foreground"),
  backgroundColor: new ThemeColor("editorInlayHint.background"),
  fontStyle: "italic",
  fontWeight: "normal",
  isWholeLine: true,
  overviewRulerLane: OverviewRulerLane.Right,
  rangeBehavior: DecorationRangeBehavior.ClosedOpen,
  before: {
    contentText: ">",
    color: new ThemeColor("editorInlayHint.foreground"),
    backgroundColor: new ThemeColor("editorInlayHint.background"),
    fontWeight: "bold",
    width: "10px",
  },
});
const decorationTypeUnchanged = window.createTextEditorDecorationType({
  before: {
    contentText: "",
    fontWeight: "bold",
    width: "10px",
  },
});
const decorationTypeInserted = window.createTextEditorDecorationType({
  backgroundColor: new ThemeColor("diffEditor.insertedTextBackground"),
  isWholeLine: true,
  overviewRulerLane: OverviewRulerLane.Right,
  rangeBehavior: DecorationRangeBehavior.ClosedClosed,
  before: {
    contentText: "+",
    backgroundColor: new ThemeColor("diffEditor.insertedTextBackground"),
    fontWeight: "bold",
    width: "10px",
  },
});
const decorationTypeDeleted = window.createTextEditorDecorationType({
  backgroundColor: new ThemeColor("diffEditor.removedTextBackground"),
  isWholeLine: true,
  overviewRulerLane: OverviewRulerLane.Right,
  rangeBehavior: DecorationRangeBehavior.ClosedClosed,
  before: {
    contentText: "-",
    backgroundColor: new ThemeColor("diffEditor.removedTextBackground"),
    fontWeight: "bold",
    width: "10px",
  },
});

export class CodeLensMiddleware implements VscodeLspCodeLensMiddleware {
  private readonly decorationMap = new Map<TextEditor, Map<TextEditorDecorationType, Range[]>>();

  async provideCodeLenses(
    document: TextDocument,
    token: CancellationToken,
    next: ProvideCodeLensesSignature,
  ): Promise<CodeLens[] | undefined | null> {
    const codeLenses = await next(document, token);
    const editor = findTextEditor(document.uri);
    if (!editor) {
      return codeLenses;
    }
    this.removeDecorations(editor);
    const result =
      codeLenses
        ?.map((codeLens) => this.handleCodeLens(codeLens, editor))
        .filter((codeLens): codeLens is CodeLens => codeLens !== null) ?? [];
    this.purgeDecorationMap();
    return result;
  }

  private handleCodeLens(codeLens: CodeLens, editor: TextEditor): CodeLens | null {
    if (!codeLens.data || codeLens.data.type !== "previewChanges") {
      return codeLens;
    }
    const decorationRange = new Range(
      codeLens.range.start.line,
      codeLens.range.start.character,
      codeLens.range.end.line,
      codeLens.range.end.character,
    );
    switch (codeLens.data.line) {
      case "header":
        this.addDecorationRange(editor, decorationTypeHeader, decorationRange);
        return codeLens;
      case "footer":
        this.addDecorationRange(editor, decorationTypeFooter, decorationRange);
        return null;
      case "commentsFirstLine":
        this.addDecorationRange(editor, decorationTypeComments, decorationRange);
        return null;
      case "comments":
        this.addDecorationRange(editor, decorationTypeComments, decorationRange);
        return null;
      case "waiting":
        this.addDecorationRange(editor, decorationTypeUnchanged, decorationRange);
        return null;
      case "inProgress":
        this.addDecorationRange(editor, decorationTypeInserted, decorationRange);
        return null;
      case "unchanged":
        this.addDecorationRange(editor, decorationTypeUnchanged, decorationRange);
        return null;
      case "inserted":
        this.addDecorationRange(editor, decorationTypeInserted, decorationRange);
        return null;
      case "deleted":
        this.addDecorationRange(editor, decorationTypeDeleted, decorationRange);
        return null;
    }
    return null;
  }

  private addDecorationRange(editor: TextEditor, decorationType: TextEditorDecorationType, range: Range) {
    let decorations: Map<TextEditorDecorationType, Range[]> | undefined;
    if (this.decorationMap.has(editor)) {
      decorations = this.decorationMap.get(editor);
    }
    if (!decorations) {
      decorations = new Map();
      this.decorationMap.set(editor, decorations);
    }
    let ranges: Range[] | undefined;
    if (decorations.has(decorationType)) {
      ranges = decorations.get(decorationType);
    }
    if (!ranges) {
      ranges = [];
      decorations.set(decorationType, ranges);
    }
    ranges.push(range);
    editor.setDecorations(decorationType, ranges);
  }

  private removeDecorations(editor: TextEditor) {
    if (this.decorationMap.has(editor)) {
      const decorations = this.decorationMap.get(editor);
      decorations?.forEach((_, decorationType) => {
        editor.setDecorations(decorationType, []);
      });
      this.decorationMap.delete(editor);
    }
  }

  private purgeDecorationMap() {
    const editorsToRemove = [...this.decorationMap.keys()].filter(
      (editor) => !window.visibleTextEditors.includes(editor),
    );
    editorsToRemove.forEach((editor) => this.decorationMap.delete(editor));
  }
}
