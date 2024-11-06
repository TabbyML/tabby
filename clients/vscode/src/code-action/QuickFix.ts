import {
  CancellationToken,
  CodeAction,
  CodeActionContext,
  CodeActionKind,
  CodeActionProvider as CodeActionProviderInterface,
  Position,
  Range,
  Selection,
  TextDocument,
  languages,
} from "vscode";
import { ContextVariables } from "../ContextVariables";
import { getLogger } from "../logger";

// quick fix range offset for each direction
const QUICK_FIX_OFFSET = 5;

export class QuickFixCodeActionProvider implements CodeActionProviderInterface {
  private readonly logger = getLogger("QuickFixCodeActionProvider");
  constructor(private readonly contextVariables: ContextVariables) {}

  provideCodeActions(
    document: TextDocument,
    _range: Range | Selection,
    context: CodeActionContext,
    token: CancellationToken,
  ): CodeAction[] | undefined {
    if (token.isCancellationRequested) {
      return;
    }
    if (!this.contextVariables.chatEnabled) {
      return;
    }
    if (context.diagnostics.length === 0) {
      return [];
    }

    const allDiagnostics = languages.getDiagnostics(document.uri);

    const getLargestDiagnosticRange = () => {
      return context.diagnostics.reduce((maxDiag, currentDiag) => {
        const maxLineDiff = maxDiag.range.end.line - maxDiag.range.start.line;
        const currentLineDiff = currentDiag.range.end.line - currentDiag.range.start.line;
        if (currentLineDiff > maxLineDiff) {
          return currentDiag;
        }
        if (currentLineDiff === maxLineDiff) {
          const maxCharDiff = maxDiag.range.end.character - maxDiag.range.start.character;
          const currentCharDiff = currentDiag.range.end.character - currentDiag.range.start.character;
          return currentCharDiff > maxCharDiff ? currentDiag : maxDiag;
        }
        return maxDiag;
      });
    };

    const getExpandedErrorRange = () => {
      const baseDiagnostic = getLargestDiagnosticRange();
      let startLine = baseDiagnostic.range.start.line;
      let endLine = baseDiagnostic.range.end.line;
      const lineCount = document.lineCount;

      const hasErrorOnLine = (line: number) => {
        return allDiagnostics.some((diag) => {
          const isNearContext = Math.abs(diag.range.start.line - baseDiagnostic.range.start.line) <= 10;
          return isNearContext && diag.range.start.line <= line && line <= diag.range.end.line;
        });
      };

      while (startLine > 0 && hasErrorOnLine(startLine - 1)) {
        startLine--;
      }

      while (endLine < lineCount - 1 && hasErrorOnLine(endLine + 1)) {
        endLine++;
      }

      const expandedDiagnostics = allDiagnostics.filter(
        (diag) => diag.range.start.line >= startLine && diag.range.end.line <= endLine,
      );

      return {
        range: new Range(new Position(startLine, 0), new Position(endLine, document.lineAt(endLine).text.length)),
        diagnostics: expandedDiagnostics,
      };
    };

    const { range: lspExpandedRange, diagnostics: expandedDiagnostics } = getExpandedErrorRange();

    const lspErrors = expandedDiagnostics
      .map((diagnostic, idx) => "Error " + idx + ": " + diagnostic.message)
      .join("\n");

    const quickFixCmd = `Here is some error information that occurred in the selection:
                        ${lspErrors}
                        Please provide the correct command to fix the error.`;
    this.logger.trace("lspErrors", lspErrors);

    const quickFixEditing = new CodeAction("Fix using Tabby", CodeActionKind.QuickFix);

    this.logger.debug("Lsp error expanded rang: ", lspExpandedRange);
    const quickFixRange = new Range(
      new Position(lspExpandedRange.start.line - QUICK_FIX_OFFSET, lspExpandedRange.start.character),
      new Position(lspExpandedRange.end.line + QUICK_FIX_OFFSET, lspExpandedRange.end.character),
    );

    this.logger.debug("QuickFix range: ", quickFixRange);

    quickFixEditing.command = {
      command: "tabby.chat.edit.start",
      title: "Fix using Tabby",
      arguments: [quickFixCmd, quickFixRange],
    };

    const explainErrorCmd = `\nHere is some error information that occurred in the selection:
                        ${lspErrors}
                        Please provide an explanation for the error.`;
    const explainError = new CodeAction("Explain using Tabby", CodeActionKind.QuickFix);
    explainError.command = {
      command: "tabby.chat.explainCodeBlock",
      title: "Explain using Tabby",
      arguments: [explainErrorCmd],
    };

    return [quickFixEditing, explainError];
  }
}
