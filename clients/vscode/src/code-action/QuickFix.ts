import {
  CancellationToken,
  CodeAction,
  CodeActionContext,
  CodeActionKind,
  CodeActionProvider as CodeActionProviderInterface,
  Range,
  Selection,
  TextDocument,
} from "vscode";
import { ContextVariables } from "../ContextVariables";
import { getLogger } from "../logger";

export class QuickFixCodeActionProvider implements CodeActionProviderInterface {
  private readonly logger = getLogger("QuickFixCodeActionProvider");
  constructor(private readonly contextVariables: ContextVariables) {}

  provideCodeActions(
    _document: TextDocument,
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

    const getMergedDiagnosticRange = () => {
      return context.diagnostics.reduce(
        (mergedRange, diagnostic) => {
          if (!mergedRange) {
            return diagnostic.range;
          }
          return mergedRange.union(diagnostic.range);
        },
        null as Range | null,
      );
    };

    const mergedRange = getMergedDiagnosticRange();
    if (!mergedRange) {
      return [];
    }

    const lspErrors = context.diagnostics
      .map((diagnostic, idx) => "Error " + idx + ": " + diagnostic.message)
      .join("\n");

    const quickFixCmd = `Here is some error information that occurred in the selection:
                        ${lspErrors}
                        Please provide the correct command to fix the error.`;
    this.logger.trace("LSP Errors collections: ", lspErrors);
    this.logger.debug("QuickFix Range: ", mergedRange);

    const quickFixEditing = new CodeAction("Fix using Tabby", CodeActionKind.QuickFix);

    quickFixEditing.command = {
      command: "tabby.chat.edit.start",
      title: "Fix using Tabby",
      arguments: [undefined, mergedRange, quickFixCmd],
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
