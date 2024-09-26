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

    const lspErrors = context.diagnostics
      .map((diagnostic, idx) => "Error " + idx + ": " + diagnostic.message)
      .join("\n");
    const quickFixCmd = `Here is some error information that occurred in the selection:
                        ${lspErrors}
                        Please provide the correct command to fix the error.`;
    getLogger("QuickFixCodeActionProvider").info("lspErrors", lspErrors);

    const quickFixEditing = new CodeAction("Fix using Tabby", CodeActionKind.QuickFix);
    quickFixEditing.command = {
      command: "tabby.chat.edit.start",
      title: "Fix using Tabby",
      arguments: [quickFixCmd],
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
