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
    const userCommand = `Here is some error information that occurred in the selection:
                        ${lspErrors}
                        Please provide the correct command to fix the error.`;
    getLogger("QuickFixCodeActionProvider").info("lspErrors", lspErrors);

    const quickFixEditing = new CodeAction("Fix with Tabby", CodeActionKind.QuickFix);
    quickFixEditing.command = {
      command: "tabby.chat.edit.start",
      title: "Fix with Tabby",
      arguments: [userCommand],
    };

    return [quickFixEditing];
  }
}
