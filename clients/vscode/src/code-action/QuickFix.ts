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

    const lspErrors = context.diagnostics.map((d) => d.message).join("\n");
    getLogger("QuickFixCodeActionProvider").info("lspErrors", lspErrors);
    if (lspErrors.length === 0) {
      return [];
    }

    const quickFixEditing = new CodeAction("Fixing with Tabby", CodeActionKind.QuickFix);
    quickFixEditing.command = {
      command: "tabby.chat.edit.start",
      title: "Fixing with Tabby",
      arguments: [
        {
          errorContext: lspErrors,
        },
      ],
    };

    return [quickFixEditing];
  }
}
