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

    //TODO: Get all diagnostics from the context and substring with max command chars
    const lspErrors = context.diagnostics[0];
    getLogger("QuickFixCodeActionProvider").info("lspErrors", lspErrors);

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
