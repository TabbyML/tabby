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

export class InlineEditCodeActionProvider implements CodeActionProviderInterface {
  constructor(private readonly contextVariables: ContextVariables) {}

  provideCodeActions(
    _document: TextDocument,
    _range: Range | Selection,
    _context: CodeActionContext,
    token: CancellationToken,
  ): CodeAction[] | undefined {
    if (token.isCancellationRequested) {
      return;
    }

    if (!this.contextVariables.chatEnabled) {
      return;
    }

    const inlineEditing = new CodeAction("Edit using Tabby", CodeActionKind.RefactorRewrite);
    inlineEditing.command = {
      command: "tabby.chat.edit.start",
      title: "Edit using Tabby",
    };

    return [inlineEditing];
  }
}
