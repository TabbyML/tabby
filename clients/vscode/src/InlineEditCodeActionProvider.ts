import {
  CancellationToken,
  CodeAction,
  CodeActionContext,
  CodeActionKind,
  CodeActionProvider,
  Range,
  Selection,
  TextDocument,
} from "vscode";

export class InlineEditCodeActionProvider implements CodeActionProvider {
  provideCodeActions(
    _document: TextDocument,
    _range: Range | Selection,
    _context: CodeActionContext,
    token: CancellationToken,
  ): CodeAction[] | undefined {
    if (token.isCancellationRequested) {
      return;
    }
    const quickFix = new CodeAction("Inline Edit with Tabby", CodeActionKind.QuickFix);

    quickFix.command = {
      command: "tabby.chat.edit.start",
      title: "Start Tabby Chat Edit",
    };

    return [quickFix];
  }
}
