import { CodeActionProvider } from "vscode";
import { InlineEditCodeActionProvider } from "./code-action/InlineEdit";
import { QuickFixCodeActionProvider } from "./code-action/QuickFix";
import { ContextVariables } from "./ContextVariables";
import { Client } from "./lsp/Client";
export class CodeActions {
  private codeActionProviders: CodeActionProvider[] = [
    new InlineEditCodeActionProvider(this.contextVariables),
    new QuickFixCodeActionProvider(this.contextVariables),
  ];
  constructor(
    private readonly client: Client,
    private readonly contextVariables: ContextVariables,
  ) {
    this.codeActionProviders.forEach((provider) => {
      this.client.registerCodeActionProvider(provider);
    });
  }
}
