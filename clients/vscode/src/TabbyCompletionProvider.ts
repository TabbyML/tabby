import {
  CancellationToken,
  InlineCompletionContext,
  InlineCompletionItem,
  InlineCompletionItemProvider,
  InlineCompletionTriggerKind,
  Position,
  Range,
  TextDocument,
  workspace,
} from "vscode";
import { CompletionResponse } from "tabby-agent";
import { agent } from "./agent";
import { notifications } from "./notifications";

export class TabbyCompletionProvider implements InlineCompletionItemProvider {
  // User Settings
  private enabled: boolean = true;

  constructor() {
    this.updateConfiguration();
    workspace.onDidChangeConfiguration((event) => {
      if (event.affectsConfiguration("tabby") || event.affectsConfiguration("editor.inlineSuggest")) {
        this.updateConfiguration();
      }
    });
  }

  public async provideInlineCompletionItems(
    document: TextDocument,
    position: Position,
    context: InlineCompletionContext,
    token: CancellationToken,
  ): Promise<InlineCompletionItem[]> {
    const emptyResponse = Promise.resolve([] as InlineCompletionItem[]);
    if (!this.enabled) {
      console.debug("Extension not enabled, skipping.");
      return emptyResponse;
    }

    // Check if autocomplete widget is visible
    if (context.selectedCompletionInfo !== undefined) {
      console.debug("Autocomplete widget is visible, skipping.");
      return emptyResponse;
    }

    if (token?.isCancellationRequested) {
      console.debug("Cancellation was requested.");
      return emptyResponse;
    }

    const replaceRange = this.calculateReplaceRange(document, position);

    const request = {
      filepath: document.uri.fsPath,
      language: document.languageId, // https://code.visualstudio.com/docs/languages/identifiers
      text: document.getText(),
      position: document.offsetAt(position),
      manually: context.triggerKind === InlineCompletionTriggerKind.Invoke,
    };

    const abortController = new AbortController();
    token?.onCancellationRequested(() => {
      console.debug("Cancellation requested.");
      abortController.abort();
    });

    const completion = await agent()
      .provideCompletions(request, { signal: abortController.signal })
      .catch((_) => {
        return null;
      });

    const completions = this.toInlineCompletions(completion, replaceRange);
    return Promise.resolve(completions);
  }

  private updateConfiguration() {
    const configuration = workspace.getConfiguration("tabby");
    this.enabled = configuration.get("codeCompletion", true);
    this.checkInlineCompletionEnabled();
  }

  private checkInlineCompletionEnabled() {
    const configuration = workspace.getConfiguration("editor.inlineSuggest");
    const inlineSuggestEnabled = configuration.get("enabled", true);
    if (this.enabled && !inlineSuggestEnabled) {
      console.debug("Tabby code completion is enabled but inline suggest is disabled.");
      notifications.showInformationWhenInlineSuggestDisabled();
    }
  }

  private toInlineCompletions(tabbyCompletion: CompletionResponse | null, range: Range): InlineCompletionItem[] {
    return (
      tabbyCompletion?.choices?.map((choice: any) => {
        let event = {
          type: "select",
          completion_id: tabbyCompletion.id,
          choice_index: choice.index,
        };
        return new InlineCompletionItem(choice.text, range, {
          title: "",
          command: "tabby.emitEvent",
          arguments: [event],
        });
      }) || []
    );
  }

  private hasSuffixParen(document: TextDocument, position: Position) {
    const suffix = document.getText(
      new Range(position.line, position.character, position.line, position.character + 1),
    );
    return ")]}".indexOf(suffix) > -1;
  }

  // FIXME: move replace range calculation to tabby-agent
  private calculateReplaceRange(document: TextDocument, position: Position): Range {
    const hasSuffixParen = this.hasSuffixParen(document, position);
    if (hasSuffixParen) {
      return new Range(position.line, position.character, position.line, position.character + 1);
    } else {
      return new Range(position, position);
    }
  }
}
