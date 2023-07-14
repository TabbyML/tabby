import {
  CancellationToken,
  InlineCompletionContext,
  InlineCompletionItem,
  InlineCompletionItemProvider,
  InlineCompletionList,
  Position,
  ProviderResult,
  Range,
  TextDocument,
  workspace,
} from "vscode";
import { CompletionResponse, CancelablePromise } from "tabby-agent";
import { agent } from "./agent";
import { sleep } from "./utils";

export class TabbyCompletionProvider implements InlineCompletionItemProvider {
  private uuid = Date.now();
  private latestTimestamp: number = 0;
  private pendingCompletion: CancelablePromise<CompletionResponse> | null = null;

  // User Settings
  private enabled: boolean = true;

  // These settings will be move to tabby-agent
  private suggestionDelay: number = 150;
  private maxPrefixLines: number = 20;
  private maxSuffixLines: number = 20;

  constructor() {
    this.updateConfiguration();
    workspace.onDidChangeConfiguration((event) => {
      if (event.affectsConfiguration("tabby")) {
        this.updateConfiguration();
      }
    });
  }

  //@ts-ignore because ASYNC and PROMISE
  //prettier-ignore
  public async provideInlineCompletionItems(document: TextDocument, position: Position, context: InlineCompletionContext, token: CancellationToken): ProviderResult<InlineCompletionItem[] | InlineCompletionList> {
    const emptyResponse = Promise.resolve([] as InlineCompletionItem[]);
    if (!this.enabled) {
      console.debug("Extension not enabled, skipping.");
      return emptyResponse;
    }

    const currentTimestamp = Date.now();
    this.latestTimestamp = currentTimestamp;

    await sleep(this.suggestionDelay);
    if (currentTimestamp < this.latestTimestamp) {
      return emptyResponse;
    }

    const replaceRange = this.calculateReplaceRange(document, position);

    if (this.pendingCompletion) {
      this.pendingCompletion.cancel();
    }

    const request = {
      filepath: document.uri.fsPath,
      language: document.languageId,  // https://code.visualstudio.com/docs/languages/identifiers
      text: document.getText(),
      position: document.offsetAt(position),
      maxPrefixLines: this.maxPrefixLines,
      maxSuffixLines: this.maxSuffixLines,
    };
    this.pendingCompletion = agent().getCompletions(request);

    const completion = await this.pendingCompletion.catch((_: Error) => {
      return null;
    });
    this.pendingCompletion = null;

    const completions = this.toInlineCompletions(completion, replaceRange);
    return Promise.resolve(completions);
  }

  private updateConfiguration() {
    const configuration = workspace.getConfiguration("tabby");
    this.enabled = configuration.get("codeCompletion", true);
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
