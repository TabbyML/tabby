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
import { CompletionResponse, EventType, ChoiceEvent, ApiError, CancelablePromise, CancelError } from "./generated";
import { TabbyClient } from "./TabbyClient";
import { CompletionCache } from "./CompletionCache";
import { sleep } from "./utils";

export class TabbyCompletionProvider implements InlineCompletionItemProvider {
  private uuid = Date.now();
  private latestTimestamp: number = 0;
  private pendingCompletion: CancelablePromise<CompletionResponse> | null = null;

  private tabbyClient = TabbyClient.getInstance();
  private completionCache = new CompletionCache();
  // User Settings
  private enabled: boolean = true;
  private suggestionDelay: number = 150;

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

    const promptRange = this.calculatePromptRange(position);
    const prompt = document.getText(promptRange);
    if (this.isNil(prompt)) {
      console.debug("Prompt is empty, skipping");
      return emptyResponse;
    }

    const currentTimestamp = Date.now();
    this.latestTimestamp = currentTimestamp;

    await sleep(this.suggestionDelay);
    if (currentTimestamp < this.latestTimestamp) {
      return emptyResponse;
    }

    const replaceRange = this.calculateReplaceRange(document, position);

    const compatibleCache = this.completionCache.findCompatible(document.uri, document.getText(), document.offsetAt(position));
    if (compatibleCache) {
      const completions = this.toInlineCompletions(compatibleCache, replaceRange);
      console.debug("Use cached completions: ", compatibleCache);
      return Promise.resolve(completions);
    }

    console.debug(
      "Requesting: ",
      {
        uuid: this.uuid,
        timestamp: currentTimestamp,
        prompt,
        language: document.languageId
      }
    );

    if (this.pendingCompletion) {
      this.pendingCompletion.cancel();
    }
    this.pendingCompletion = this.tabbyClient.api.default.completionsV1CompletionsPost({
      prompt: prompt as string,   // Prompt is already nil-checked
      language: document.languageId,  // https://code.visualstudio.com/docs/languages/identifiers
    });

    const completion = await this.pendingCompletion.then((response: CompletionResponse) => {
      this.tabbyClient.changeStatus("ready");
      return response;
    }).catch((_: CancelError) => {
      return null;
    }).catch((err: ApiError) => {
      console.error(err);
      this.tabbyClient.changeStatus("disconnected");
      return null;
    });
    this.pendingCompletion = null;

    if (completion) {
      this.completionCache.add({
        documentId: document.uri,
        promptRange: { start: document.offsetAt(promptRange.start), end: document.offsetAt(promptRange.end) },
        prompt,
        completion,
      });
    }
    const completions = this.toInlineCompletions(completion, replaceRange);
    console.debug("Result completions: ", completions);
    return Promise.resolve(completions);
  }

  private updateConfiguration() {
    const configuration = workspace.getConfiguration("tabby");
    this.enabled = configuration.get("enabled", true);
    this.suggestionDelay = configuration.get("suggestionDelay", 150);
  }

  private isNil(value: string | undefined | null): boolean {
    return value === undefined || value === null || value.length === 0;
  }

  private toInlineCompletions(tabbyCompletion: CompletionResponse | null, range: Range): InlineCompletionItem[] {
    return (
      tabbyCompletion?.choices?.map((choice: any) => {
        let event: ChoiceEvent = {
          type: EventType.SELECT,
          completion_id: tabbyCompletion.id,
          choice_index: choice.index,
        };
        return new InlineCompletionItem(choice.text, range, {
          title: "Tabby: Emit Event",
          command: "tabby.emitEvent",
          arguments: [event],
        });
      }) || []
    );
  }

  private hasSuffixParen(document: TextDocument, position: Position) {
    const suffix = document.getText(
      new Range(position.line, position.character, position.line, position.character + 1)
    );
    return ")]}".indexOf(suffix) > -1;
  }

  private calculatePromptRange(position: Position): Range {
    const maxLines = 20;
    const firstLine = Math.max(position.line - maxLines, 0);
    return new Range(firstLine, 0, position.line, position.character);
  }

  private calculateReplaceRange(document: TextDocument, position: Position): Range {
    const hasSuffixParen = this.hasSuffixParen(document, position);
    if (hasSuffixParen) {
      return new Range(position.line, position.character, position.line, position.character + 1);
    } else {
      return new Range(position, position);
    }
  }
}
