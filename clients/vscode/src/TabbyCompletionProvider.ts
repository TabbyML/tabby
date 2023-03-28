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
import axios, { AxiosResponse } from "axios";
import { EventType } from "./EventHandler";

export class TabbyCompletionProvider implements InlineCompletionItemProvider {
  private uuid = Date.now();
  private latestTimestamp: number = 0;

  // User Settings
  private enabled: boolean = true;
  private tabbyServerUrl: string = "";

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

    const prompt = this.getPrompt(document, position);
    if (this.isNil(prompt)) {
      console.debug("Prompt is empty, skipping");
      return emptyResponse;
    }

    const currentTimestamp = Date.now();
    this.latestTimestamp = currentTimestamp;

    const suggestionDelay = 150;
    await this.sleep(suggestionDelay);
    if (currentTimestamp < this.latestTimestamp) {
      return emptyResponse;
    }

    console.debug(
      "Requesting: ",
      {
        uuid: this.uuid,
        timestamp: currentTimestamp,
        prompt
      }
    );
    // Prompt is already nil-checked
    const response = await this.getCompletions(prompt as String);

    const hasSuffixParen = this.hasSuffixParen(document, position);
    const replaceRange = hasSuffixParen
      ? new Range(
          position.line,
          position.character,
          position.line,
          position.character + 1
        )
      : new Range(position, position);
    const completions = this.toInlineCompletions(response.data, replaceRange);
    console.debug("Result completions: ", completions);
    return Promise.resolve(completions);
  }

  private updateConfiguration() {
    const configuration = workspace.getConfiguration("tabby");
    this.enabled = configuration.get("enabled", true);
    this.tabbyServerUrl = configuration.get("serverUrl", "http://127.0.0.1:5000");
  }

  private getPrompt(document: TextDocument, position: Position): String | undefined {
    const maxLines = 20;
    const firstLine = Math.max(position.line - maxLines, 0);

    return document.getText(new Range(firstLine, 0, position.line, position.character));
  }

  private isNil(value: String | undefined | null): boolean {
    return value === undefined || value === null || value.length === 0;
  }

  private sleep(milliseconds: number) {
    return new Promise((r) => setTimeout(r, milliseconds));
  }

  private toInlineCompletions(value: any, range: Range): InlineCompletionItem[] {
    return (
      value.choices?.map(
        (choice: any) =>
          new InlineCompletionItem(choice.text, range, {
            title: "Tabby: Emit Event",
            command: "tabby.emitEvent",
            arguments: [
              {
                type: EventType.InlineCompletionAccepted,
                id: value.id,
                index: choice.index,
              },
            ],
          })
      ) || []
    );
  }

  private getCompletions(prompt: String): Promise<AxiosResponse<any, any>> {
    return axios.post(`${this.tabbyServerUrl}/v1/completions`, {
      prompt,
    });
  }

  private hasSuffixParen(document: TextDocument, position: Position) {
    const suffix = document.getText(
      new Range(position.line, position.character, position.line, position.character + 1)
    );
    return ")]}".indexOf(suffix) > -1;
  }
}
