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
import { Language as SupportedLanguage, CompletionResponse, EventType, ChoiceEvent, ApiError } from "tabby-client";
import { Tabby } from "./Tabby";
import { sleep } from "./utils";

export class TabbyCompletionProvider implements InlineCompletionItemProvider {
  private uuid = Date.now();
  private latestTimestamp: number = 0;

  // User Settings
  private enabled: boolean = true;

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

    const languageId = document.languageId;
    let language = SupportedLanguage.UNKNOWN;
    if (Object.values(SupportedLanguage).map(x => x.toString()).includes(languageId)) {
      language = languageId as SupportedLanguage;
    }
    if (language == SupportedLanguage.UNKNOWN) {
      console.debug(`Unsupported language '${languageId}', skipping`);
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
    await sleep(suggestionDelay);
    if (currentTimestamp < this.latestTimestamp) {
      return emptyResponse;
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

    Tabby.getInstance().healthCheck();
    const completion = await Tabby.getInstance().api.default.completionsV1CompletionsPost({
      prompt: prompt as string, // Prompt is already nil-checked
      language
    }).catch((err: ApiError) => {
      console.error(err);
      Tabby.getInstance().healthCheck(true);
      return null;
    });

    const hasSuffixParen = this.hasSuffixParen(document, position);
    const replaceRange = hasSuffixParen
      ? new Range(
          position.line,
          position.character,
          position.line,
          position.character + 1
        )
      : new Range(position, position);
    const completions = this.toInlineCompletions(completion, replaceRange);
    console.debug("Result completions: ", completions);
    return Promise.resolve(completions);
  }

  private updateConfiguration() {
    const configuration = workspace.getConfiguration("tabby");
    this.enabled = configuration.get("enabled", true);
  }

  private getPrompt(document: TextDocument, position: Position): string | undefined {
    const maxLines = 20;
    const firstLine = Math.max(position.line - maxLines, 0);

    return document.getText(new Range(firstLine, 0, position.line, position.character));
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
}
