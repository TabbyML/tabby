import {
  InlineCompletionItemProvider,
  InlineCompletionContext,
  InlineCompletionTriggerKind,
  InlineCompletionItem,
  TextDocument,
  Position,
  CancellationToken,
  SnippetString,
  Range,
  window,
} from "vscode";
import { InlineCompletionParams } from "vscode-languageclient";
import { InlineCompletionRequest, InlineCompletionList, EventParams } from "tabby-agent";
import { EventEmitter } from "events";
import { getLogger } from "./logger";
import { Client } from "./lsp/client";
import { Config } from "./Config";

interface DisplayedCompletion {
  id: string;
  displayedAt: number;
  completions: InlineCompletionList;
  index: number;
}

export class InlineCompletionProvider extends EventEmitter implements InlineCompletionItemProvider {
  private readonly logger = getLogger();
  private displayedCompletion: DisplayedCompletion | null = null;
  private ongoing: Promise<InlineCompletionList | null> | null = null;
  private triggerMode: "automatic" | "manual";

  constructor(
    private readonly client: Client,
    private readonly config: Config,
  ) {
    super();
    this.triggerMode = this.config.inlineCompletionTriggerMode;
    this.config.on("updated", () => {
      this.triggerMode = this.config.inlineCompletionTriggerMode;
    });
  }

  get isLoading(): boolean {
    return this.ongoing !== null;
  }

  async provideInlineCompletionItems(
    document: TextDocument,
    position: Position,
    context: InlineCompletionContext,
    token: CancellationToken,
  ): Promise<InlineCompletionItem[] | null> {
    this.logger.debug("Function provideInlineCompletionItems called.");

    if (this.displayedCompletion) {
      // auto dismiss by new completion
      this.handleEvent("dismiss");
    }

    if (context.triggerKind === InlineCompletionTriggerKind.Automatic && this.triggerMode === "manual") {
      this.logger.debug("Skip automatic trigger when triggerMode is manual.");
      return null;
    }

    // Skip when trigger automatically and text selected
    if (
      context.triggerKind === InlineCompletionTriggerKind.Automatic &&
      window.activeTextEditor &&
      !window.activeTextEditor.selection.isEmpty
    ) {
      this.logger.debug("Text selected, skipping.");
      return null;
    }

    if (token.isCancellationRequested) {
      this.logger.debug("Completion request is canceled before send request.");
      return null;
    }

    const params: InlineCompletionParams = {
      context,
      textDocument: {
        uri: document.uri.toString(),
      },
      position: {
        line: position.line,
        character: position.character,
      },
    };
    let request: Promise<InlineCompletionList | null> | undefined = undefined;
    try {
      this.client.fileTrack.addingChangeEditor(window.activeTextEditor);
      request = this.client.languageClient.sendRequest(InlineCompletionRequest.method, params, token);
      this.ongoing = request;
      this.emit("didChangeLoading", true);
      const result = await this.ongoing;
      this.ongoing = null;
      this.emit("didChangeLoading", false);

      if (token.isCancellationRequested) {
        return null;
      }
      if (!result || result.items.length === 0) {
        return null;
      }

      this.handleEvent("show", result);

      return result.items.map((item, index) => {
        return new InlineCompletionItem(
          typeof item.insertText === "string" ? item.insertText : new SnippetString(item.insertText.value),
          item.range
            ? new Range(
                item.range.start.line,
                item.range.start.character,
                item.range.end.line,
                item.range.end.character,
              )
            : undefined,
          {
            title: "",
            command: "tabby.applyCallback",
            arguments: [
              () => {
                this.handleEvent("accept", result, index);
              },
            ],
          },
        );
      });
    } catch (error) {
      if (this.ongoing === request) {
        // the request was not replaced by a new request
        this.ongoing = null;
        this.emit("didChangeLoading", false);
      }
      return null;
    }
  }

  // FIXME: We don't listen to the user cycling through the items,
  // so we don't know the 'index' (except for the 'accept' event).
  // For now, just use the first item to report other events.
  async handleEvent(
    event: "show" | "accept" | "dismiss" | "accept_word" | "accept_line",
    completions?: InlineCompletionList,
    index = 0,
  ) {
    if (event === "show" && completions) {
      const item = completions.items[index];
      const cmplId = item?.data?.eventId?.completionId.replace("cmpl-", "");
      const timestamp = Date.now();
      this.displayedCompletion = {
        id: `view-${cmplId}-at-${timestamp}`,
        completions,
        index,
        displayedAt: timestamp,
      };
      await this.postEvent(event, this.displayedCompletion);
    } else if (this.displayedCompletion) {
      this.displayedCompletion.index = index;
      await this.postEvent(event, this.displayedCompletion);
      this.displayedCompletion = null;
    }
  }

  private async postEvent(
    event: "show" | "accept" | "dismiss" | "accept_word" | "accept_line",
    displayedCompletion: DisplayedCompletion,
  ) {
    const { id, completions, index, displayedAt } = displayedCompletion;
    const eventId = completions.items[index]?.data?.eventId;
    if (!eventId) {
      return;
    }
    const elapsed = Date.now() - displayedAt;
    let eventData: { type: "view" | "select" | "dismiss"; selectKind?: "line"; elapsed?: number };
    switch (event) {
      case "show":
        eventData = { type: "view" };
        break;
      case "accept":
        eventData = { type: "select", elapsed };
        break;
      case "dismiss":
        eventData = { type: "dismiss", elapsed };
        break;
      case "accept_word":
        // select_kind should be "word" but not supported by Tabby Server yet, use "line" instead
        eventData = { type: "select", selectKind: "line", elapsed };
        break;
      case "accept_line":
        eventData = { type: "select", selectKind: "line", elapsed };
        break;
      default:
        // unknown event type, should be unreachable
        return;
    }
    const params: EventParams = {
      ...eventData,
      eventId,
      viewId: id,
    };
    // await not required
    this.client.telemetry.postEvent(params);
  }

  /**
   * Calculate the edited range in the modified document as if the current completion item has been accepted.
   * @return {Range | undefined} - The range with the current completion item
   */
  public calcEditedRangeAfterAccept(): Range | undefined {
    const item = this.displayedCompletion?.completions.items[this.displayedCompletion.index];
    const range = item?.range;
    if (!range) {
      // FIXME: If the item has a null range, we can use current position and text length to calculate the result range
      return undefined;
    }
    if (!item) {
      return undefined;
    }
    const length = (item.insertText as string).split("\n").length - 1; //remove current line count;
    const completionRange = new Range(
      new Position(range.start.line, range.start.character),
      new Position(range.end.line + length + 1, 0),
    );
    this.logger.debug("Calculate edited range for displayed completion item:", completionRange);
    return completionRange;
  }
}
