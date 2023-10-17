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
import { EventEmitter } from "events";
import { CompletionRequest, CompletionResponse, LogEventRequest } from "tabby-agent";
import { agent } from "./agent";

export class TabbyCompletionProvider extends EventEmitter implements InlineCompletionItemProvider {
  static instance: TabbyCompletionProvider;
  static getInstance(): TabbyCompletionProvider {
    if (!TabbyCompletionProvider.instance) {
      TabbyCompletionProvider.instance = new TabbyCompletionProvider();
    }
    return TabbyCompletionProvider.instance;
  }

  private triggerMode: "automatic" | "manual" | "disabled" = "automatic";
  private onGoingRequestAbortController: AbortController | null = null;
  private loading: boolean = false;
  private latestCompletions: CompletionResponse | null = null;

  private constructor() {
    super();
    this.updateConfiguration();
    workspace.onDidChangeConfiguration((event) => {
      if (event.affectsConfiguration("tabby") || event.affectsConfiguration("editor.inlineSuggest")) {
        this.updateConfiguration();
      }
    });
  }

  public getTriggerMode(): "automatic" | "manual" | "disabled" {
    return this.triggerMode;
  }

  public isLoading(): boolean {
    return this.loading;
  }

  public async provideInlineCompletionItems(
    document: TextDocument,
    position: Position,
    context: InlineCompletionContext,
    token: CancellationToken,
  ): Promise<InlineCompletionItem[] | null> {
    console.debug("Call provideInlineCompletionItems.");

    if (context.triggerKind === InlineCompletionTriggerKind.Automatic && this.triggerMode === "manual") {
      console.debug("Skip automatic trigger when triggerMode is manual.");
      return null;
    }

    // Check if autocomplete widget is visible
    if (context.selectedCompletionInfo !== undefined) {
      console.debug("Autocomplete widget is visible, skipping.");
      return null;
    }

    if (token?.isCancellationRequested) {
      console.debug("Completion request is canceled before agent request.");
      return null;
    }

    const request: CompletionRequest = {
      filepath: document.uri.fsPath,
      language: document.languageId, // https://code.visualstudio.com/docs/languages/identifiers
      text: document.getText(),
      position: document.offsetAt(position),
      manually: context.triggerKind === InlineCompletionTriggerKind.Invoke,
    };

    this.latestCompletions = null;

    const abortController = new AbortController();
    this.onGoingRequestAbortController = abortController;
    token?.onCancellationRequested(() => {
      console.debug("Completion request is canceled.");
      abortController.abort();
    });

    try {
      this.loading = true;
      this.emit("loadingStatusUpdated");
      const result = await agent().provideCompletions(request, { signal: abortController.signal });
      this.loading = false;
      this.emit("loadingStatusUpdated");

      if (token?.isCancellationRequested) {
        console.debug("Completion request is canceled after agent request.");
        return null;
      }

      // Assume only one choice is provided, do not support multiple choices for now
      if (result.choices.length > 0) {
        this.latestCompletions = result;
        const choice = result.choices[0];

        this.postEvent("show");

        return [
          new InlineCompletionItem(
            choice.text,
            new Range(document.positionAt(choice.replaceRange.start), document.positionAt(choice.replaceRange.end)),
            {
              title: "",
              command: "tabby.applyCallback",
              arguments: [
                () => {
                  this.postEvent("accept");
                },
              ],
            },
          ),
        ];
      }
    } catch (error: any) {
      if (this.onGoingRequestAbortController === abortController) {
        // the request was not replaced by a new request, set loading to false safely
        this.loading = false;
        this.emit("loadingStatusUpdated");
      }
      if (error.name !== "AbortError") {
        console.debug("Error when providing completions", { error });
      }
    }

    return null;
  }

  public postEvent(event: "show" | "accept" | "accept_word" | "accept_line") {
    const completion = this.latestCompletions;
    if (completion && completion.choices.length > 0) {
      let postBody: LogEventRequest = {
        type: event === "show" ? "view" : "select",
        completion_id: completion.id,
        // Assume only one choice is provided for now
        choice_index: completion.choices[0].index,
      };
      switch (event) {
        case "accept_word":
          // select_kind should be "word" but not supported by Tabby Server yet, use "line" instead
          postBody = { ...postBody, select_kind: "line" };
          break;
        case "accept_line":
          postBody = { ...postBody, select_kind: "line" };
          break;
        default:
          break;
      }
      console.debug(`Post event ${event}`, { postBody });
      try {
        agent().postEvent(postBody);
      } catch (error: any) {
        console.debug("Error when posting event", { error });
      }
    }
  }

  private updateConfiguration() {
    if (!workspace.getConfiguration("editor").get("inlineSuggest.enabled", true)) {
      this.triggerMode = "disabled";
      this.emit("triggerModeUpdated");
    } else {
      this.triggerMode = workspace.getConfiguration("tabby").get("inlineCompletion.triggerMode", "automatic");
      this.emit("triggerModeUpdated");
    }
  }
}
