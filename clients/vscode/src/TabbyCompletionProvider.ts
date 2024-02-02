import { EventEmitter } from "events";
import { normalize } from "path";
import { CompletionRequest, CompletionResponse, LogEventRequest } from "tabby-agent";
import {
  CancellationToken,
  InlineCompletionContext,
  InlineCompletionItem,
  InlineCompletionItemProvider,
  InlineCompletionTriggerKind,
  NotebookDocument,
  NotebookRange,
  Position,
  Range,
  TextDocument,
  window,
  workspace,
  ExtensionContext,
} from "vscode";
import { agent } from "./agent";
import { ContextMixer } from "./completions/context/context-mixer";
import { ContextStrategy, DefaultContextStrategyFactory } from "./completions/context/context-strategy";
import { getCurrentDocContext } from "./completions/get-current-doc-context";

const getContextRetrieverStrategy = (): ContextStrategy =>
  workspace.getConfiguration("rumicode").get("inlineCompletion.contextRetriever", "none");

const createContextMixer = (context: ExtensionContext) =>
  new ContextMixer(new DefaultContextStrategyFactory(getContextRetrieverStrategy(), context));

type DisplayedCompletion = {
  id: string;
  completion: CompletionResponse;
  displayedAt: number;
};

export class TabbyCompletionProvider extends EventEmitter implements InlineCompletionItemProvider {
  #context: ExtensionContext;
  private triggerMode: "automatic" | "manual" | "disabled" = "automatic";
  private cursorContextLimits: {
    maxPrefixLines: number;
    maxSuffixLines: number;
  } = {
    maxPrefixLines: 20,
    maxSuffixLines: 20,
  };
  private onGoingRequestAbortController: AbortController | null = null;
  private loading: boolean = false;
  private latestCompletions: CompletionResponse | null = null;
  private contextMixer?: ContextMixer;
  private displayedCompletion: DisplayedCompletion | null = null;

  public constructor(context: ExtensionContext) {
    super();
    this.#context = context;
    this.updateConfiguration();
    workspace.onDidChangeConfiguration((event) => {
      if (event.affectsConfiguration("rumicode") || event.affectsConfiguration("editor.inlineSuggest")) {
        this.updateConfiguration();
      }
    });
  }

  public dispose(): void {
    this.contextMixer?.dispose();
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

    if (this.displayedCompletion) {
      // auto dismiss by new completion
      this.handleEvent("dismiss");
    }

    if (context.triggerKind === InlineCompletionTriggerKind.Automatic && this.triggerMode === "manual") {
      console.debug("Skip automatic trigger when triggerMode is manual.");
      return null;
    }

    // Skip when trigger automatically and text selected
    if (
      context.triggerKind === InlineCompletionTriggerKind.Automatic &&
      window.activeTextEditor &&
      !window.activeTextEditor.selection.isEmpty
    ) {
      console.debug("Text selected, skipping.");
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

    const additionalContext = this.buildAdditionalContext(document);

    const abortController = new AbortController();

    const docContext = getCurrentDocContext({
      document,
      position,
      maxPrefixLength: 1024,
      maxSuffixLength: 128,
      // We ignore the current context selection if completeSuggestWidgetSelection is not enabled
      context: undefined,
      dynamicMultilineCompletions: true,
    });

    const { context: snippets } = this.contextMixer
      ? await this.contextMixer.getContext({
          document,
          position,
          docContext,
          abortSignal: abortController.signal,
          maxChars: 1024,
        })
      : { context: [] };

    const request: CompletionRequest = {
      path: normalize(workspace.asRelativePath(document.uri.fsPath)),
      filepath: document.uri.fsPath,
      language: document.languageId, // https://code.visualstudio.com/docs/languages/identifiers
      text: additionalContext.prefix + document.getText() + additionalContext.suffix,
      position: additionalContext.prefix.length + document.offsetAt(position),
      indentation: this.getEditorIndentation(),
      maxPrefixLines: this.cursorContextLimits.maxPrefixLines,
      maxSuffixLines: this.cursorContextLimits.maxSuffixLines,
      manually: context.triggerKind === InlineCompletionTriggerKind.Invoke,
      snippets: snippets.map((snippet) => ({
        content: snippet.content,
        file_name: snippet.fileName,
      })),
    };

    this.latestCompletions = null;

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
        const choice = result.choices[0]!;
        this.handleEvent("show", result);

        return [
          new InlineCompletionItem(
            choice.text,
            new Range(
              document.positionAt(choice.replaceRange.start - additionalContext.prefix.length),
              document.positionAt(choice.replaceRange.end - additionalContext.prefix.length),
            ),
            {
              title: "",
              command: "rumicode.applyCallback",
              arguments: [
                () => {
                  this.handleEvent("accept");
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

  public handleEvent(
    event: "show" | "accept" | "dismiss" | "accept_word" | "accept_line",
    completion?: CompletionResponse,
  ) {
    if (event === "show" && completion) {
      const cmplId = completion.id.replace("cmpl-", "");
      const timestamp = Date.now();
      this.displayedCompletion = {
        id: `view-${cmplId}-at-${timestamp}`,
        completion,
        displayedAt: timestamp,
      };
      this.postEvent(event, this.displayedCompletion);
    } else if (this.displayedCompletion) {
      this.postEvent(event, this.displayedCompletion);
      this.displayedCompletion = null;
    }
  }

  private postEvent(
    event: "show" | "accept" | "dismiss" | "accept_word" | "accept_line",
    displayedCompletion: DisplayedCompletion,
  ) {
    const { id, completion, displayedAt } = displayedCompletion;
    const elapsed = Date.now() - displayedAt;
    let eventData: { type: string; select_kind?: "line"; elapsed?: number };
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
        eventData = { type: "select", select_kind: "line", elapsed };
        break;
      case "accept_line":
        eventData = { type: "select", select_kind: "line", elapsed };
        break;
      default:
        // unknown event type, should be unreachable
        return;
    }
    try {
      const postBody: LogEventRequest = {
        ...eventData,
        completion_id: completion.id,
        // Assume only one choice is provided for now
        choice_index: completion.choices[0]!.index,
        view_id: id,
      };
      console.debug(`Post event ${event}`, { postBody });
      agent().postEvent(postBody);
    } catch (error: any) {
      console.debug("Error when posting event", { error });
    }
  }

  private getEditorIndentation(): string | undefined {
    const editor = window.activeTextEditor;
    if (!editor) {
      return undefined;
    }

    const { insertSpaces, tabSize } = editor.options;
    if (insertSpaces && typeof tabSize === "number" && tabSize > 0) {
      return " ".repeat(tabSize);
    } else if (!insertSpaces) {
      return "\t";
    }
    return undefined;
  }

  private updateConfiguration() {
    if (!workspace.getConfiguration("editor").get("inlineSuggest.enabled", true)) {
      this.triggerMode = "disabled";
      this.emit("triggerModeUpdated");
    } else {
      const configuration = workspace.getConfiguration("rumicode");

      this.triggerMode = configuration.get("inlineCompletion.triggerMode", "automatic");
      this.emit("triggerModeUpdated");

      const cursorContextMode = configuration.get<"small" | "medium" | "large">(
        "inlineCompletion.cursorContext",
        "medium",
      );

      switch (cursorContextMode) {
        case "small":
          this.cursorContextLimits.maxPrefixLines = 20;
          this.cursorContextLimits.maxSuffixLines = 10;
          break;
        case "medium":
          this.cursorContextLimits.maxPrefixLines = 30;
          this.cursorContextLimits.maxSuffixLines = 20;
          break;
        case "large":
          this.cursorContextLimits.maxPrefixLines = 50;
          this.cursorContextLimits.maxSuffixLines = 20;
      }
    }

    this.contextMixer?.dispose();
    this.contextMixer = createContextMixer(this.#context);
  }

  private buildAdditionalContext(document: TextDocument): { prefix: string; suffix: string } {
    if (
      document.uri.scheme === "vscode-notebook-cell" &&
      window.activeNotebookEditor?.notebook.uri.path === document.uri.path
    ) {
      // Add all the cells in the notebook as context
      const notebook = window.activeNotebookEditor.notebook;
      const current = window.activeNotebookEditor.selection.start;
      const prefix = this.buildNotebookContext(notebook, new NotebookRange(0, current), document.languageId) + "\n\n";
      const suffix =
        "\n\n" +
        this.buildNotebookContext(notebook, new NotebookRange(current + 1, notebook.cellCount), document.languageId);
      return { prefix, suffix };
    }
    return { prefix: "", suffix: "" };
  }

  private notebookLanguageComments: { [languageId: string]: (code: string) => string } = {
    markdown: (code) => "```\n" + code + "\n```",
    python: (code) =>
      code
        .split("\n")
        .map((l) => "# " + l)
        .join("\n"),
  };

  private buildNotebookContext(notebook: NotebookDocument, range: NotebookRange, languageId: string): string {
    return notebook
      .getCells(range)
      .map((cell) => {
        if (cell.document.languageId === languageId) {
          return cell.document.getText();
        } else if (Object.keys(this.notebookLanguageComments).includes(languageId)) {
          return this.notebookLanguageComments[languageId]!(cell.document.getText());
        } else {
          return "";
        }
      })
      .join("\n\n");
  }
}
