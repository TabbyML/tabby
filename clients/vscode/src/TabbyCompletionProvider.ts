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

export class TabbyCompletionProvider extends EventEmitter implements InlineCompletionItemProvider {
  #context: ExtensionContext;
  private triggerMode: "automatic" | "manual" | "disabled" = "automatic";
  private onGoingRequestAbortController: AbortController | null = null;
  private loading: boolean = false;
  private latestCompletions: CompletionResponse | null = null;
  private contextMixer?: ContextMixer;

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

    const additionalContext = this.buildAdditionalContext(document);

    const abortController = new AbortController();

    const docContext = getCurrentDocContext({
      document,
      position,
      maxPrefixLength: 25,
      maxSuffixLength: 20,
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
        this.latestCompletions = result;
        const choice = result.choices[0]!;

        this.postEvent("show");

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
        choice_index: completion.choices[0]!.index,
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
      this.triggerMode = workspace.getConfiguration("rumicode").get("inlineCompletion.triggerMode", "automatic");
      this.emit("triggerModeUpdated");
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
      const prefix = this.buildNotebookContext(notebook, new NotebookRange(0, current), document.languageId);
      const suffix = this.buildNotebookContext(
        notebook,
        new NotebookRange(current + 1, notebook.cellCount),
        document.languageId,
      );
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
          return cell.document.getText() + "\n\n";
        } else if (Object.keys(this.notebookLanguageComments).includes(languageId)) {
          return this.notebookLanguageComments[languageId]!(cell.document.getText()) + "\n\n";
        } else {
          return "";
        }
      })
      .join("");
  }
}
