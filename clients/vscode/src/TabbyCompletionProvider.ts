import {
  CancellationToken,
  InlineCompletionContext,
  InlineCompletionItem,
  InlineCompletionItemProvider,
  InlineCompletionTriggerKind,
  Position,
  Range,
  LocationLink,
  TextDocument,
  NotebookDocument,
  NotebookRange,
  Uri,
  commands,
  extensions,
  window,
  workspace,
} from "vscode";
import { EventEmitter } from "events";
import { CompletionRequest, CompletionResponse, LogEventRequest } from "tabby-agent";
import { API as GitAPI } from "./types/git";
import { logger } from "./logger";
import { agent } from "./agent";
import { getWordStartIndices } from "./utils";

type DisplayedCompletion = {
  id: string;
  completion: CompletionResponse;
  displayedAt: number;
};

export class TabbyCompletionProvider extends EventEmitter implements InlineCompletionItemProvider {
  private readonly logger = logger();
  private triggerMode: "automatic" | "manual" | "disabled" = "automatic";
  private onGoingRequestAbortController: AbortController | null = null;
  private loading: boolean = false;
  private displayedCompletion: DisplayedCompletion | null = null;
  private gitApi: GitAPI | null = null;

  public constructor() {
    super();
    this.updateConfiguration();
    workspace.onDidChangeConfiguration((event) => {
      if (event.affectsConfiguration("tabby") || event.affectsConfiguration("editor.inlineSuggest")) {
        this.updateConfiguration();
      }
    });

    const gitExt = extensions.getExtension("vscode.git");
    if (gitExt && gitExt.isActive) {
      this.gitApi = gitExt.exports.getAPI(1); // version: 1
    }
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
    this.logger.debug("Call provideInlineCompletionItems.");

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

    // Check if autocomplete widget is visible
    if (context.selectedCompletionInfo !== undefined) {
      this.logger.debug("Autocomplete widget is visible, skipping.");
      return null;
    }

    if (token?.isCancellationRequested) {
      this.logger.debug("Completion request is canceled before agent request.");
      return null;
    }

    const additionalContext = this.buildAdditionalContext(document);

    const request: CompletionRequest = {
      filepath: document.uri.toString(),
      language: document.languageId, // https://code.visualstudio.com/docs/languages/identifiers
      text: additionalContext.prefix + document.getText() + additionalContext.suffix,
      position: additionalContext.prefix.length + document.offsetAt(position),
      indentation: this.getEditorIndentation(),
      manually: context.triggerKind === InlineCompletionTriggerKind.Invoke,
      workspace: workspace.getWorkspaceFolder(document.uri)?.uri.toString(),
      git: this.getGitContext(document.uri),
      declarations: await this.collectDeclarationSnippets(document, position),
    };

    const abortController = new AbortController();
    this.onGoingRequestAbortController = abortController;
    token?.onCancellationRequested(() => {
      this.logger.debug("Completion request is canceled.");
      abortController.abort();
    });

    try {
      this.loading = true;
      this.emit("loadingStatusUpdated");
      const result = await agent().provideCompletions(request, { signal: abortController.signal });
      this.loading = false;
      this.emit("loadingStatusUpdated");

      if (token?.isCancellationRequested) {
        this.logger.debug("Completion request is canceled after agent request.");
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
              command: "tabby.applyCallback",
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
        this.logger.error("Error when providing completions", { error });
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
      this.logger.debug(`Post event ${event}`, { postBody });
      agent().postEvent(postBody);
    } catch (error: any) {
      this.logger.error("Error when posting event", { error });
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
      this.triggerMode = workspace.getConfiguration("tabby").get("inlineCompletion.triggerMode", "automatic");
      this.emit("triggerModeUpdated");
    }
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

  private getGitContext(uri: Uri): CompletionRequest["git"] | undefined {
    this.logger.trace("Begin getGitContext", { uri });
    const repo = this.gitApi?.getRepository(uri);
    if (!repo) {
      this.logger.trace("End getGitContext, no repo available.");
      return undefined;
    }
    const context = {
      root: repo.rootUri.toString(),
      remotes: repo.state.remotes
        .map((remote) => ({
          name: remote.name,
          url: remote.fetchUrl ?? remote.pushUrl ?? "",
        }))
        .filter((remote) => {
          return remote.url.length > 0;
        }),
    };
    this.logger.trace("End getGitContext", { context });
    return context;
  }

  private async collectDeclarationSnippets(
    document: TextDocument,
    position: Position,
  ): Promise<{ filepath: string; text: string }[] | undefined> {
    const config = agent().getConfig().completion.prompt;
    if (!config.fillDeclarations.enabled) {
      return undefined;
    }
    this.logger.trace("Begin collectDeclarationSnippets", { document, position });
    const snippets: { filepath: string; text: string }[] = [];
    const snippetLocations: LocationLink[] = [];
    // Find symbol positions in the previous lines
    const prefixRange = new Range(
      Math.max(0, position.line - config.maxPrefixLines + 1),
      0,
      position.line,
      position.character,
    );
    const prefixText = document.getText(prefixRange);
    const prefixRangeStartOffset = document.offsetAt(prefixRange.start);
    const symbolPositions = getWordStartIndices(prefixText).map((offset) =>
      document.positionAt(prefixRangeStartOffset + offset),
    );
    this.logger.trace("Found symbol positions in prefix text", { prefixText, symbolPositions });
    // Loop through the symbol positions backwards
    for (let symbolIndex = symbolPositions.length - 1; symbolIndex >= 0; symbolIndex--) {
      if (snippets.length >= config.fillDeclarations.maxSnippets) {
        // Stop collecting snippets if the max number of snippets is reached
        break;
      }
      const symbolPosition = symbolPositions[symbolIndex];
      const declarationLinks = await commands.executeCommand(
        "vscode.executeDefinitionProvider",
        document.uri,
        symbolPosition,
      );
      if (
        Array.isArray(declarationLinks) &&
        declarationLinks.length > 0 &&
        "targetUri" in declarationLinks[0] &&
        "targetRange" in declarationLinks[0]
      ) {
        const declarationLink = declarationLinks[0] as LocationLink;
        this.logger.trace("Processing declaration link", { declarationLink });
        if (
          declarationLink.targetUri.toString() == document.uri.toString() &&
          prefixRange.contains(declarationLink.targetRange.start)
        ) {
          // this symbol's declaration is already contained in the prefix range
          // this also includes the case of the symbol's declaration is at this position itself
          this.logger.trace("Skipping snippet as it is contained in the prefix");
          continue;
        }
        if (
          snippetLocations.find(
            (link) =>
              link.targetUri.toString() == declarationLink.targetUri.toString() &&
              link.targetRange.isEqual(declarationLink.targetRange),
          )
        ) {
          this.logger.trace("Skipping snippet as it is already collected");
          continue;
        }
        let text = new TextDecoder()
          .decode(await workspace.fs.readFile(declarationLink.targetUri))
          .split("\n")
          .slice(declarationLink.targetRange.start.line, declarationLink.targetRange.end.line + 1)
          .join("\n");
        if (text.length > config.fillDeclarations.maxCharsPerSnippet) {
          // crop the text to fit within the chars limit
          text = text.slice(0, config.fillDeclarations.maxCharsPerSnippet);
          text = text.slice(0, text.lastIndexOf("\n") + 1);
        }
        if (text.length > 0) {
          this.logger.trace("Collected declaration snippet", { text });
          snippets.push({ filepath: declarationLink.targetUri.toString(), text });
          snippetLocations.push(declarationLink);
        }
      }
    }
    this.logger.trace("End collectDeclarationSnippets", { snippets });
    return snippets;
  }
}
