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
  TextEditorDecorationType,
  ThemeColor,
  MarkdownString,
  DecorationOptions,
  commands,
  Selection,
} from "vscode";
import { InlineCompletionRequest, InlineNESCompletionRequest, InlineCompletionList, EventParams } from "tabby-agent";
import { EventEmitter } from "events";
import { getLogger } from "./logger";
import { Client } from "./lsp/client";
import { Config } from "./Config";
import { InlineCompletionParams } from "vscode-languageclient";

// Create unique decoration type for NES completions
const NES_LABEL = "✨ next edit";

// NES decoration types
const nesDecorationTypes: Record<string, TextEditorDecorationType> = {};

// NES content highlight decoration type - slight background color change
const nesContentHighlightType = window.createTextEditorDecorationType({
  backgroundColor: new ThemeColor("editor.symbolHighlightBackground"),
  opacity: "0.3",
  borderRadius: "3px",
  isWholeLine: false,
});

// NES content replacement decoration type - strikethrough effect
const nesContentReplaceType = window.createTextEditorDecorationType({
  textDecoration: "line-through",
  color: new ThemeColor("editorError.foreground"),
  backgroundColor: new ThemeColor("diffEditor.removedTextBackground"),
  opacity: "0.3",
  isWholeLine: false,
});

// NES intelligent suggestion decoration type - deep green with high transparency
const nesIntelligentSuggestionType = window.createTextEditorDecorationType({
  backgroundColor: "#004400",
  opacity: "0.15",
  borderRadius: "3px",
  fontWeight: "normal",
  border: "1px dotted #006600",
  isWholeLine: false,
});

// NES exact match content decoration type
const nesMatchedContentType = window.createTextEditorDecorationType({
  backgroundColor: new ThemeColor("diffEditor.unchangedTextBackground"),
  border: "1px dotted " + new ThemeColor("editorHint.border"),
  borderRadius: "2px",
  opacity: "0.25",
  isWholeLine: false,
});

// Create different decoration types for different scenarios
// nest edit label color
function getNESDecorationType(id: string): TextEditorDecorationType {
  if (!nesDecorationTypes[id]) {
    // Create a new decoration type with more prominent markers
    nesDecorationTypes[id] = window.createTextEditorDecorationType({
      after: {
        contentText: ` ${NES_LABEL}`,
        color: new ThemeColor("editorInlayHint.foreground"),
        fontStyle: "italic",
        margin: "0 0 0 20px",
      },
      light: {
        after: {
          color: "#1b80b2",
        },
      },
      dark: {
        after: {
          color: "#69c0fa",
        },
      },
    });
  }

  // Return a valid decoration type, create a new empty one as fallback if not found
  return nesDecorationTypes[id] || window.createTextEditorDecorationType({});
}

// Clean up unused decoration types
function disposeNESDecorationType(id: string) {
  const decorationType = nesDecorationTypes[id];
  if (decorationType) {
    decorationType.dispose();
    // eslint-disable-next-line @typescript-eslint/no-dynamic-delete
    delete nesDecorationTypes[id];
  }
}

interface DisplayedCompletion {
  id: string;
  displayedAt: number;
  completions: InlineCompletionList;
  index: number;
  isNES?: boolean; // Marks if this is an NES completion
}

export class InlineCompletionProvider extends EventEmitter implements InlineCompletionItemProvider {
  private readonly logger = getLogger();
  private displayedCompletion: DisplayedCompletion | null = null;
  private ongoing: Promise<InlineCompletionList | null> | null = null;
  private triggerMode: "automatic" | "manual";
  private activeNESDecorations: Record<string, string[]> = {}; // Track active NES decoration IDs
  private temporaryNormalCompletion = false;
  private normalCompletionAbortController: AbortController | null = null;

  constructor(
    private readonly client: Client,
    private readonly config: Config,
  ) {
    super();
    this.triggerMode = this.config.inlineCompletionTriggerMode;
    this.config.on("updated", () => {
      this.triggerMode = this.config.inlineCompletionTriggerMode;
    });

    // Listen for editor changes to update decorations
    window.onDidChangeActiveTextEditor((editor) => {
      if (editor && this.displayedCompletion?.isNES) {
        // When editor changes, update NES decorations
        this.updateNESDecorations(editor.document.uri.toString());
      }
    });
  }

  /**
   * Analyze differences between current text and NES completion
   */
  private analyzeNESDifferences(
    document: TextDocument,
    position: Position,
    completionText: string,
  ): DecorationOptions[] {
    const decorations: DecorationOptions[] = [];
    const currentLineText = document.lineAt(position.line).text;
    const currentLineSuffix = currentLineText.substring(position.character);

    // Handle multi-line completions
    const completionLines = completionText.split("\n");

    if (completionLines.length === 1 && completionLines[0]) {
      // Single-line completion case
      const completionLine = completionLines[0];

      // Use character-level diff calculation for more detailed difference info
      const diff = this.calculateStringDiff(currentLineSuffix, completionLine);

      // Use matching content decoration for matched parts
      if (diff.matched.length > 0) {
        // Simplified handling: create separate decorations for each matching character
        for (const pos of diff.matched) {
          decorations.push({
            range: new Range(
              new Position(position.line, position.character + pos),
              new Position(position.line, position.character + pos + 1),
            ),
            hoverMessage: new MarkdownString("**Preserved Content**\n\nThis text will be kept"),
          });
        }

        // Handle newly added content
        for (const pos of diff.added) {
          decorations.push({
            range: new Range(
              new Position(position.line, position.character + pos),
              new Position(position.line, position.character + pos + 1),
            ),
            hoverMessage: new MarkdownString(
              "**Suggested New Content**\n\n✨ Tabby intelligent suggestion to add this content",
            ),
          });
        }

        // If there is content to be deleted, add the entire suffix as deletion
        if (diff.removed.length > 0 && currentLineSuffix.length > 0) {
          decorations.push({
            range: new Range(position, new Position(position.line, position.character + currentLineSuffix.length)),
            hoverMessage: new MarkdownString(
              "**Content To Be Replaced**\n\nThis content will be removed when accepting the suggestion",
            ),
          });
        }
      } else {
        // Completely different, mark all as replacement and new content
        if (completionLine.length > 0) {
          decorations.push({
            range: new Range(position, new Position(position.line, position.character + completionLine.length)),
            hoverMessage: new MarkdownString("**Suggested New Content**\n\n✨ Tabby intelligent complete suggestion"),
          });
        }

        if (currentLineSuffix.length > 0) {
          decorations.push({
            range: new Range(position, new Position(position.line, position.character + currentLineSuffix.length)),
            hoverMessage: new MarkdownString(
              "**Content To Be Replaced**\n\nThis content will be completely replaced when accepting the suggestion",
            ),
          });
        }
      }
    } else if (completionLines.length > 1) {
      // Multi-line completion case
      // First line
      if (currentLineSuffix.length > 0) {
        decorations.push({
          range: new Range(position, new Position(position.line, currentLineText.length)),
          hoverMessage: new MarkdownString("**Content To Be Replaced**\n\nThe remainder of this line will be replaced"),
        });
      }

      const firstLine = completionLines[0] || "";
      if (firstLine.length > 0) {
        decorations.push({
          range: new Range(position, new Position(position.line, position.character + firstLine.length)),
          hoverMessage: new MarkdownString(
            "**Suggested New Content (Line 1)**\n\n✨ Tabby intelligent suggestion for first line",
          ),
        });
      }

      // Middle lines
      for (let i = 1; i < completionLines.length; i++) {
        const lineContent = completionLines[i] || "";
        if (lineContent.length > 0) {
          decorations.push({
            range: new Range(new Position(position.line + i, 0), new Position(position.line + i, lineContent.length)),
            hoverMessage: new MarkdownString(
              `**Suggested New Content (Line ${i + 1})**\n\n✨ Tabby intelligent multi-line suggestion`,
            ),
          });
        }
      }
    }

    return decorations;
  }

  /**
   * Calculate character-level differences, returning deleted and added character indices
   * @param oldText Original text
   * @param newText New text
   * @returns Returns the changed character positions
   */
  private calculateStringDiff(
    oldText: string,
    newText: string,
  ): { matched: number[]; added: number[]; removed: number[] } {
    const result = {
      matched: [] as number[], // Matching character positions (in the new text)
      added: [] as number[], // Added character positions (in the new text)
      removed: [] as number[], // Removed character positions (in the original text)
    };

    // Calculate matching prefix from the text beginning
    let prefixLength = 0;
    const minLength = Math.min(oldText.length, newText.length);
    while (prefixLength < minLength && oldText[prefixLength] === newText[prefixLength]) {
      result.matched.push(prefixLength);
      prefixLength++;
    }

    // Calculate matching suffix from the text end
    let oldSuffixIndex = oldText.length - 1;
    let newSuffixIndex = newText.length - 1;
    while (
      oldSuffixIndex >= prefixLength &&
      newSuffixIndex >= prefixLength &&
      oldText[oldSuffixIndex] === newText[newSuffixIndex]
    ) {
      result.matched.unshift(newSuffixIndex);
      oldSuffixIndex--;
      newSuffixIndex--;
    }

    // Calculate added and removed characters
    for (let i = prefixLength; i <= newSuffixIndex; i++) {
      if (!result.matched.includes(i)) {
        result.added.push(i);
      }
    }

    for (let i = prefixLength; i <= oldSuffixIndex; i++) {
      result.removed.push(i);
    }

    return result;
  }

  /**
   * Update NES completion decorations
   */
  private updateNESDecorations(documentUri: string) {
    if (!this.displayedCompletion?.isNES) return;

    const editor = window.activeTextEditor;
    if (!editor || editor.document.uri.toString() !== documentUri) return;

    const item = this.displayedCompletion.completions.items[this.displayedCompletion.index];
    if (!item || !item.range) return;

    // Create decoration position
    const range = new Range(
      item.range.start.line,
      item.range.start.character,
      item.range.end.line,
      item.range.end.character,
    );

    // Get or create decoration type
    const decorationType = getNESDecorationType(this.displayedCompletion.id);

    // Apply label decoration
    editor.setDecorations(decorationType, [range]);

    // Apply content highlight decoration - using diff editor colors to highlight completion content
    editor.setDecorations(nesContentHighlightType, [range]);

    // Analyze and apply detailed differences
    const document = editor.document;

    // Analyze differences and apply specific decorations
    const diffDecorations = this.analyzeNESDifferences(
      document,
      item.range.start instanceof Position
        ? item.range.start
        : new Position(item.range.start.line, item.range.start.character),
      typeof item.insertText === "string" ? item.insertText : item.insertText.value,
    );

    const replaceDecorations = diffDecorations.filter((d) => {
      const msg = d.hoverMessage;
      return msg instanceof MarkdownString
        ? msg.value.includes("Content To Be Replaced")
        : String(msg).includes("Content To Be Replaced");
    });
    editor.setDecorations(nesContentReplaceType, replaceDecorations);

    const newContentDecorations = diffDecorations.filter((d) => {
      const msg = d.hoverMessage;
      return msg instanceof MarkdownString
        ? msg.value.includes("Suggested New Content")
        : String(msg).includes("Suggested New Content");
    });
    editor.setDecorations(nesIntelligentSuggestionType, newContentDecorations);

    const matchedDecorations = diffDecorations.filter((d) => {
      const msg = d.hoverMessage;
      return msg instanceof MarkdownString
        ? msg.value.includes("Preserved Content")
        : String(msg).includes("Preserved Content");
    });
    editor.setDecorations(nesMatchedContentType, matchedDecorations);

    if (!this.activeNESDecorations[documentUri]) {
      this.activeNESDecorations[documentUri] = [];
    }

    const decorations = this.activeNESDecorations[documentUri];
    if (decorations) {
      decorations.push(this.displayedCompletion.id);
    }
  }

  /**
   * Clean up decorations
   */
  private clearNESDecorations(documentUri?: string) {
    if (documentUri) {
      const decorationIds = this.activeNESDecorations[documentUri] || [];
      decorationIds.forEach((id) => disposeNESDecorationType(id));

      const editor = window.activeTextEditor;
      if (editor && editor.document.uri.toString() === documentUri) {
        // Clear all decoration types
        editor.setDecorations(nesContentHighlightType, []);
        editor.setDecorations(nesContentReplaceType, []);
        editor.setDecorations(nesIntelligentSuggestionType, []);
        editor.setDecorations(nesMatchedContentType, []);
      }

      if (documentUri in this.activeNESDecorations) {
        // eslint-disable-next-line @typescript-eslint/no-dynamic-delete
        delete this.activeNESDecorations[documentUri];
      }
    } else {
      const uris = Object.keys(this.activeNESDecorations);
      uris.forEach((uri) => {
        const decorationIds = this.activeNESDecorations[uri];
        if (decorationIds) {
          decorationIds.forEach((id) => disposeNESDecorationType(id));
        }

        window.visibleTextEditors.forEach((editor) => {
          if (editor.document.uri.toString() === uri) {
            // Clear all decoration types
            editor.setDecorations(nesContentHighlightType, []);
            editor.setDecorations(nesContentReplaceType, []);
            editor.setDecorations(nesIntelligentSuggestionType, []);
            editor.setDecorations(nesMatchedContentType, []);
          }
        });
      });
      this.activeNESDecorations = {};
    }
  }

  /**
   * Triggers a next edit suggestion by sending a request to the LSP server
   */
  public async provideManuallyTriggerNextEditSuggestionTest(): Promise<void> {
    this.logger.debug("Function provideNextEditSuggestion called.");

    const document = window.activeTextEditor?.document;
    const position = window.activeTextEditor?.selection.active;

    if (!document || !position) {
      this.logger.debug("No active document or position found.");
      return;
    }

    const params: InlineCompletionParams = {
      context: {
        triggerKind: InlineCompletionTriggerKind.Invoke,
        selectedCompletionInfo: undefined,
      },
      textDocument: {
        uri: document.uri.toString(),
      },
      position: {
        line: position.line,
        character: position.character,
      },
    };

    try {
      const request: Promise<InlineCompletionList | null> = this.client.languageClient.sendRequest(
        InlineCompletionRequest.method,
        params,
      );
      this.ongoing = request;
      this.emit("didChangeLoading", true);

      const result = await this.ongoing;
      this.ongoing = null;
      this.emit("didChangeLoading", false);

      if (!result || result.items.length === 0) {
        this.logger.debug("No next edit suggestions received.");
        return;
      }

      this.logger.debug("Next edit suggestion received:", result);

      window.showInformationMessage(`Next edit suggestion received with ${result.items.length} items`);

      this.logger.debug("Inline completions shown successfully.");
    } catch (error) {
      if (this.ongoing) {
        this.ongoing = null;
        this.emit("didChangeLoading", false);
      }
      this.logger.error("Error requesting next edit suggestion:", error);

      // Show error message to user for testing purposes
      window.showErrorMessage(`Next edit suggestion failed: ${error instanceof Error ? error.message : String(error)}`);
    }
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

      // Clean up decorations
      this.clearNESDecorations(document.uri.toString());

      // Update NES visibility
      this.updateNESVisibilityContext(false);
    }

    // Cancel any pending normal completion if we're displaying temporary ones
    if (this.temporaryNormalCompletion && this.normalCompletionAbortController) {
      this.logger.debug("Canceling temporary normal completion as new request is initiated");
      this.normalCompletionAbortController.abort();
      this.normalCompletionAbortController = null;
      this.temporaryNormalCompletion = false;
    }

    if (context.triggerKind === InlineCompletionTriggerKind.Automatic && this.triggerMode === "manual") {
      this.logger.debug("Skip automatic trigger when triggerMode is manual.");
      return null;
    }

    // Skip if the current language is disabled
    const currentLanguage = document.languageId;
    if (this.config.disabledLanguages.includes(currentLanguage)) {
      this.logger.debug(`Skipping completion for disabled language: ${currentLanguage}`);
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

    // Base request parameters
    const baseParams: InlineCompletionParams = {
      context,
      textDocument: {
        uri: document.uri.toString(),
      },
      position: {
        line: position.line,
        character: position.character,
      },
    };

    // NES specific parameters
    const nesParams: InlineCompletionParams = {
      context,
      textDocument: {
        uri: document.uri.toString(),
      },
      position: {
        line: position.line,
        character: position.character,
      },
    };

    try {
      this.client.fileTrack.addingChangeEditor(window.activeTextEditor);

      // Create an abort controller for normal completions
      this.normalCompletionAbortController = new AbortController();

      // Create both request promises
      const normalRequest: Promise<InlineCompletionList | null> = this.client.languageClient.sendRequest(
        InlineCompletionRequest.method,
        baseParams,
        token,
      );

      const nesRequest: Promise<InlineCompletionList | null> = this.client.languageClient.sendRequest(
        InlineNESCompletionRequest.method,
        nesParams,
        token,
      );

      // Setup loading indicator
      this.ongoing = Promise.race([normalRequest, nesRequest]);
      this.emit("didChangeLoading", true);

      // Create a flag to track if NES has completed
      let nesCompleted = false;

      // Use Promise.race to handle NES priority
      const firstResult = await Promise.race([
        nesRequest.then((result) => {
          nesCompleted = true;
          return { type: "nes", result };
        }),
        normalRequest.then((result) => {
          return { type: "normal", result };
        }),
      ]);

      // If NES completed first and has valid results
      if (
        firstResult.type === "nes" &&
        firstResult.result &&
        firstResult.result.items &&
        firstResult.result.items.length > 0
      ) {
        this.logger.info("NES completed first with valid results, using NES");

        // Update NES visibility to true
        this.updateNESVisibilityContext(true);

        // Cancel normal completion
        if (this.normalCompletionAbortController) {
          this.normalCompletionAbortController.abort();
          this.normalCompletionAbortController = null;
        }

        this.ongoing = null;
        this.emit("didChangeLoading", false);

        // Process and return NES completion
        return this.processNESCompletions(document, firstResult.result);
      }

      // If normal completed first, but we need to wait for NES
      if (firstResult.type === "normal" && !nesCompleted) {
        // Show normal completion temporarily while waiting for NES
        if (firstResult.result && firstResult.result.items && firstResult.result.items.length > 0) {
          this.logger.info("Normal completed first, showing temporary results while waiting for NES");
          this.temporaryNormalCompletion = true;

          // Start a new promise to wait for NES result
          nesRequest
            .then((nesResult) => {
              // If NES eventually returns valid results, we should replace normal completions
              if (nesResult && nesResult.items && nesResult.items.length > 0 && this.temporaryNormalCompletion) {
                this.logger.info("NES completed after normal, replacing with NES results");
                this.temporaryNormalCompletion = false;

                // Force a completion refresh to show NES results
                const editor = window.activeTextEditor;
                if (editor) {
                  // Use empty string to trigger a minimal change that forces refresh
                  editor.edit((editBuilder) => {
                    editBuilder.insert(editor.selection.active, "");
                  });
                }
              }
            })
            .catch((err) => {
              this.logger.error("Error waiting for NES after normal completion:", err);
              this.temporaryNormalCompletion = false;
            });

          // Show normal completion temporarily
          this.handleEvent("show", firstResult.result);

          // Setup normal completion items
          return firstResult.result.items.map((item, index) => {
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
                    this.handleEvent("accept", firstResult.result, index);
                    this.temporaryNormalCompletion = false;
                  },
                ],
              },
            );
          });
        }
      }

      // If both completed, wait for final results
      this.ongoing = null;
      this.emit("didChangeLoading", false);

      // Wait for both to complete
      const [normalResult, nesResult] = await Promise.all([
        nesCompleted
          ? Promise.resolve(firstResult.type === "nes" ? firstResult.result : null)
          : normalRequest.catch((err) => {
              this.logger.error("Error in normal completion request:", err);
              return null;
            }),
        nesCompleted
          ? Promise.resolve(firstResult.result)
          : nesRequest.catch((err) => {
              this.logger.error("Error in NES completion request:", err);
              return null;
            }),
      ]);

      // Always prioritize NES if it has results
      if (nesResult && nesResult.items && nesResult.items.length > 0) {
        this.logger.info("Using NES result for completion (final decision)");
        return this.processNESCompletions(document, nesResult);
      } else if (normalResult && normalResult.items && normalResult.items.length > 0) {
        this.handleEvent("show", normalResult);

        return normalResult.items.map((item, index) => {
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
                  this.handleEvent("accept", normalResult, index);
                },
              ],
            },
          );
        });
      }

      // No results from either source
      return null;
    } catch (error) {
      if (this.ongoing) {
        this.ongoing = null;
        this.emit("didChangeLoading", false);
      }
      this.logger.error("Error in completion requests:", error);
      return null;
    } finally {
      // Clean up any temporary state
      this.temporaryNormalCompletion = false;
      if (this.normalCompletionAbortController) {
        this.normalCompletionAbortController = null;
      }
    }
  }

  /**
   * Process NES completions with special styling and markers
   */
  private processNESCompletions(_document: TextDocument, nesResult: InlineCompletionList): InlineCompletionItem[] {
    // Record as NES completion
    this.handleEvent("show", nesResult, 0, true);

    // Update NES visibility
    this.updateNESVisibilityContext(true);

    // Special handling for NES
    return nesResult.items.map((item, index) => {
      // Create a completion item with NES marker
      const insertText =
        typeof item.insertText === "string" ? item.insertText : new SnippetString(item.insertText.value);

      const range = item.range
        ? new Range(item.range.start.line, item.range.start.character, item.range.end.line, item.range.end.character)
        : undefined;

      // Create custom documentation
      const documentation = new MarkdownString();
      documentation.appendMarkdown(`**${NES_LABEL}**\n\n`);
      documentation.appendMarkdown("Smart next edit suggestion");
      documentation.isTrusted = true;

      // Create standard VS Code inline completion item
      const completionItem = new InlineCompletionItem(insertText, range);

      completionItem.command = {
        title: "",
        command: "tabby.applyCallback",
        arguments: [
          () => {
            this.logger.info(`NES completion accepted through standard mechanism, index ${index}`);
            this.handleEvent("accept", nesResult, index);

            if (window.activeTextEditor) {
              this.clearNESDecorations(window.activeTextEditor.document.uri.toString());
            }

            this.updateNESVisibilityContext(false);
          },
        ],
      };

      return completionItem;
    });
  }

  // FIXME: We don't listen to the user cycling through the items,
  // so we don't know the 'index' (except for the 'accept' event).
  // For now, just use the first item to report other events.
  async handleEvent(
    event: "show" | "accept" | "dismiss" | "accept_word" | "accept_line",
    completions?: InlineCompletionList | null,
    index = 0,
    isNES = false,
  ) {
    if (event === "show" && completions) {
      const item = completions.items[index];
      const cmplId = item?.data?.eventId?.completionId?.replace("cmpl-", "") || Math.random().toString(36).substring(7);
      const timestamp = Date.now();
      this.displayedCompletion = {
        id: `view-${cmplId}-at-${timestamp}`,
        completions,
        index,
        displayedAt: timestamp,
        isNES,
      };

      // If NES completion, add decoration
      if (isNES && window.activeTextEditor) {
        this.updateNESDecorations(window.activeTextEditor.document.uri.toString());
        // Update NES visibility
        this.updateNESVisibilityContext(true);
      }

      await this.postEvent(event, this.displayedCompletion);
    } else if (this.displayedCompletion) {
      this.displayedCompletion.index = index;
      await this.postEvent(event, this.displayedCompletion);

      // Cleanup after event handling
      if (event === "accept" || event === "dismiss") {
        // If NES completion, clean up decorations
        if (this.displayedCompletion.isNES && window.activeTextEditor) {
          this.clearNESDecorations(window.activeTextEditor.document.uri.toString());
          // Update NES visibility
          this.updateNESVisibilityContext(false);
        }
        this.displayedCompletion = null;
      }
    }
  }

  private async postEvent(
    event: "show" | "accept" | "dismiss" | "accept_word" | "accept_line",
    displayedCompletion: DisplayedCompletion,
  ) {
    const { id, completions, index, displayedAt, isNES } = displayedCompletion;
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
        eventData = {
          type: "select",
          elapsed,
          // Add extra marker for NES completions
          selectKind: isNES ? "line" : undefined,
        };
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

  // Function to accept the current NES completion
  public acceptCurrentNESCompletion() {
    if (this.displayedCompletion?.isNES) {
      this.logger.info("Manual acceptance of NES completion triggered");
      const item = this.displayedCompletion.completions.items[this.displayedCompletion.index];

      // Only proceed if there's an active item
      if (item) {
        const editor = window.activeTextEditor;
        if (editor) {
          // Get the insertion range
          const range = item.range
            ? new Range(
                item.range.start.line,
                item.range.start.character,
                item.range.end.line,
                item.range.end.character,
              )
            : editor.selection;

          // Check if cursor is within the NES range
          const cursorPosition = editor.selection.active;
          const isInRange = range.contains(cursorPosition);

          if (!isInRange) {
            this.logger.info("Cursor not in NES range, skipping acceptance");
            return;
          }

          // Get the text to insert
          const insertText = typeof item.insertText === "string" ? item.insertText : item.insertText.value;

          this.logger.info(
            `Applying NES edit: "${insertText}" at range ${range.start.line}:${range.start.character} to ${range.end.line}:${range.end.character}`,
          );

          // Perform the edit
          editor
            .edit((editBuilder) => {
              editBuilder.replace(range, insertText);
            })
            .then((success) => {
              if (success) {
                this.logger.info("NES edit successfully applied");

                // Move cursor to the end of inserted text
                if (insertText) {
                  const textLines = insertText.split("\n");
                  if (textLines.length > 0) {
                    const lastLineLength = textLines[textLines.length - 1]?.length || 0;
                    const newLine = range.start.line + textLines.length - 1;
                    const newCharacter = textLines.length > 1 ? lastLineLength : range.start.character + lastLineLength;

                    const newPosition = new Position(newLine, newCharacter);
                    editor.selection = new Selection(newPosition, newPosition);
                  }
                }

                // Handle the acceptance event
                this.handleEvent("accept", this.displayedCompletion?.completions, this.displayedCompletion?.index || 0);

                // Clean up decorations
                if (window.activeTextEditor) {
                  this.clearNESDecorations(window.activeTextEditor.document.uri.toString());
                }

                // Clear the displayed completion
                this.displayedCompletion = null;

                // Update NES visibility
                this.updateNESVisibilityContext(false);
              } else {
                this.logger.error("Failed to apply NES edit");
              }
            });
        } else {
          this.logger.error("No active editor found when trying to accept NES completion");
        }
      } else {
        this.logger.warn("No item found in displayedCompletion when trying to accept NES completion");
      }
    } else {
      this.logger.warn("No NES completion to accept");
    }
  }

  // Update NES visibility context variable for keybinding conditions
  private updateNESVisibilityContext(visible: boolean) {
    commands.executeCommand("setContext", "tabby.nesCompletionVisible", visible);
    this.logger.info(`NES completion visibility set to: ${visible}`);
  }

  /**
   * Handle resource disposal
   */
  public dispose() {
    // Clean up all decorations
    this.clearNESDecorations();
    // Reset context
    this.updateNESVisibilityContext(false);
  }
}
