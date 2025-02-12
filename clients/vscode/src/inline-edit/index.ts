import type { Location } from "vscode-languageclient";
import { window, TextEditor, Selection, Position, CancellationToken, Range } from "vscode";
import { Client } from "../lsp/client";
import { Config } from "../Config";
import { ContextVariables } from "../ContextVariables";
import { getLogger } from "../logger";
import { InlineEditCommand, UserCommandQuickpick } from "./quickPick";

export class InlineEditController {
  private readonly logger = getLogger("InlineEditController");
  private readonly editLocation: Location;

  constructor(
    private client: Client,
    private config: Config,
    private contextVariables: ContextVariables,
    private editor: TextEditor,
    private range: Range,
  ) {
    this.editLocation = {
      uri: this.editor.document.uri.toString(),
      range: {
        start: { line: this.range.start.line, character: 0 },
        end: {
          line: this.range.end.character === 0 ? this.range.end.line : this.range.end.line + 1,
          character: 0,
        },
      },
    };
  }

  async start(userCommand: string | undefined, cancellationToken: CancellationToken) {
    const inlineEditCommand: InlineEditCommand | undefined = userCommand
      ? { command: userCommand }
      : await this.showQuickPick();
    if (inlineEditCommand?.command) {
      await this.provideEditWithCommand(inlineEditCommand, cancellationToken);
    }
  }

  private async showQuickPick(): Promise<InlineEditCommand | undefined> {
    const quickPick = new UserCommandQuickpick(this.client, this.config, this.editor, this.editLocation);
    return await quickPick.start();
  }

  private async provideEditWithCommand(command: InlineEditCommand, cancellationToken: CancellationToken) {
    // Lock the cursor (editor selection) at start position, it will be unlocked after the edit is done
    const startPosition = new Position(this.range.start.line, 0);
    const resetEditorSelection = () => {
      this.editor.selection = new Selection(startPosition, startPosition);
    };
    const selectionListenerDisposable = window.onDidChangeTextEditorSelection((event) => {
      if (event.textEditor === this.editor) {
        resetEditorSelection();
      }
    });
    resetEditorSelection();

    this.contextVariables.chatEditInProgress = true;
    this.logger.log(`Provide edit with command: ${JSON.stringify(command)}`);
    try {
      await this.client.chat.provideEdit(
        {
          location: this.editLocation,
          command: command.command,
          context: command.context,
          format: "previewChanges",
        },
        cancellationToken,
      );
    } catch (error) {
      if (typeof error === "object" && error && "message" in error && typeof error["message"] === "string") {
        if (cancellationToken.isCancellationRequested || error["message"].includes("This operation was aborted")) {
          // user canceled
        } else {
          window.showErrorMessage(error["message"]);
        }
      }
    }
    selectionListenerDisposable.dispose();
    this.contextVariables.chatEditInProgress = false;
  }
}
