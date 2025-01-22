import type { Location } from "vscode-languageclient";
import {
  CancellationTokenSource,
  QuickPickItem,
  ThemeIcon,
  QuickPickItemKind,
  window,
  TextEditor,
  Selection,
  Position,
  CancellationToken,
  Range,
} from "vscode";
import { ChatEditCommand } from "tabby-agent";
import { Client } from "../lsp/Client";
import { Config } from "../Config";
import { ContextVariables } from "../ContextVariables";
import { getLogger } from "../logger";

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
    const command = userCommand ?? (await this.showQuickPick());
    if (command) {
      this.logger.log(`Start inline edit with user command: ${command}`);
      await this.provideEditWithCommand(command, cancellationToken);
    }
  }

  private async showQuickPick(): Promise<string | undefined> {
    return new Promise((resolve) => {
      const quickPick = window.createQuickPick<CommandQuickPickItem>();
      quickPick.placeholder = "Enter the command for editing";
      quickPick.matchOnDescription = true;

      const recentlyCommand = this.config.chatEditRecentlyCommand.slice(0, this.config.maxChatEditHistory);
      const suggestedCommand: ChatEditCommand[] = [];

      const updateQuickPickList = () => {
        const input = quickPick.value;
        const list: CommandQuickPickItem[] = [];
        list.push(
          ...suggestedCommand.map((item) => ({
            label: item.label,
            value: item.command,
            iconPath: item.source === "preset" ? new ThemeIcon("run") : new ThemeIcon("spark"),
            description: item.source === "preset" ? item.command : "Suggested",
          })),
        );
        if (list.length > 0) {
          list.push({
            label: "",
            value: "",
            kind: QuickPickItemKind.Separator,
            alwaysShow: true,
          });
        }
        const recentlyCommandToAdd = recentlyCommand.filter((item) => !list.find((i) => i.value === item));
        list.push(
          ...recentlyCommandToAdd.map((item) => ({
            label: item,
            value: item,
            iconPath: new ThemeIcon("history"),
            description: "History",
            buttons: [
              {
                iconPath: new ThemeIcon("edit"),
              },
              {
                iconPath: new ThemeIcon("settings-remove"),
              },
            ],
          })),
        );
        if (input.length > 0 && !list.find((i) => i.value === input)) {
          list.unshift({
            label: input,
            value: input,
            iconPath: new ThemeIcon("run"),
            description: "",
            alwaysShow: true,
          });
        }
        quickPick.items = list;
      };

      quickPick.onDidChangeValue(() => updateQuickPickList());

      const fetchingSuggestedCommandCancellationTokenSource = new CancellationTokenSource();
      this.client.chat.provideEditCommands(
        { location: this.editLocation },
        { commands: suggestedCommand, callback: () => updateQuickPickList() },
        fetchingSuggestedCommandCancellationTokenSource.token,
      );

      quickPick.onDidTriggerItemButton(async (event) => {
        const item = event.item;
        const button = event.button;
        if (button.iconPath instanceof ThemeIcon && button.iconPath.id === "settings-remove") {
          const index = recentlyCommand.indexOf(item.value);
          if (index !== -1) {
            recentlyCommand.splice(index, 1);
            await this.config.updateChatEditRecentlyCommand(recentlyCommand);
            updateQuickPickList();
          }
        }

        if (button.iconPath instanceof ThemeIcon && button.iconPath.id === "edit") {
          quickPick.value = item.value;
        }
      });

      quickPick.onDidAccept(async () => {
        const command = quickPick.selectedItems[0]?.value;
        if (!command) {
          resolve(undefined);
          return;
        }
        if (command && command.length > 200) {
          window.showErrorMessage("Command is too long.");
          resolve(undefined);
          return;
        }

        const recentlyCommand = this.config.chatEditRecentlyCommand;
        const updatedRecentlyCommand = [command]
          .concat(recentlyCommand.filter((item) => item !== command))
          .slice(0, this.config.maxChatEditHistory);
        await this.config.updateChatEditRecentlyCommand(updatedRecentlyCommand);

        resolve(command);
        quickPick.hide();
      });
      quickPick.onDidHide(() => {
        fetchingSuggestedCommandCancellationTokenSource.cancel();
        resolve(undefined);
      });

      quickPick.show();
    });
  }

  private async provideEditWithCommand(command: string, cancellationToken: CancellationToken) {
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
    this.logger.log(`Provide edit with command: ${command}`);
    try {
      await this.client.chat.provideEdit(
        {
          location: this.editLocation,
          command,
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

interface CommandQuickPickItem extends QuickPickItem {
  value: string;
}
