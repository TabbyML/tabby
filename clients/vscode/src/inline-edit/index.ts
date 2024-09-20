import { ChatEditCommand } from "tabby-agent";
import { Config } from "../Config";
import {
  CancellationTokenSource,
  QuickPickItem,
  ThemeIcon,
  QuickPickItemKind,
  window,
  TextEditor,
  Selection,
  Position,
  QuickPick,
  QuickPickItemButtonEvent,
} from "vscode";
import { Client } from "../lsp/Client";
import { ContextVariables } from "../ContextVariables";

export class InlineEditController {
  private chatEditCancellationTokenSource: CancellationTokenSource | null = null;
  private quickPick: QuickPick<EditCommand>;

  private recentlyCommand: string[] = [];
  private suggestedCommand: ChatEditCommand[] = [];

  constructor(
    private client: Client,
    private config: Config,
    private contextVariables: ContextVariables,
    private editor: TextEditor,
    private editLocation: EditLocation,
    private userCommand?: string,
  ) {
    this.recentlyCommand = this.config.chatEditRecentlyCommand.slice(0, this.config.maxChatEditHistory);

    const fetchingSuggestedCommandCancellationTokenSource = new CancellationTokenSource();
    this.client.chat.provideEditCommands(
      { location: editLocation },
      { commands: this.suggestedCommand, callback: () => this.updateQuickPickList() },
      fetchingSuggestedCommandCancellationTokenSource.token,
    );

    const quickPick = window.createQuickPick<EditCommand>();
    quickPick.placeholder = "Enter the command for editing";
    quickPick.matchOnDescription = true;
    quickPick.onDidChangeValue(() => this.updateQuickPickList());
    quickPick.onDidHide(() => {
      fetchingSuggestedCommandCancellationTokenSource.cancel();
    });
    quickPick.onDidAccept(this.onDidAccept, this);
    quickPick.onDidTriggerItemButton(this.onDidTriggerItemButton, this);

    this.quickPick = quickPick;
  }

  async start() {
    this.userCommand ? await this.provideEditWithCommand(this.userCommand) : this.quickPick.show();
  }

  private async onDidAccept() {
    const command = this.quickPick.selectedItems[0]?.value;
    this.quickPick.hide();
    if (!command) {
      return;
    }
    if (command && command.length > 200) {
      window.showErrorMessage("Command is too long.");
      return;
    }
    await this.provideEditWithCommand(command);
  }

  private async provideEditWithCommand(command: string) {
    const startPosition = new Position(this.editLocation.range.start.line, this.editLocation.range.start.character);

    if (!this.userCommand) {
      const updatedRecentlyCommand = [command]
        .concat(this.recentlyCommand.filter((item) => item !== command))
        .slice(0, this.config.maxChatEditHistory);
      this.config.chatEditRecentlyCommand = updatedRecentlyCommand;
    }

    this.editor.selection = new Selection(startPosition, startPosition);
    this.contextVariables.chatEditInProgress = true;
    this.chatEditCancellationTokenSource = new CancellationTokenSource();
    try {
      await this.client.chat.provideEdit(
        {
          location: this.editLocation,
          command,
          format: "previewChanges",
        },
        this.chatEditCancellationTokenSource.token,
      );
    } catch (error) {
      if (typeof error === "object" && error && "message" in error && typeof error["message"] === "string") {
        window.showErrorMessage(error["message"]);
      }
    }
    this.chatEditCancellationTokenSource.dispose();
    this.chatEditCancellationTokenSource = null;
    this.contextVariables.chatEditInProgress = false;
    this.editor.selection = new Selection(startPosition, startPosition);
  }

  private onDidTriggerItemButton(event: QuickPickItemButtonEvent<EditCommand>) {
    const item = event.item;
    const button = event.button;
    if (button.iconPath instanceof ThemeIcon && button.iconPath.id === "settings-remove") {
      const index = this.recentlyCommand.indexOf(item.value);
      if (index !== -1) {
        this.recentlyCommand.splice(index, 1);
        this.config.chatEditRecentlyCommand = this.recentlyCommand;
        this.updateQuickPickList();
      }
    }

    if (button.iconPath instanceof ThemeIcon && button.iconPath.id === "edit") {
      this.quickPick.value = item.value;
    }
  }

  private updateQuickPickList() {
    const input = this.quickPick.value;
    const list: (QuickPickItem & { value: string })[] = [];
    list.push(
      ...this.suggestedCommand.map((item) => ({
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
    const recentlyCommandToAdd = this.recentlyCommand.filter((item) => !list.find((i) => i.value === item));
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
    this.quickPick.items = list;
  }
}

interface EditCommand extends QuickPickItem {
  value: string;
}

interface EditLocation {
  uri: string;
  range: {
    start: { line: number; character: number };
    end: { line: number; character: number };
  };
}
