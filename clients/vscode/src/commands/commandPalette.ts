import { commands, window, Command, QuickPick, QuickPickItem, QuickPickItemKind, ThemeIcon } from "vscode";
import { State as LanguageClientState } from "vscode-languageclient";
import { Client } from "../lsp/Client";
import { Config } from "../Config";

const MENU_ITEM_INDENT_SPACING = "      ";

interface CommandPaletteItem extends QuickPickItem {
  command?: string | Command | (() => void | Promise<void>);
  picked?: boolean;
}

export class CommandPalette {
  constructor(
    private readonly client: Client,
    private readonly config: Config,
  ) {}

  show() {
    const quickPick: QuickPick<CommandPaletteItem> = window.createQuickPick();
    quickPick.title = "Tabby Command Palette";
    let items: CommandPaletteItem[] = [];

    const disconnectStatues = ["disconnected", "unauthorized"];
    if (disconnectStatues.includes(this.client.status.current?.status || "")) {
      items.push(this.itemForStatus());
    }

    // Status section
    this.client.status.on("didChange", () => {
      items[0] = this.itemForStatus();
      quickPick.items = items;
    });

    // Features section
    const validStatuses = ["ready", "readyForAutoTrigger", "readyForManualTrigger"];
    if (validStatuses.includes(this.client.status.current?.status || "")) {
      items = items.concat([
        { label: "enable/disable features", kind: QuickPickItemKind.Separator },
        {
          label:
            (this.config.inlineCompletionTriggerMode === "automatic" ? "" : MENU_ITEM_INDENT_SPACING) +
            "Code Completion",
          detail: MENU_ITEM_INDENT_SPACING + "Toggle between automatic and manual completion mode",
          picked: this.config.inlineCompletionTriggerMode === "automatic",
          command: "tabby.toggleInlineCompletionTriggerMode",
          iconPath: this.config.inlineCompletionTriggerMode === "automatic" ? new ThemeIcon("check") : undefined,
          alwaysShow: true,
        },
      ]);
    }

    // Chat section
    if (this.client.chat.isAvailable) {
      items.push({
        label: "$(comment) Chat",
        command: "tabby.chatView.focus",
      });
    }

    // Settings section
    items = items.concat([
      { label: "settings", kind: QuickPickItemKind.Separator },
      {
        label: "Connect to Server",
        command: "tabby.connectToServer",
        iconPath: new ThemeIcon("plug"),
      },
      {
        label: "Settings",
        command: "tabby.openSettings",
        iconPath: new ThemeIcon("gear"),
      },
      {
        label: "Agent Settings",
        command: "tabby.openTabbyAgentSettings",
        iconPath: new ThemeIcon("tools"),
      },
      {
        label: "Show Logs",
        command: "tabby.outputPanel.focus",
        iconPath: new ThemeIcon("output"),
      },
    ]);

    // Help section
    items = items.concat([
      { label: "help & support", kind: QuickPickItemKind.Separator },
      {
        label: "$(question) Help",
        description: "Open online documentation",
        command: "tabby.openOnlineHelp",
      },
    ]);

    if (validStatuses.includes(this.client.status.current?.status || "")) {
      items.push({ label: "server status", kind: QuickPickItemKind.Separator });

      items.push(this.itemForStatus());
    }

    quickPick.items = items;
    quickPick.onDidAccept(async () => {
      quickPick.hide();
      const command = quickPick.activeItems[0]?.command;
      if (command) {
        if (typeof command === "string") {
          await commands.executeCommand(command);
        } else if (typeof command === "function") {
          await command();
        } else if (command.arguments) {
          await commands.executeCommand(command.command, ...command.arguments);
        } else {
          await commands.executeCommand(command.command);
        }
      }
    });
    quickPick.show();
  }

  private itemForStatus(): CommandPaletteItem {
    const STATUS_PREFIX = MENU_ITEM_INDENT_SPACING + "Status: ";
    const languageClientState = this.client.languageClient.state;
    switch (languageClientState) {
      case LanguageClientState.Stopped:
      case LanguageClientState.Starting: {
        return {
          label: `${STATUS_PREFIX}Initializing...`,
        };
      }
      case LanguageClientState.Running: {
        const statusInfo = this.client.status.current;
        switch (statusInfo?.status) {
          case "connecting": {
            return {
              label: `${STATUS_PREFIX}Connecting...`,
            };
          }
          case "unauthorized": {
            return {
              label: `${STATUS_PREFIX}Unauthorized`,
              description: "Update the settings to connect to Tabby Server",
              command: "tabby.connectToServer",
            };
          }
          case "disconnected": {
            return {
              label: `${STATUS_PREFIX}Disconnected`,
              description: "Update the settings to connect to Tabby Server",
              command: "tabby.connectToServer",
            };
          }
          case "ready":
          case "readyForAutoTrigger":
          case "readyForManualTrigger":
          case "fetching": {
            return {
              label: `${STATUS_PREFIX}Ready`,
              description: this.client.agentConfig.current?.server.endpoint,
              command: "tabby.outputPanel.focus",
            };
          }
          case "completionResponseSlow": {
            return {
              label: `${STATUS_PREFIX}Slow Response`,
              description: "Completion requests appear to take too much time.",
              command: async () => {
                const currentStatusInfo = await this.client.status.fetchAgentStatusInfo();
                window
                  .showWarningMessage(
                    "Completion requests appear to take too much time.",
                    {
                      modal: true,
                      detail: currentStatusInfo.helpMessage,
                    },
                    "Online Help...",
                    "Don't Show Again",
                  )
                  .then((selection) => {
                    switch (selection) {
                      case "Online Help...":
                        commands.executeCommand("tabby.openOnlineHelp");
                        break;
                      case "Don't Show Again":
                        commands.executeCommand("tabby.status.addIgnoredIssues", "completionResponseSlow");
                        break;
                    }
                  });
              },
            };
          }
          default: {
            return {
              label: `${STATUS_PREFIX}Unknown Status`,
              description: "Please check the logs for more information.",
              command: "tabby.outputPanel.focus",
            };
          }
        }
      }
    }
  }
}
