import { commands, window, Command, QuickPick, QuickPickItem, QuickPickItemKind, ThemeIcon } from "vscode";
import { State as LanguageClientState } from "vscode-languageclient";
import { Client } from "../lsp/client";
import { Config } from "../Config";
import { isBrowser } from "../env";

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
    const items: CommandPaletteItem[] = [];

    // Status section
    const status = this.client.status.current?.status;
    items.push(
      {
        label: "status",
        kind: QuickPickItemKind.Separator,
      },
      this.itemForStatus(),
    );

    this.client.status.on("didChange", () => {
      items[1] = this.itemForStatus();
      quickPick.items = items;
    });

    // Features section
    const validStatuses = ["ready", "readyForAutoTrigger", "readyForManualTrigger"];
    if (status !== undefined && validStatuses.includes(status)) {
      const iconPath = this.config.inlineCompletionTriggerMode === "automatic" ? new ThemeIcon("check") : undefined;
      const labelPrefix = iconPath ? "" : MENU_ITEM_INDENT_SPACING;

      items.push(
        {
          label: "enable/disable features",
          kind: QuickPickItemKind.Separator,
        },
        {
          label: labelPrefix + "Code Completion",
          detail: MENU_ITEM_INDENT_SPACING + "Toggle between automatic and manual completion mode",
          picked: this.config.inlineCompletionTriggerMode === "automatic",
          command: "tabby.toggleInlineCompletionTriggerMode",
          iconPath: iconPath,
          alwaysShow: true,
        },
      );
    }

    // Chat section
    if (this.client.chat.isAvailable) {
      items.push({
        label: "Chat",
        command: "tabby.chatView.focus",
        iconPath: new ThemeIcon("comment"),
      });
    }

    // Settings section
    items.push(
      {
        label: "settings",
        kind: QuickPickItemKind.Separator,
      },
      {
        label: "Connect to Server",
        command: "tabby.connectToServer",
        iconPath: new ThemeIcon("plug"),
      },
    );
    if (status === "unauthorized") {
      items.push({
        label: "Update Token",
        command: "tabby.updateToken",
        iconPath: new ThemeIcon("key"),
      });
    }
    items.push({
      label: "Settings",
      command: "tabby.openSettings",
      iconPath: new ThemeIcon("settings"),
    });
    if (!isBrowser) {
      items.push({
        label: "Agent Settings",
        command: "tabby.openTabbyAgentSettings",
        iconPath: new ThemeIcon("tools"),
      });
    }
    items.push({
      label: "Show Logs",
      command: "tabby.outputPanel.focus",
      iconPath: new ThemeIcon("output"),
    });

    // Help section
    items.push(
      {
        label: "help & support",
        kind: QuickPickItemKind.Separator,
      },
      {
        label: "Help",
        description: "Open online documentation",
        command: "tabby.openOnlineHelp",
        iconPath: new ThemeIcon("question"),
      },
    );

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
              description: "Update your token to connect to Tabby Server",
              command: "tabby.updateToken",
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
          case "rateLimitExceeded": {
            return {
              label: `${STATUS_PREFIX}Too Many Requests`,
              description: "Request limit exceeded",
              command: "tabby.outputPanel.focus",
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
