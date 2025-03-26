import { commands, window, Command, QuickPick, QuickPickItem, QuickPickItemKind, ThemeIcon } from "vscode";
import { State as LanguageClientState } from "vscode-languageclient";
import { Client } from "../lsp/client";
import { Config } from "../Config";
import { isBrowser } from "../env";

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
    quickPick.items = this.buildMenuItems();

    this.client.status.on("didChange", () => {
      quickPick.items = this.buildMenuItems();
    });

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

  private buildMenuItems(): CommandPaletteItem[] {
    const items: CommandPaletteItem[] = [];
    const status = this.client.status.current?.status;

    // Status section
    items.push({
      label: "status",
      kind: QuickPickItemKind.Separator,
    });
    items.push(this.itemForStatus());

    // Chat section
    items.push({
      label: "chat",
      kind: QuickPickItemKind.Separator,
    });
    if (this.client.chat.isAvailable) {
      items.push({
        label: "Chat",
        command: "tabby.chatView.focus",
        iconPath: new ThemeIcon("comment"),
      });
    }

    // Completion section
    items.push({
      label: "code completion",
      kind: QuickPickItemKind.Separator,
    });
    const invalidStatuses = ["connecting", "unauthorized", "disconnected"];
    if (status !== undefined && !invalidStatuses.includes(status)) {
      const isAutomatic = this.config.inlineCompletionTriggerMode === "automatic";

      const currentLanguageId = window.activeTextEditor?.document.languageId;
      const isLanguageDisabled = currentLanguageId ? this.config.disabledLanguages.includes(currentLanguageId) : false;

      items.push({
        label: (isAutomatic ? "Disable" : "Enable") + " auto completions",
        picked: isAutomatic,
        command: "tabby.toggleInlineCompletionTriggerMode",
        alwaysShow: true,
      });

      if (currentLanguageId) {
        items.push({
          label: (isLanguageDisabled ? "Enable" : "Disable") + ` completions for ${currentLanguageId}`,
          picked: !isLanguageDisabled,
          command: {
            title: "toggleLanguageInlineCompletion",
            command: "tabby.toggleLanguageInlineCompletion",
            arguments: [currentLanguageId],
          },
          alwaysShow: true,
        });
      }
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

    return items;
  }

  private itemForStatus(): CommandPaletteItem {
    const STATUS_PREFIX = "Status: ";
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
