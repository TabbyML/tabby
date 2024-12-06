import { commands, window, Command, QuickPick, QuickPickItem, QuickPickItemKind } from "vscode";
import { State as LanguageClientState } from "vscode-languageclient";
import { Client } from "../lsp/Client";
import { Config } from "../Config";

interface CommandPaletteItem extends QuickPickItem {
  command?: string | Command | (() => void | Promise<void>);
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

    items.push(this.itemForStatus());
    this.client.status.on("didChange", () => {
      items[0] = this.itemForStatus();
      quickPick.items = items;
    });

    if (this.client.chat.isAvailable) {
      items.push({
        label: "$(comment) Chat",
        command: "tabby.chatView.focus",
      });
    }

    items.push(
      {
        label: "",
        kind: QuickPickItemKind.Separator,
      },
      {
        label: this.config.inlineCompletionTriggerMode === "manual" ? "Enable Completions" : "Disable Completions",
        command: "tabby.toggleInlineCompletionTriggerMode",
      },
      {
        label: "",
        kind: QuickPickItemKind.Separator,
      },
      {
        label: "$(plug) Connect to Server...",
        command: "tabby.connectToServer",
      },
      {
        label: "$(settings) Settings...",
        command: "tabby.openSettings",
      },
      {
        label: "$(gear) Agent Settings...",
        command: "tabby.openTabbyAgentSettings",
      },
      {
        label: "$(output) Show Logs...",
        command: "tabby.outputPanel.focus",
      },
      {
        label: "",
        kind: QuickPickItemKind.Separator,
      },
      {
        label: "$(question) Help",
        command: "tabby.openOnlineHelp",
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
    const languageClientState = this.client.languageClient.state;
    switch (languageClientState) {
      case LanguageClientState.Stopped:
      case LanguageClientState.Starting: {
        return {
          label: "$(loading~spin) Initializing...",
        };
      }
      case LanguageClientState.Running: {
        const statusInfo = this.client.status.current;
        switch (statusInfo?.status) {
          case "connecting": {
            return {
              label: "$(loading~spin) Connecting...",
            };
          }
          case "unauthorized": {
            return {
              label: "$(key) Unauthorized",
              description: "Update the settings to connect to Tabby Server",
              command: "tabby.connectToServer",
            };
          }
          case "disconnected": {
            return {
              label: "$(debug-disconnect) Disconnected",
              description: "Update the settings to connect to Tabby Server",
              command: "tabby.connectToServer",
            };
          }
          case "ready":
          case "readyForAutoTrigger":
          case "readyForManualTrigger":
          case "fetching": {
            return {
              label: "$(check) Ready",
              description: this.client.agentConfig.current?.server.endpoint,
              command: "tabby.outputPanel.focus",
            };
          }
          case "completionResponseSlow": {
            return {
              label: "$(warning) Slow Response",
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
              label: "$(warning) Unknown Status",
              description: "Please check the logs for more information.",
              command: "tabby.outputPanel.focus",
            };
          }
        }
      }
    }
  }
}
