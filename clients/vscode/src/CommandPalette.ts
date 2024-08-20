import { commands, QuickPick, QuickPickItem, QuickPickItemKind, ThemeIcon, window } from "vscode";
import { State as LanguageClientState } from "vscode-languageclient";
import { Client } from "./lsp/Client";
import { Config } from "./Config";
import { Issues } from "./Issues";

export default class CommandPalette {
  quickPick: QuickPick<CommandPaletteItem>;

  constructor(
    private readonly client: Client,
    private readonly config: Config,
    private readonly issues: Issues,
  ) {
    this.quickPick = window.createQuickPick();
    this.quickPick.title = "Tabby Command Palette";

    let items: CommandPaletteItem[] = [this.itemForStatus()];

    if (this.client.chat.isAvailable) {
      items.push({
        label: "Chat",
        command: "tabby.chatView.focus",
        iconPath: new ThemeIcon("comment"),
      });
    }

    items = items.concat([
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
        label: "Set Credentials",
        command: "tabby.setApiToken",
        iconPath: new ThemeIcon("key"),
      },
      {
        label: "Settings...",
        command: "tabby.openSettings",
        iconPath: new ThemeIcon("extensions-manage"),
      },
      {
        label: "Agent Settings...",
        command: "tabby.openTabbyAgentSettings",
        iconPath: new ThemeIcon("console"),
      },
      {
        label: "Show Logs...",
        command: "tabby.outputPanel.focus",
      },
      {
        label: "",
        kind: QuickPickItemKind.Separator,
      },
      {
        label: "Help",
        command: "tabby.openOnlineHelp",
        iconPath: new ThemeIcon("question"),
      },
    ]);

    this.quickPick.items = items;
    this.quickPick.onDidAccept(this.onDidAccept, this);
    this.quickPick.show();
  }

  onDidAccept() {
    this.quickPick.hide();
    const item = this.quickPick.activeItems[0];
    if (item?.command) {
      if (typeof item.command === "function") {
        item.command();
      } else {
        commands.executeCommand(item.command);
      }
    }
  }

  private itemForStatus(): CommandPaletteItem {
    const lspState = this.client.languageClient.state;
    const agentStatus = this.client.agent.status;
    const item: CommandPaletteItem = {
      label: "Status",
      iconPath: new ThemeIcon("warning"),
    };
    if (lspState === LanguageClientState.Starting || agentStatus === "notInitialized") {
      item.label = "Starting...";
      item.iconPath = new ThemeIcon("sync");
    } else if (lspState === LanguageClientState.Stopped || agentStatus === "finalized") {
      item.label = "Disabled";
      item.iconPath = new ThemeIcon("circle-slash");
    } else if (agentStatus === "disconnected" || this.issues.first === "connectionFailed") {
      item.label = "Disconnected";
      item.description = "Cannot connect to Tabby Server";
      item.command = "tabby.openSettings";
    } else if (agentStatus === "unauthorized") {
      item.label = "Unauthorized";
      item.description = "Your credentials are invalid";
      item.command = "tabby.setApiToken";
    } else if (this.issues.length > 0) {
      switch (this.issues.first) {
        case "highCompletionTimeoutRate":
          item.label = "Timeout"
          item.description = "Most completion requests timed out.";
          break;
        case "slowCompletionResponseTime":
          item.label = "Slow Response";
          item.description = "Completion requests appear to take too much time.";
          break;
      }
      item.command = () => this.issues.showHelpMessage();
    } else if (agentStatus === "ready") {
      item.label = "Ready";
      item.iconPath = new ThemeIcon("check");
      item.command = "tabby.outputPanel.focus";
    }

    return item;
  }
}

interface CommandPaletteItem extends QuickPickItem {
  command?: string | CallbackCommand;
}

type CallbackCommand = () => void;