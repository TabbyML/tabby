import { commands, QuickPick, QuickPickItem, QuickPickItemKind, ThemeIcon, window } from "vscode";
import { State as LanguageClientState } from "vscode-languageclient";
import { Client } from "./lsp/Client";
import { Config } from "./Config";
import { Issues } from "./Issues";

const MENU_ITEM_INDENT_SPACING = "      ";

export default class CommandPalette {
  quickPick: QuickPick<CommandPaletteItem>;

  constructor(
    private readonly client: Client,
    private readonly config: Config,
    private readonly issues: Issues,
  ) {
    this.quickPick = window.createQuickPick();
    this.quickPick.title = "Tabby Command Palette";

    let items: CommandPaletteItem[] = [];

    // only show features if the agent is ready
    if (this.client.agent.status === "ready") {
      // TODO: check different feature health 1by1, they could be a server without completion model but only with chat model
      // Feature sections
      items = items.concat([
        // Features Section
        { label: "enable/disable features", kind: QuickPickItemKind.Separator },
        {
          label:
            (this.config.inlineCompletionTriggerMode === "automatic" ? "" : MENU_ITEM_INDENT_SPACING) +
            "Code Completions",
          detail: MENU_ITEM_INDENT_SPACING + "Toggle between automatic and manual completion mode",
          picked: this.config.inlineCompletionTriggerMode === "automatic",
          command: "tabby.toggleInlineCompletionTriggerMode",
          iconPath: this.config.inlineCompletionTriggerMode === "automatic" ? new ThemeIcon("check") : undefined,
          alwaysShow: true,
        },
      ]);
    }

    if (this.client.chat.isAvailable) {
      items = items.concat([
        {
          label: "Open Tabby Chat",
          command: "tabby.chatView.focus",
          iconPath: new ThemeIcon("comment"),
        },
      ]);
    }

    // settings Section
    items = items.concat([
      { label: "settings", kind: QuickPickItemKind.Separator },
      {
        label: "Set Credentials",
        command: "tabby.setApiToken",
        iconPath: new ThemeIcon("key"),
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

    items = items.concat([
      { label: "help & support", kind: QuickPickItemKind.Separator },
      {
        label: "Help",
        description: "Open online documentation",
        command: "tabby.openOnlineHelp",
        iconPath: new ThemeIcon("remote-explorer-documentation"),
      },
    ]);

    items = items.concat(
      { label: "", kind: QuickPickItemKind.Separator },
      { label: "connection status", kind: QuickPickItemKind.Separator },
      this.itemForStatus(),
    );
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
    const STATUS_PREFIX = "Status: ";
    const lspState = this.client.languageClient.state;
    const agentStatus = this.client.agent.status;
    const item: CommandPaletteItem = {
      label: `${STATUS_PREFIX}Checking...`,
    };

    if (lspState === LanguageClientState.Starting || agentStatus === "notInitialized") {
      item.label = `${STATUS_PREFIX}Starting...`;
    } else if (lspState === LanguageClientState.Stopped || agentStatus === "finalized") {
      item.label = `${STATUS_PREFIX}Disabled`;
    } else if (agentStatus === "disconnected" || this.issues.first === "connectionFailed") {
      item.label = `${STATUS_PREFIX}Disconnected`;
      item.description = "Cannot connect to Tabby Server";
      item.command = "tabby.openSettings";
    } else if (agentStatus === "unauthorized") {
      item.label = `${STATUS_PREFIX}Unauthorized`;
      item.description = "Your credentials are invalid";
      item.command = "tabby.setApiToken";
    } else if (this.issues.length > 0) {
      switch (this.issues.first) {
        case "highCompletionTimeoutRate":
          item.label = `${STATUS_PREFIX}Timeout`;
          item.description = "Most completion requests timed out.";
          break;
        case "slowCompletionResponseTime":
          item.label = `${STATUS_PREFIX}Slow Response`;
          item.description = "Completion requests appear to take too much time.";
          break;
      }
      item.command = () => this.issues.showHelpMessage(undefined, true);
    } else if (agentStatus === "ready") {
      item.label = `${STATUS_PREFIX}Ready`;
      item.description = this.config.serverEndpoint;
      item.command = "tabby.outputPanel.focus";
    }

    return item;
  }
}

interface CommandPaletteItem extends QuickPickItem {
  command?: string | CallbackCommand;
}

type CallbackCommand = () => void;
