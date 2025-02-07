import {
  commands,
  window,
  ThemeIcon,
  QuickPickItem,
  QuickPickItemKind,
  QuickPickItemButtonEvent,
  ProgressLocation,
} from "vscode";
import { isBrowser } from "../env";
import { Client } from "../lsp/client";
import { Config } from "../Config";

interface ServerQuickPickItem extends QuickPickItem {
  endpoint: string;
}

export class ConnectToServerWidget {
  private quickPick = window.createQuickPick<ServerQuickPickItem>();

  constructor(
    private client: Client,
    private config: Config,
  ) {}

  /**
   * Show the widget to connect to a Tabby Server:
   * 1. Input the URL of the Tabby Server, show the recent server list for quick selection.
   * 2. If no token saved for the server, ask for the token.
   * 3. Ensure the connection to the server.
   */
  async show(): Promise<void> {
    return new Promise((resolve) => {
      const quickPick = this.quickPick;
      quickPick.title = "Enter the URL of your Tabby Server";
      quickPick.items = this.buildQuickPickItems();
      quickPick.onDidChangeValue(() => {
        quickPick.items = this.buildQuickPickItems();
      });
      quickPick.onDidAccept(async () => {
        quickPick.hide();
        const selected = quickPick.activeItems[0];
        if (selected) {
          if (selected.endpoint == "") {
            // Use config in tabby agent config file
            await this.config.updateServerEndpoint("");
          } else {
            const serverRecords = this.config.serverRecords;
            const record = serverRecords.get(selected.endpoint);
            if (!record) {
              const token = await window.showInputBox({
                title: "Enter your token",
                placeHolder: "auth_" + "*".repeat(32),
                password: true,
                ignoreFocusOut: true,
              });
              if (token == undefined) {
                // User canceled
                resolve();
                return;
              }
              serverRecords.set(selected.endpoint, { token, updatedAt: Date.now() });
            } else {
              record.updatedAt = Date.now();
            }
            await this.config.updateServerRecords(serverRecords);
            await this.config.updateServerEndpoint(selected.endpoint);
          }
          await this.ensureConnection();
        }
        resolve();
      });
      quickPick.onDidTriggerItemButton(async (event: QuickPickItemButtonEvent<ServerQuickPickItem>) => {
        const item = event.item;
        const button = event.button;
        if (button.iconPath instanceof ThemeIcon) {
          if (button.iconPath.id == "settings-remove") {
            const serverRecords = this.config.serverRecords;
            serverRecords.delete(item.endpoint);
            await this.config.updateServerRecords(serverRecords);
            quickPick.items = this.buildQuickPickItems();
          } else if (button.iconPath.id == "settings-edit") {
            commands.executeCommand("tabby.openTabbyAgentSettings");
          }
        }
      });
      quickPick.show();
    });
  }

  /**
   * Show the widget to update the token for the current server:
   * 1. Ask for the new token.
   * 2. Ensure the connection to the server.
   */
  async showUpdateTokenWidget(): Promise<void> {
    const serverRecords = this.config.serverRecords;
    const endpoint = this.config.serverEndpoint;

    if (endpoint == "") {
      // Should not reach here
      throw new Error("This method should not be called when using the config from Tabby Agent Settings.");
    }

    const token = await window.showInputBox({
      title: "Your token is invalid. Please update your token",
      placeHolder: "auth_" + "*".repeat(32),
      password: true,
      ignoreFocusOut: true,
    });
    if (token == undefined) {
      // User canceled
      return;
    }
    serverRecords.set(endpoint, { token, updatedAt: Date.now() });
    await this.config.updateServerRecords(serverRecords);
    await this.ensureConnection();
  }

  private async ensureConnection(): Promise<void> {
    const endpoint = this.config.serverEndpoint;
    const statusInfo = await window.withProgress(
      {
        location: ProgressLocation.Notification,
        title: "Connecting to Tabby Server...",
        cancellable: true,
      },
      async () => {
        return await this.client.status.fetchAgentStatusInfo({ recheckConnection: true });
      },
    );

    if (statusInfo.status == "disconnected") {
      const selected = await window.showErrorMessage(
        "Failed to connect to Tabby Server.",
        {
          modal: true,
          detail: statusInfo.helpMessage,
        },
        "Select Server",
      );
      if (selected == "Select Server") {
        const newWidget = new ConnectToServerWidget(this.client, this.config);
        await newWidget.show();
      }
    } else if (statusInfo.status == "unauthorized") {
      if (endpoint == "") {
        const selected = await window.showErrorMessage(
          "Your token is invalid. Please update your token in Tabby Agent Settings.",
          { modal: true },
          "Tabby Agent Settings...",
        );
        if (selected == "Tabby Agent Settings...") {
          await commands.executeCommand("tabby.openTabbyAgentSettings");
        }
      } else {
        const selected = await window.showErrorMessage(
          "Your token is invalid. Please update your token.",
          { modal: true },
          "Update Token",
        );
        if (selected == "Update Token") {
          await this.showUpdateTokenWidget();
        }
      }
    }
  }

  private buildQuickPickItems(): ServerQuickPickItem[] {
    const serverRecords = this.config.serverRecords;
    const defaultEndpoint = "http://localhost:8080";

    const items: ServerQuickPickItem[] = Array.from(serverRecords.entries())
      .map(([endpoint, record]) => {
        const isCurrent = endpoint == this.config.serverEndpoint;
        return {
          label: endpoint,
          description: isCurrent ? "Current" : "",
          iconPath: new ThemeIcon("server"),
          buttons: isCurrent
            ? []
            : [
                {
                  iconPath: new ThemeIcon("settings-remove"),
                  tooltip: "Remove from Recent Server List",
                },
              ],
          endpoint,
          ...record,
        };
      })
      .toSorted((a, b) => b.updatedAt - a.updatedAt);

    if (!items.find((i) => i.label == defaultEndpoint)) {
      items.push({
        label: defaultEndpoint,
        iconPath: new ThemeIcon("server"),
        endpoint: defaultEndpoint,
      });
    }

    if (!isBrowser) {
      items.push({
        label: "",
        endpoint: "",
        kind: QuickPickItemKind.Separator,
      });

      items.push({
        label: "Use Configuration in Tabby Agent Settings",
        endpoint: "",
        description: this.config.serverEndpoint == "" ? "Current" : "",
        iconPath: new ThemeIcon("settings"),
        buttons: [
          {
            iconPath: new ThemeIcon("settings-edit"),
            tooltip: "Edit Tabby Agent Settings",
          },
        ],
        alwaysShow: true,
      });
    }

    const input = this.quickPick.value;
    if (this.isValidateServerEndpoint(input) && !items.find((i) => i.label === input)) {
      items.unshift({
        label: input,
        endpoint: input,
        iconPath: new ThemeIcon("add"),
        description: "Add New Server",
        alwaysShow: true,
      });
    }

    return items;
  }

  private isValidateServerEndpoint(input: string): boolean {
    try {
      const url = new URL(input);
      return url.protocol == "http:" || url.protocol == "https:";
    } catch {
      return false;
    }
  }
}
