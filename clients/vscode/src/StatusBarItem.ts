import { commands, window, ExtensionContext, StatusBarAlignment, ThemeColor } from "vscode";
import { State as LanguageClientState } from "vscode-languageclient";
import { Client } from "./lsp/client";
import { Config } from "./Config";

const label = "Tabby";
const iconAutomatic = "$(check)";
const iconManual = "$(chevron-right)";
const iconDisabled = "$(x)";
const iconLoading = "$(loading~spin)";
const iconDisconnected = "$(debug-disconnect)";
const iconUnauthorized = "$(key)";
const iconWarning = "$(warning)";
const colorNormal = new ThemeColor("statusBar.foreground");
const colorWarning = new ThemeColor("statusBarItem.warningForeground");
const backgroundColorNormal = new ThemeColor("statusBar.background");
const backgroundColorWarning = new ThemeColor("statusBarItem.warningBackground");

export class StatusBarItem {
  private item = window.createStatusBarItem(StatusBarAlignment.Right);

  constructor(
    private readonly client: Client,
    private readonly config: Config,
  ) {
    this.item.command = {
      title: "Show Tabby Command Palette",
      command: "tabby.commandPalette.trigger",
    };

    this.update();
    this.client.languageClient.onDidChangeState(() => this.update());
    this.client.status.on("didChange", () => this.update());
    this.config.on("updated", () => this.update());
  }

  registerInContext(context: ExtensionContext) {
    context.subscriptions.push(this.item);
    this.item.show();
  }

  update() {
    const languageClientState = this.client.languageClient.state;
    switch (languageClientState) {
      case LanguageClientState.Stopped:
      case LanguageClientState.Starting: {
        this.setColorNormal();
        this.setIcon(iconLoading);
        this.setTooltip("Tabby: Initializing...");
        break;
      }
      case LanguageClientState.Running: {
        const statusInfo = this.client.status.current;
        switch (statusInfo?.status) {
          case "connecting": {
            this.setColorNormal();
            this.setIcon(iconLoading);
            this.setTooltip(statusInfo.tooltip);
            break;
          }
          case "unauthorized": {
            this.setColorWarning();
            this.setIcon(iconUnauthorized);
            this.setTooltip(statusInfo.tooltip);
            break;
          }
          case "disconnected": {
            this.setColorWarning();
            this.setIcon(iconDisconnected);
            this.setTooltip(statusInfo.tooltip);
            break;
          }
          case "ready":
          case "readyForAutoTrigger": {
            if (this.checkIfVSCodeInlineCompletionEnabled()) {
              this.setColorNormal();
              this.setIcon(iconAutomatic);
              this.setTooltip(statusInfo.tooltip);
            }
            break;
          }
          case "readyForManualTrigger": {
            if (this.checkIfVSCodeInlineCompletionEnabled()) {
              this.setColorNormal();
              this.setIcon(iconManual);
              this.setTooltip(statusInfo.tooltip);
            }
            break;
          }
          case "fetching": {
            this.setColorNormal();
            this.setIcon(iconLoading);
            this.setTooltip(statusInfo.tooltip);
            break;
          }
          case "completionResponseSlow":
          case "rateLimitExceeded": {
            this.setColorWarning();
            this.setIcon(iconWarning);
            this.setTooltip(statusInfo.tooltip);
            break;
          }
        }
        break;
      }
    }
  }

  private checkIfVSCodeInlineCompletionEnabled() {
    if (this.config.vscodeInlineSuggestEnabled) {
      return true;
    } else {
      this.setColorWarning();
      this.setIcon(iconDisabled);
      this.setTooltip("Tabby: inline completion is disabled.");
      this.showInformationWhenInlineSuggestDisabled();
      return false;
    }
  }

  private setColorNormal() {
    this.item.color = colorNormal;
    this.item.backgroundColor = backgroundColorNormal;
  }

  private setColorWarning() {
    this.item.color = colorWarning;
    this.item.backgroundColor = backgroundColorWarning;
  }

  private setIcon(icon: string) {
    this.item.text = `${icon} ${label}`;
  }

  private setTooltip(tooltip: string | undefined) {
    this.item.tooltip = tooltip;
  }

  // show notifications only once per session
  private informationWhenInlineSuggestDisabledShown = false;

  private showInformationWhenInlineSuggestDisabled() {
    if (this.informationWhenInlineSuggestDisabledShown) {
      return;
    }
    this.informationWhenInlineSuggestDisabledShown = true;
    window
      .showWarningMessage(
        "Tabby's suggestion is not showing because inline suggestion is disabled. Please enable it first.",
        "Enable",
        "Settings",
      )
      .then(async (selection) => {
        switch (selection) {
          case "Enable":
            await this.config.updateVscodeInlineSuggestEnabled(true);
            break;
          case "Settings":
            commands.executeCommand("workbench.action.openSettings", "@id:editor.inlineSuggest.enabled");
            break;
        }
      });
  }
}
