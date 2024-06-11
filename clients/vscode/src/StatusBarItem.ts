import { commands, window, ExtensionContext, StatusBarAlignment, ThemeColor } from "vscode";
import { State as LanguageClientState } from "vscode-languageclient";
import { Client } from "./lsp/Client";
import { Config } from "./Config";
import { Issues } from "./Issues";
import { InlineCompletionProvider } from "./InlineCompletionProvider";

const label = "Tabby";
const iconAutomatic = "$(check)";
const iconManual = "$(chevron-right)";
const iconDisabled = "$(x)";
const iconLoading = "$(loading~spin)";
const iconDisconnected = "$(plug)";
const iconUnauthorized = "$(key)";
const iconIssueExist = "$(warning)";
const colorNormal = new ThemeColor("statusBar.foreground");
const colorWarning = new ThemeColor("statusBarItem.warningForeground");
const backgroundColorNormal = new ThemeColor("statusBar.background");
const backgroundColorWarning = new ThemeColor("statusBarItem.warningBackground");

export class StatusBarItem {
  private item = window.createStatusBarItem(StatusBarAlignment.Right);
  private status:
    | "initializing"
    | "automatic"
    | "manual"
    | "disabled"
    | "loading"
    | "disconnected"
    | "unauthorized"
    | "issueExist" = "initializing";

  constructor(
    private readonly context: ExtensionContext,
    private readonly client: Client,
    private readonly config: Config,
    private readonly issues: Issues,
    private readonly inlineCompletionProvider: InlineCompletionProvider,
  ) {
    this.updateStatus();
    this.item.show();
    this.context.subscriptions.push(this.item);

    this.client.languageClient.onDidChangeState(() => this.updateStatus());
    this.client.agent.on("didChangeStatus", () => this.updateStatus());
    this.issues.on("updated", () => this.updateStatus());
    this.inlineCompletionProvider.on("didChangeLoading", () => this.updateStatus());
    this.config.on("updated", () => this.updateStatus());
  }

  updateStatus(): void {
    const languageClientState = this.client.languageClient.state;
    if (languageClientState === LanguageClientState.Stopped) {
      return;
    }
    if (languageClientState === LanguageClientState.Starting) {
      return this.toInitializing();
    }
    // languageClientState === LanguageClientState.Running

    const agentStatus = this.client.agent.status;
    if (agentStatus === "finalized") {
      return;
    }
    if (agentStatus === "notInitialized") {
      return this.toInitializing();
    }
    if (agentStatus === "disconnected") {
      return this.toDisconnected();
    }
    if (agentStatus === "unauthorized") {
      return this.toUnauthorized();
    }
    /// agentStatus === "ready"

    if (this.issues.length > 0) {
      return this.toIssuesExist();
    }
    if (this.inlineCompletionProvider.isLoading) {
      return this.toLoading();
    }
    if (!this.config.inlineCompletionEnabled) {
      return this.toDisabled();
    }
    const triggerMode = this.config.inlineCompletionTriggerMode;
    if (triggerMode === "automatic") {
      return this.toAutomatic();
    }
    if (triggerMode === "manual") {
      return this.toManual();
    }
  }

  private toInitializing() {
    if (this.status === "initializing") {
      return;
    }
    this.status = "initializing";
    this.item.color = colorNormal;
    this.item.backgroundColor = backgroundColorNormal;
    this.item.text = `${iconLoading} ${label}`;
    this.item.tooltip = "Tabby is initializing.";
    this.item.command = {
      title: "",
      command: "tabby.applyCallback",
      arguments: [() => this.showInformationWhenInitializing()],
    };
  }

  private toAutomatic() {
    if (this.status === "automatic") {
      return;
    }
    this.status = "automatic";
    this.item.color = colorNormal;
    this.item.backgroundColor = backgroundColorNormal;
    this.item.text = `${iconAutomatic} ${label}`;
    this.item.tooltip = "Tabby automatic code completion is enabled.";
    this.item.command = {
      title: "",
      command: "tabby.applyCallback",
      arguments: [() => this.showInformationWhenAutomaticTrigger()],
    };
  }

  private toManual() {
    if (this.status === "manual") {
      return;
    }
    this.status = "manual";
    this.item.color = colorNormal;
    this.item.backgroundColor = backgroundColorNormal;
    this.item.text = `${iconManual} ${label}`;
    this.item.tooltip = "Tabby is standing by, click or press `Alt + \\` to trigger code completion.";
    this.item.command = {
      title: "",
      command: "tabby.applyCallback",
      arguments: [() => this.showInformationWhenManualTrigger()],
    };
  }

  private toDisabled() {
    if (this.status === "disabled") {
      return;
    }
    this.status = "disabled";
    this.item.color = colorWarning;
    this.item.backgroundColor = backgroundColorWarning;
    this.item.text = `${iconDisabled} ${label}`;
    this.item.tooltip = "Tabby is disabled. Click to check settings.";
    this.item.command = {
      title: "",
      command: "tabby.applyCallback",
      arguments: [() => this.showInformationWhenInlineSuggestDisabled()],
    };
    this.showInformationWhenInlineSuggestDisabled();
  }

  private toLoading() {
    if (this.status === "loading") {
      return;
    }
    this.status = "loading";
    this.item.color = colorNormal;
    this.item.backgroundColor = backgroundColorNormal;
    this.item.text = `${iconLoading} ${label}`;
    this.item.tooltip = "Tabby is generating code completions.";
    this.item.command = {
      title: "",
      command: "tabby.applyCallback",
      arguments: [() => this.showInformationWhenLoading()],
    };
  }

  private toDisconnected() {
    if (this.status === "disconnected") {
      return;
    }
    this.status = "disconnected";
    this.item.color = colorWarning;
    this.item.backgroundColor = backgroundColorWarning;
    this.item.text = `${iconDisconnected} ${label}`;
    this.item.tooltip = "Cannot connect to Tabby Server. Click to open settings.";
    this.item.command = {
      title: "",
      command: "tabby.applyCallback",
      arguments: [() => this.issues.showHelpMessage("connectionFailed")],
    };
  }

  private toUnauthorized() {
    if (this.status === "unauthorized") {
      return;
    }
    this.status = "unauthorized";
    this.item.color = colorWarning;
    this.item.backgroundColor = backgroundColorWarning;
    this.item.text = `${iconUnauthorized} ${label}`;
    this.item.tooltip = "Tabby Server requires authorization. Please set your personal token.";
    this.item.command = {
      title: "",
      command: "tabby.applyCallback",
      arguments: [() => this.showInformationWhenUnauthorized()],
    };
    this.showInformationWhenUnauthorized();
  }

  private toIssuesExist() {
    if (this.status === "issueExist") {
      return;
    }
    this.status = "issueExist";
    this.item.color = colorWarning;
    this.item.backgroundColor = backgroundColorWarning;
    this.item.text = `${iconIssueExist} ${label}`;
    switch (this.issues.first) {
      case "highCompletionTimeoutRate":
        this.item.tooltip = "Most completion requests timed out.";
        break;
      case "slowCompletionResponseTime":
        this.item.tooltip = "Completion requests appear to take too much time.";
        break;
      default:
        this.item.tooltip = "";
        break;
    }
    this.item.command = {
      title: "",
      command: "tabby.applyCallback",
      arguments: [() => this.issues.showHelpMessage()],
    };
  }

  private showInformationWhenInitializing() {
    window.showInformationMessage("Tabby is initializing.", "Settings").then((selection) => {
      switch (selection) {
        case "Settings":
          commands.executeCommand("tabby.openSettings");
          break;
      }
    });
  }

  private showInformationWhenAutomaticTrigger() {
    window.showInformationMessage("Tabby automatic code completion is enabled.", "Settings").then((selection) => {
      switch (selection) {
        case "Settings":
          commands.executeCommand("tabby.openSettings");
          break;
      }
    });
  }

  private showInformationWhenManualTrigger() {
    window
      .showInformationMessage(
        "Tabby is standing by. Trigger code completion manually?",
        "Trigger",
        "Automatic Mode",
        "Settings",
      )
      .then((selection) => {
        switch (selection) {
          case "Trigger":
            commands.executeCommand("editor.action.inlineSuggest.trigger");
            break;
          case "Automatic Mode":
            commands.executeCommand("tabby.toggleInlineCompletionTriggerMode", "automatic");
            break;
          case "Settings":
            commands.executeCommand("tabby.openSettings");
            break;
        }
      });
  }

  private showInformationWhenLoading() {
    window.showInformationMessage("Tabby is generating code completions.", "Settings").then((selection) => {
      switch (selection) {
        case "Settings":
          commands.executeCommand("tabby.openSettings");
          break;
      }
    });
  }

  private showInformationWhenInlineSuggestDisabled() {
    window
      .showWarningMessage(
        "Tabby's suggestion is not showing because inline suggestion is disabled. Please enable it first.",
        "Enable",
        "Settings",
      )
      .then((selection) => {
        switch (selection) {
          case "Enable":
            this.config.inlineCompletionEnabled = true;
            break;
          case "Settings":
            commands.executeCommand("workbench.action.openSettings", "@id:editor.inlineSuggest.enabled");
            break;
        }
      });
  }

  private showInformationWhenUnauthorized() {
    const message = "Tabby server requires authentication, please set your token.";
    window.showWarningMessage(message, "Set Token...", "I Don't Have a Token", "Settings").then((selection) => {
      switch (selection) {
        case "Set Token...":
          commands.executeCommand("tabby.setApiToken");
          break;
        case "I Don't Have a Token":
          commands.executeCommand("tabby.openOnlineHelp", "/docs/quick-start/register-account");
          break;
        case "Settings":
          commands.executeCommand("tabby.openSettings");
          break;
      }
    });
  }
}
