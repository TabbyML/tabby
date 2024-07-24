import { EventEmitter } from "events";
import { workspace, ExtensionContext, WorkspaceConfiguration, ConfigurationTarget, Memento } from "vscode";
import { ClientProvidedConfig } from "tabby-agent";

interface AdvancedSettings {
  "inlineCompletion.triggerMode"?: "automatic" | "manual";
  "chatEdit.history"?: number;
}

export class Config extends EventEmitter {
  constructor(private readonly context: ExtensionContext) {
    super();
    context.subscriptions.push(
      workspace.onDidChangeConfiguration(async (event) => {
        if (event.affectsConfiguration("tabby")) {
          this.emit("updated");
        }
      }),
    );
  }

  get workspace(): WorkspaceConfiguration {
    return workspace.getConfiguration("tabby");
  }

  get memento(): Memento {
    return this.context.globalState;
  }

  get serverEndpoint(): string {
    return this.workspace.get("endpoint", "");
  }

  set serverEndpoint(value: string) {
    if (value !== this.serverEndpoint) {
      this.workspace.update("endpoint", value, ConfigurationTarget.Global);
    }
  }

  get serverToken(): string {
    return this.memento.get("server.token", "");
  }

  set serverToken(value: string) {
    if (value !== this.serverToken) {
      this.memento.update("server.token", value);
      this.emit("updated");
    }
  }

  get inlineCompletionTriggerMode(): "automatic" | "manual" {
    const advancedSettings = this.workspace.get("settings.advanced", {}) as AdvancedSettings;
    return advancedSettings["inlineCompletion.triggerMode"] || "automatic";
  }

  set inlineCompletionTriggerMode(value: "automatic" | "manual") {
    if (value !== this.inlineCompletionTriggerMode) {
      const advancedSettings = this.workspace.get("settings.advanced", {}) as AdvancedSettings;
      const updatedValue = { ...advancedSettings, "inlineCompletion.triggerMode": value };
      this.workspace.update("settings.advanced", updatedValue, ConfigurationTarget.Global);
      this.emit("updated");
    }
  }

  get chatEditHistory(): number {
    const advancedSettings = this.workspace.get("settings.advanced", {}) as AdvancedSettings;
    const numHistory = advancedSettings["chatEdit.history"] === undefined ? 20 : advancedSettings["chatEdit.history"];
    if (numHistory < 0) {
      return 20;
    } else if (numHistory === 0) {
      return 0;
    } else {
      return numHistory;
    }
  }

  set chatEditHistory(value: number) {
    if (value != this.chatEditHistory) {
      const advancedSettings = this.workspace.get("settings.advanced", {}) as AdvancedSettings;
      const updateValue = { ...advancedSettings, "chatEdit.history": value };
      this.workspace.update("settings.advanced", updateValue, ConfigurationTarget.Global);
      this.emit("updated");
    }
  }

  get inlineCompletionEnabled(): boolean {
    return workspace.getConfiguration("editor").get("inlineSuggest.enabled", true);
  }

  set inlineCompletionEnabled(value: boolean) {
    if (value !== this.inlineCompletionEnabled) {
      workspace.getConfiguration("editor").update("inlineSuggest.enabled", value, ConfigurationTarget.Global);
    }
  }

  get keybindings(): "vscode-style" | "tabby-style" {
    return this.workspace.get("keybindings", "vscode-style");
  }

  get anonymousUsageTrackingDisabled(): boolean {
    return this.workspace.get("config.telemetry", false);
  }

  get mutedNotifications(): string[] {
    return this.memento.get("notifications.muted", []);
  }

  set mutedNotifications(value: string[]) {
    this.memento.update("notifications.muted", value);
    this.emit("updated");
  }

  get chatEditRecentlyCommand(): string[] {
    return this.memento.get("edit.recentlyCommand", []);
  }

  set chatEditRecentlyCommand(value: string[]) {
    this.memento.update("edit.recentlyCommand", value);
  }

  buildClientProvidedConfig(): ClientProvidedConfig {
    return {
      server: {
        endpoint: this.serverEndpoint,
        token: this.serverToken,
      },
      inlineCompletion: {
        triggerMode: this.inlineCompletionTriggerMode == "automatic" ? "auto" : "manual",
      },
      keybindings: this.keybindings == "tabby-style" ? "tabby-style" : "default",
      anonymousUsageTracking: {
        disable: this.anonymousUsageTrackingDisabled,
      },
    };
  }
}
