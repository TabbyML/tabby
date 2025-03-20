import { EventEmitter } from "events";
import { workspace, ExtensionContext, WorkspaceConfiguration, ConfigurationTarget, Memento } from "vscode";
import { ClientProvidedConfig } from "tabby-agent";

interface AdvancedSettings {
  "inlineCompletion.triggerMode"?: "automatic" | "manual";
  "inlineCompletion.disabledLanguages"?: string[];
  "chatEdit.history"?: number;
  useVSCodeProxy?: boolean;
}

export interface ServerRecordValue {
  token: string;
  updatedAt: number;
}
export type ServerRecords = Map<string, ServerRecordValue>;

export class Config extends EventEmitter {
  constructor(private readonly context: ExtensionContext) {
    super();
    context.subscriptions.push(
      workspace.onDidChangeConfiguration(async (event) => {
        if (
          event.affectsConfiguration("tabby") ||
          event.affectsConfiguration("editor.inlineSuggest") ||
          event.affectsConfiguration("http.proxy") ||
          event.affectsConfiguration("https.proxy") ||
          event.affectsConfiguration("http.proxyAuthorization")
        ) {
          this.emit("updated");
        }
      }),
    );
    this.migrateServerRecordFromPastServerConfigs();
  }

  private get workspace(): WorkspaceConfiguration {
    return workspace.getConfiguration("tabby");
  }

  get memento(): Memento {
    return this.context.globalState;
  }

  get serverEndpoint(): string {
    return this.workspace.get("endpoint", "");
  }

  async updateServerEndpoint(value: string) {
    if (value !== this.serverEndpoint) {
      await this.workspace.update("endpoint", value, ConfigurationTarget.Global);
    }
  }

  get serverRecords(): ServerRecords {
    const records = this.memento.get("server.serverRecords", {});
    return new Map(Object.entries(records));
  }

  async updateServerRecords(value: ServerRecords) {
    const obj: Record<string, ServerRecordValue> = {};
    value.forEach((v, k) => {
      obj[k] = v;
    });
    await this.memento.update("server.serverRecords", obj);
    this.emit("updated");
  }

  private async migrateServerRecordFromPastServerConfigs() {
    const pastServerConfigs = this.memento.get("server.pastServerConfigs", []);
    if (pastServerConfigs.length > 0) {
      await this.updateServerRecords(
        pastServerConfigs.reduce((acc: ServerRecords, config) => {
          acc.set(config["endpoint"], {
            token: config["token"],
            updatedAt: Date.now(),
          });
          return acc;
        }, this.serverRecords),
      );
      this.memento.update("server.pastServerConfigs", undefined);
    }
  }

  get inlineCompletionTriggerMode(): "automatic" | "manual" {
    const advancedSettings = this.workspace.get("settings.advanced", {}) as AdvancedSettings;
    return advancedSettings["inlineCompletion.triggerMode"] || "automatic";
  }

  async updateInlineCompletionTriggerMode(value: "automatic" | "manual") {
    if (value !== this.inlineCompletionTriggerMode) {
      const advancedSettings = this.workspace.get("settings.advanced", {}) as AdvancedSettings;
      advancedSettings["inlineCompletion.triggerMode"] = value;
      await this.workspace.update("settings.advanced", advancedSettings, ConfigurationTarget.Global);
    }
  }

  get disabledLanguages(): string[] {
    const advancedSettings = this.workspace.get("settings.advanced", {}) as AdvancedSettings;
    return advancedSettings["inlineCompletion.disabledLanguages"] || [];
  }

  async updateDisabledLanguages(value: string[]) {
    const advancedSettings = this.workspace.get("settings.advanced", {}) as AdvancedSettings;
    advancedSettings["inlineCompletion.disabledLanguages"] = value;
    await this.workspace.update("settings.advanced", advancedSettings, ConfigurationTarget.Global);
    this.emit("updated");
  }

  get useVSCodeProxy(): boolean {
    const advancedSettings = this.workspace.get("settings.advanced", {}) as AdvancedSettings;
    return advancedSettings["useVSCodeProxy"] ?? true;
  }

  get maxChatEditHistory(): number {
    const advancedSettings = this.workspace.get("settings.advanced", {}) as AdvancedSettings;
    const numHistory = advancedSettings["chatEdit.history"] ?? 20;
    return Math.max(0, numHistory);
  }

  get vscodeInlineSuggestEnabled(): boolean {
    return workspace.getConfiguration("editor").get("inlineSuggest.enabled", true);
  }

  async updateVscodeInlineSuggestEnabled(value: boolean) {
    if (value !== this.vscodeInlineSuggestEnabled) {
      await workspace.getConfiguration("editor").update("inlineSuggest.enabled", value, ConfigurationTarget.Global);
    }
  }

  get keybindings(): "vscode-style" | "tabby-style" {
    return this.workspace.get("keybindings", "vscode-style");
  }

  get anonymousUsageTrackingDisabled(): boolean {
    return this.workspace.get("config.telemetry", false);
  }

  get chatEditRecentlyCommand(): string[] {
    return this.memento.get("edit.recentlyCommand", []);
  }

  async updateChatEditRecentlyCommand(value: string[]) {
    await this.memento.update("edit.recentlyCommand", value);
  }

  get httpConfig() {
    return workspace.getConfiguration("http");
  }

  get proxyAuthorization() {
    return this.httpConfig.get("authorization", "");
  }

  get proxyUrl() {
    const https = workspace.getConfiguration("https");
    const httpsProxy = https.get("proxy", "");
    const httpProxy = this.httpConfig.get("proxy", "");

    return httpsProxy || httpProxy;
  }

  buildClientProvidedConfig(): ClientProvidedConfig {
    const url = this.useVSCodeProxy ? this.proxyUrl : "";
    const authorization = this.useVSCodeProxy ? this.proxyAuthorization : "";

    return {
      // Note: current we only support http.proxy | http.authorization
      // More properties we will land later.
      proxy: {
        url,
        authorization,
      },
      server: {
        endpoint: this.serverEndpoint,
        token: this.serverEndpoint ? this.serverRecords.get(this.serverEndpoint)?.token : undefined,
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
