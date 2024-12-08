import { BaseLanguageClient, StaticFeature, FeatureState, ClientCapabilities } from "vscode-languageclient";
import { DidChangeConfigurationNotification, ClientProvidedConfig } from "tabby-agent";
import { Config } from "../Config";

export class ConfigurationSyncFeature implements StaticFeature {
  constructor(
    private readonly client: BaseLanguageClient,
    private readonly config: Config,
  ) {}

  getState(): FeatureState {
    return { kind: "static" };
  }

  fillInitializeParams() {
    // nothing
  }

  fillClientCapabilities(capabilities: ClientCapabilities): void {
    capabilities.workspace = {
      ...capabilities.workspace,
      didChangeConfiguration: { dynamicRegistration: false },
    };
  }

  preInitialize(): void {
    // nothing
  }

  initialize(): void {
    this.config.on("updated", () => this.sync());
  }

  clear(): void {
    this.config.off("updated", () => this.sync());
  }

  async sync(): Promise<void> {
    const clientProvidedConfig: ClientProvidedConfig = this.config.buildClientProvidedConfig();
    this.client.sendNotification(DidChangeConfigurationNotification.method, { settings: clientProvidedConfig });
  }
}
