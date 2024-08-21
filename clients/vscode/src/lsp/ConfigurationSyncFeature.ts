import { BaseLanguageClient, StaticFeature, FeatureState, ClientCapabilities } from "vscode-languageclient";
import { DidChangeConfigurationNotification, ClientProvidedConfig, InitializeParams } from "tabby-agent";
import { Config } from "../Config";

export class ConfigurationSyncFeature implements StaticFeature {
  constructor(
    private readonly client: BaseLanguageClient,
    private readonly config: Config,
  ) {}

  getState(): FeatureState {
    return { kind: "static" };
  }

  fillInitializeParams(params: InitializeParams) {
    params.initializationOptions = {
      ...params.initializationOptions,
      settings: this.config.buildClientProvidedConfig(),
    };
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
    this.config.on("updated", this.listener);
  }

  clear(): void {
    this.config.off("updated", this.listener);
  }

  private listener = () => {
    const clientProvidedConfig: ClientProvidedConfig = this.config.buildClientProvidedConfig();
    this.client.sendNotification(DidChangeConfigurationNotification.method, { settings: clientProvidedConfig });
  };
}
