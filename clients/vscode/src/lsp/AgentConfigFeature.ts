import { EventEmitter } from "events";
import { BaseLanguageClient, StaticFeature, FeatureState, Disposable } from "vscode-languageclient";
import { ClientCapabilities, Config as AgentConfig, ConfigRequest, ConfigDidChangeNotification } from "tabby-agent";

export class AgentConfigFeature extends EventEmitter implements StaticFeature {
  private disposables: Disposable[] = [];

  private config: AgentConfig | undefined = undefined;

  constructor(private readonly client: BaseLanguageClient) {
    super();
  }

  getState(): FeatureState {
    return { kind: "static" };
  }

  fillInitializeParams() {
    // nothing
  }

  fillClientCapabilities(capabilities: ClientCapabilities): void {
    const tabbyCapabilities = capabilities.tabby || {};
    tabbyCapabilities.configDidChangeListener = true;
    capabilities.tabby = tabbyCapabilities;
  }

  preInitialize(): void {
    // nothing
  }

  initialize(): void {
    this.disposables.push(
      this.client.onNotification(ConfigDidChangeNotification.type, (params: AgentConfig) => {
        this.config = params;
        this.emit("didChange", params);
      }),
    );
  }

  clear(): void {
    this.disposables.forEach((disposable) => disposable.dispose());
    this.disposables = [];
  }

  get current(): AgentConfig | undefined {
    return this.config;
  }

  async fetchAgentConfig(): Promise<AgentConfig> {
    const agentConfig: AgentConfig = await this.client.sendRequest(ConfigRequest.method, {});
    this.config = agentConfig;
    return agentConfig;
  }
}
