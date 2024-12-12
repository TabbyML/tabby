import { EventEmitter } from "events";
import { BaseLanguageClient, StaticFeature, FeatureState, Disposable } from "vscode-languageclient";
import {
  ClientCapabilities,
  StatusInfo,
  StatusRequest,
  StatusRequestParams,
  StatusDidChangeNotification,
  StatusShowHelpMessageRequest,
  StatusIgnoredIssuesEditRequest,
  StatusIgnoredIssuesEditParams,
} from "tabby-agent";

export class AgentStatusFeature extends EventEmitter implements StaticFeature {
  private disposables: Disposable[] = [];

  private statusInfo: StatusInfo | undefined = undefined;

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
    tabbyCapabilities.statusDidChangeListener = true;
    capabilities.tabby = tabbyCapabilities;
  }

  preInitialize(): void {
    // nothing
  }

  initialize(): void {
    this.disposables.push(
      this.client.onNotification(StatusDidChangeNotification.type, (params: StatusInfo) => {
        this.statusInfo = params;
        this.emit("didChange", params);
      }),
    );
  }

  clear(): void {
    this.disposables.forEach((disposable) => disposable.dispose());
    this.disposables = [];
  }

  get current(): StatusInfo | undefined {
    return this.statusInfo;
  }

  async fetchAgentStatusInfo(params?: StatusRequestParams | undefined): Promise<StatusInfo> {
    const statusInfo: StatusInfo = await this.client.sendRequest(StatusRequest.method, params || {});
    this.statusInfo = statusInfo;
    return statusInfo;
  }

  async showHelpMessage(): Promise<void> {
    await this.client.sendRequest(StatusShowHelpMessageRequest.method);
  }

  async editIgnoredIssues(params: StatusIgnoredIssuesEditParams): Promise<void> {
    await this.client.sendRequest(StatusIgnoredIssuesEditRequest.method, params);
  }
}
