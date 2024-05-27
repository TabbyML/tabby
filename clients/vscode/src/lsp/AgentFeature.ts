import { EventEmitter } from "events";
import { BaseLanguageClient, StaticFeature, FeatureState, Disposable } from "vscode-languageclient";
import {
  ClientCapabilities,
  AgentServerConfigRequest,
  AgentServerConfigSync,
  ServerConfig,
  AgentStatusRequest,
  AgentStatusSync,
  Status,
  AgentIssuesRequest,
  AgentIssuesSync,
  IssueList,
  AgentIssueDetailRequest,
  IssueDetailParams,
  IssueDetailResult,
} from "tabby-agent";

export class AgentFeature extends EventEmitter implements StaticFeature {
  private disposables: Disposable[] = [];
  private statusValue: Status = "notInitialized";

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
    capabilities.tabby = {
      ...capabilities.tabby,
      agent: true,
    };
  }

  preInitialize(): void {
    // nothing
  }

  initialize(): void {
    this.disposables.push(
      this.client.onNotification(AgentServerConfigSync.type, (params) => {
        this.emit("didChangeServerConfig", params.server);
      }),
    );
    this.disposables.push(
      this.client.onNotification(AgentStatusSync.type, (params) => {
        this.statusValue = params.status;
        this.emit("didChangeStatus", params.status);
      }),
    );
    this.disposables.push(
      this.client.onNotification(AgentIssuesSync.type, (params) => {
        this.emit("didUpdateIssues", params.issues);
      }),
    );
    // schedule a initial status sync
    this.fetchStatus().then((status) => {
      if (status !== this.statusValue) {
        this.statusValue = status;
        this.emit("didChangeStatus", status);
      }
    });
  }

  clear(): void {
    this.disposables.forEach((disposable) => disposable.dispose());
    this.disposables = [];
  }

  get status(): Status {
    return this.statusValue;
  }

  async fetchServerConfig(): Promise<ServerConfig> {
    return this.client.sendRequest(AgentServerConfigRequest.type);
  }

  async fetchStatus(): Promise<Status> {
    return this.client.sendRequest(AgentStatusRequest.type);
  }

  async fetchIssues(): Promise<IssueList> {
    return this.client.sendRequest(AgentIssuesRequest.type);
  }

  async fetchIssueDetail(params: IssueDetailParams): Promise<IssueDetailResult> {
    return this.client.sendRequest(AgentIssueDetailRequest.method, params);
  }
}
