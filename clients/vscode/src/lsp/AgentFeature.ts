import { EventEmitter } from "events";
import deepEqual from "deep-equal";
import { BaseLanguageClient, StaticFeature, FeatureState, Disposable } from "vscode-languageclient";
import {
  ClientCapabilities,
  AgentServerInfoRequest,
  AgentServerInfoSync,
  ServerInfo,
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
  private serverInfo: ServerInfo | undefined = undefined;
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
      this.client.onNotification(AgentServerInfoSync.type, (params) => {
        if (!deepEqual(params.serverInfo, this.serverInfo)) {
          this.serverInfo = params.serverInfo;
          this.emit("didUpdateServerInfo", params.serverInfo);
        }
      }),
    );
    this.disposables.push(
      this.client.onNotification(AgentStatusSync.type, (params) => {
        if (params.status !== this.statusValue) {
          this.statusValue = params.status;
          this.emit("didChangeStatus", params.status);
        }
      }),
    );
    this.disposables.push(
      this.client.onNotification(AgentIssuesSync.type, (params) => {
        this.emit("didUpdateIssues", params.issues);
      }),
    );
    // schedule a initial sync
    this.fetchServerInfo().then((serverInfo) => {
      if (!deepEqual(serverInfo, this.serverInfo)) {
        this.serverInfo = serverInfo;
        this.emit("didUpdateServerInfo", serverInfo);
      }
    });
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

  async fetchServerInfo(): Promise<ServerInfo> {
    return this.client.sendRequest(AgentServerInfoRequest.type);
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
