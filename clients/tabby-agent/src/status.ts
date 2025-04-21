import type { Connection } from "vscode-languageserver";
import type { Feature } from "./feature";
import type { DataStore, StoredData } from "./dataStore";
import type { Configurations } from "./config";
import type { TabbyApiClient } from "./http/tabbyApiClient";
import { EventEmitter } from "events";
import { ShowMessageRequest, ShowMessageRequestParams, MessageType } from "vscode-languageserver";
import deepEqual from "deep-equal";
import {
  ClientCapabilities,
  ServerCapabilities,
  ClientProvidedConfig,
  StatusInfo,
  StatusRequest,
  StatusDidChangeNotification,
  StatusShowHelpMessageRequest,
  StatusIgnoredIssuesEditRequest,
  StatusIgnoredIssuesEditParams,
  StatusIssuesName,
} from "./protocol";
import { getLogger } from "./logger";
import "./utils/array";
import { CompletionProvider } from "./codeCompletion";

export class StatusProvider extends EventEmitter implements Feature {
  private readonly logger = getLogger("StatusProvider");

  private lspConnection: Connection | undefined = undefined;
  private clientCapabilities: ClientCapabilities | undefined = undefined;

  constructor(
    private readonly dataStore: DataStore,
    private readonly configurations: Configurations,
    private readonly tabbyApiClient: TabbyApiClient,
    private readonly completionProvider: CompletionProvider,
  ) {
    super();
  }

  initialize(connection: Connection, clientCapabilities: ClientCapabilities): ServerCapabilities {
    this.lspConnection = connection;
    this.clientCapabilities = clientCapabilities;

    connection.onRequest(StatusRequest.type, async (params) => {
      if (params?.recheckConnection) {
        await this.configurations.refreshClientProvidedConfig();
        await this.tabbyApiClient.connect();
      }
      return this.buildStatusInfo({ includeHelpMessage: true });
    });
    connection.onRequest(StatusShowHelpMessageRequest.type, async () => {
      return this.showStatusHelpMessage();
    });
    connection.onRequest(StatusIgnoredIssuesEditRequest.type, async (params) => {
      return this.editStatusIgnoredIssues(params);
    });
    if (clientCapabilities.tabby?.statusDidChangeListener) {
      this.on("updated", (status: StatusInfo) => {
        connection.sendNotification(StatusDidChangeNotification.type, status);
      });
    }

    this.tabbyApiClient.on("statusUpdated", async () => {
      this.notify();
    });
    this.tabbyApiClient.on("isConnectingUpdated", async () => {
      this.notify();
    });
    this.completionProvider.on("isAvailableUpdated", async () => {
      this.notify();
    });
    this.completionProvider.on("latencyIssueUpdated", async () => {
      this.notify();
    });
    this.completionProvider.on("isRateLimitExceededUpdated", async () => {
      this.notify();
    });
    this.completionProvider.on("isFetchingUpdated", async () => {
      this.notify();
    });

    this.configurations.on(
      "clientProvidedConfigUpdated",
      (config: ClientProvidedConfig, oldConfig: ClientProvidedConfig) => {
        if (config.inlineCompletion?.triggerMode !== oldConfig.inlineCompletion?.triggerMode) {
          this.notify();
        }
      },
    );

    this.dataStore.on("updated", async (data: Partial<StoredData>, old: Partial<StoredData>) => {
      if (!deepEqual(data.statusIgnoredIssues, old.statusIgnoredIssues)) {
        this.notify();
      }
    });

    return {};
  }

  async initialized(connection: Connection): Promise<void> {
    if (this.clientCapabilities?.tabby?.statusDidChangeListener) {
      const statusInfo = this.buildStatusInfo();
      connection.sendNotification(StatusDidChangeNotification.type, statusInfo);
    }
  }

  private async notify() {
    const statusInfo = this.buildStatusInfo();
    this.emit("updated", statusInfo);
  }

  async showStatusHelpMessage(): Promise<boolean | null> {
    const connection = this.lspConnection;
    if (!connection) {
      return null;
    }

    let params: ShowMessageRequestParams;
    let issue: StatusIssuesName | undefined = undefined;

    const statusInfo = this.buildStatusInfo();
    switch (statusInfo.status) {
      case "disconnected":
        {
          const message = this.tabbyApiClient.getHelpMessage();
          if (!message) {
            return false;
          }
          params = {
            type: MessageType.Error,
            message,
            actions: [{ title: "OK" }],
          };
        }
        break;
      case "codeCompletionNotAvailable":
      case "rateLimitExceeded":
        {
          const message = this.completionProvider.getHelpMessage();
          if (!message) {
            return false;
          }
          params = {
            type: MessageType.Error,
            message,
            actions: [{ title: "OK" }],
          };
        }
        break;
      case "completionResponseSlow":
        {
          const message = this.completionProvider.getHelpMessage();
          if (!message) {
            return false;
          }
          params = {
            type: MessageType.Info,
            message,
            actions: [{ title: "OK" }, { title: "Never Show Again" }],
          };
          issue = "completionResponseSlow";
        }
        break;
      default:
        return false;
        break;
    }

    const result = await connection.sendRequest(ShowMessageRequest.type, params);
    switch (result?.title) {
      case "Never Show Again":
        if (issue) {
          await this.editStatusIgnoredIssues({ operation: "add", issues: [issue] });
          await this.notify();
        }
        break;
      case "OK":
        break;
      default:
        break;
    }
    return true;
  }

  private async editStatusIgnoredIssues(params: StatusIgnoredIssuesEditParams): Promise<boolean> {
    const issues = Array.isArray(params.issues) ? params.issues : [params.issues];
    const dataStore = this.dataStore;
    switch (params.operation) {
      case "add": {
        const current = dataStore.data.statusIgnoredIssues ?? [];
        dataStore.data.statusIgnoredIssues = current.concat(issues).distinct();
        this.logger.debug(
          "Adding ignored issues: [" +
            current.join(",") +
            "] -> [" +
            dataStore.data.statusIgnoredIssues.join(",") +
            "].",
        );
        await dataStore.save();
        return true;
      }
      case "remove": {
        const current = dataStore.data.statusIgnoredIssues ?? [];
        dataStore.data.statusIgnoredIssues = current.filter((item) => !issues.includes(item));
        this.logger.debug(
          "Removing ignored issues: [" +
            current.join(",") +
            "] -> [" +
            dataStore.data.statusIgnoredIssues.join(",") +
            "].",
        );

        await dataStore.save();
        return true;
      }
      case "removeAll": {
        dataStore.data.statusIgnoredIssues = [];
        this.logger.debug("Removing all ignored issues.");
        await dataStore.save();
        return true;
      }
      default:
        break;
    }
    return false;
  }

  private buildStatusInfo(options: { includeHelpMessage?: boolean } = {}): StatusInfo {
    let statusInfo: StatusInfo;
    const apiClientStatus = this.tabbyApiClient.getStatus();
    if (this.tabbyApiClient.isConnecting()) {
      statusInfo = { status: "connecting" };
    } else {
      switch (apiClientStatus) {
        case "noConnection":
          statusInfo = { status: "disconnected" };
          break;
        case "unauthorized":
          statusInfo = { status: "unauthorized" };
          break;
        case "ready":
          {
            const ignored = this.dataStore.data.statusIgnoredIssues ?? [];
            if (!this.completionProvider.isAvailable()) {
              statusInfo = { status: "codeCompletionNotAvailable" };
            } else if (this.completionProvider.isRateLimitExceeded()) {
              statusInfo = { status: "rateLimitExceeded" };
            } else if (
              this.completionProvider.getLatencyIssue() != undefined &&
              !ignored.includes("completionResponseSlow")
            ) {
              statusInfo = { status: "completionResponseSlow" };
            } else if (this.completionProvider.isFetching()) {
              statusInfo = { status: "fetching" };
            } else {
              switch (this.configurations.getClientProvidedConfig().inlineCompletion?.triggerMode) {
                case "auto":
                  statusInfo = { status: "readyForAutoTrigger" };
                  break;
                case "manual":
                  statusInfo = { status: "readyForManualTrigger" };
                  break;
                default:
                  statusInfo = { status: "ready" };
                  break;
              }
            }
          }
          break;
      }
    }
    let hasHelpMessage = false;
    switch (statusInfo.status) {
      case "connecting":
        statusInfo.tooltip = "Tabby: Connecting to Server...";
        break;
      case "unauthorized":
        statusInfo.tooltip = "Tabby: Authorization Required";
        break;
      case "disconnected":
        statusInfo.tooltip = "Tabby: Connect to Server Failed";
        hasHelpMessage = true;
        break;
      case "ready":
        statusInfo.tooltip = "Tabby: Code Completion Enabled";
        break;
      case "readyForAutoTrigger":
        statusInfo.tooltip = "Tabby: Automatic Code Completion Enabled";
        break;
      case "readyForManualTrigger":
        statusInfo.tooltip = "Tabby: Manual Code Completion Enabled";
        break;
      case "fetching":
        statusInfo.tooltip = "Tabby: Generating Completions...";
        break;
      case "codeCompletionNotAvailable":
        statusInfo.tooltip = "Tabby: Code Completion Not Available";
        hasHelpMessage = true;
        break;
      case "rateLimitExceeded":
        statusInfo.tooltip = "Tabby: Too Many Requests";
        hasHelpMessage = true;
        break;
      case "completionResponseSlow":
        statusInfo.tooltip = "Tabby: Slow Completion Response Detected";
        hasHelpMessage = true;
        break;
      default:
        break;
    }
    statusInfo.serverHealth = this.tabbyApiClient.getServerHealth();
    if (hasHelpMessage) {
      statusInfo.command = {
        title: "Detail",
        command: "tabby/status/showHelpMessage",
        arguments: [{}],
      };

      if (options.includeHelpMessage) {
        statusInfo.helpMessage = this.tabbyApiClient.getHelpMessage() ?? this.completionProvider.getHelpMessage();
      }
    }

    return statusInfo;
  }
}
