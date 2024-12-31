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

export class StatusProvider extends EventEmitter implements Feature {
  private readonly logger = getLogger("StatusProvider");

  private lspConnection: Connection | undefined = undefined;
  private clientCapabilities: ClientCapabilities | undefined = undefined;

  constructor(
    private readonly dataStore: DataStore,
    private readonly configurations: Configurations,
    private readonly tabbyApiClient: TabbyApiClient,
  ) {
    super();
  }

  initialize(connection: Connection, clientCapabilities: ClientCapabilities): ServerCapabilities {
    this.lspConnection = connection;
    this.clientCapabilities = clientCapabilities;

    connection.onRequest(StatusRequest.type, async (params) => {
      if (params?.recheckConnection) {
        await this.configurations.refreshClientProvidedConfig();
        await this.tabbyApiClient.connect({ reset: true });
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
    this.tabbyApiClient.on("isFetchingCompletionUpdated", async () => {
      this.notify();
    });
    this.tabbyApiClient.on("hasCompletionResponseTimeIssueUpdated", async () => {
      this.notify();
    });
    this.tabbyApiClient.on("isRateLimitExceededUpdated", async () => {
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
      const statusInfo = await this.buildStatusInfo();
      connection.sendNotification(StatusDidChangeNotification.type, statusInfo);
    }
  }

  private async notify() {
    const statusInfo = await this.buildStatusInfo();
    this.emit("updated", statusInfo);
  }

  async showStatusHelpMessage(): Promise<boolean | null> {
    let params: ShowMessageRequestParams;
    let issue: StatusIssuesName | undefined = undefined;

    const connection = this.lspConnection;
    if (!connection) {
      return null;
    }

    const message = this.tabbyApiClient.getHelpMessage();
    if (!message) {
      return false;
    }

    if (this.tabbyApiClient.getStatus() === "noConnection") {
      params = {
        type: MessageType.Error,
        message,
        actions: [{ title: "OK" }],
      };
    } else if (this.tabbyApiClient.hasCompletionResponseTimeIssue()) {
      params = {
        type: MessageType.Info,
        message,
        actions: [{ title: "OK" }, { title: "Never Show Again" }],
      };
      issue = "completionResponseSlow";
    } else {
      return false;
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

  async editStatusIgnoredIssues(params: StatusIgnoredIssuesEditParams): Promise<boolean> {
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
    switch (apiClientStatus) {
      case "noConnection":
        statusInfo = { status: this.tabbyApiClient.isConnecting() ? "connecting" : "disconnected" };
        break;
      case "unauthorized":
        statusInfo = { status: this.tabbyApiClient.isConnecting() ? "connecting" : "unauthorized" };
        break;
      case "ready":
        {
          const ignored = this.dataStore.data.statusIgnoredIssues ?? [];
          if (this.tabbyApiClient.isRateLimitExceeded()) {
            statusInfo = { status: "rateLimitExceeded" };
          } else if (
            this.tabbyApiClient.hasCompletionResponseTimeIssue() &&
            !ignored.includes("completionResponseSlow")
          ) {
            statusInfo = { status: "completionResponseSlow" };
          } else if (this.tabbyApiClient.isFetchingCompletion()) {
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
    this.fillToolTip(statusInfo);
    statusInfo.serverHealth = this.tabbyApiClient.getServerHealth();
    const hasHelpMessage = this.tabbyApiClient.hasHelpMessage();
    statusInfo.command = hasHelpMessage
      ? {
          title: "Detail",
          command: "tabby/status/showHelpMessage",
          arguments: [{}],
        }
      : undefined;
    statusInfo.helpMessage =
      hasHelpMessage && options.includeHelpMessage ? this.tabbyApiClient.getHelpMessage() : undefined;
    return statusInfo;
  }

  private fillToolTip(statusInfo: StatusInfo) {
    switch (statusInfo.status) {
      case "connecting":
        statusInfo.tooltip = "Tabby: Connecting to Server...";
        break;
      case "unauthorized":
        statusInfo.tooltip = "Tabby: Authorization Required";
        break;
      case "disconnected":
        statusInfo.tooltip = "Tabby: Connect to Server Failed";
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
      case "completionResponseSlow":
        statusInfo.tooltip = "Tabby: Slow Completion Response Detected";
        break;
      case "rateLimitExceeded":
        statusInfo.tooltip = "Tabby: Too Many Requests";
        break;
      default:
        break;
    }
  }
}
