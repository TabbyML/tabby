import { EventEmitter } from "events";
import { Connection, ShowMessageRequest, ShowMessageRequestParams, MessageType } from "vscode-languageserver";
import {
  ClientCapabilities,
  ServerCapabilities,
  StatusInfo,
  StatusRequest,
  StatusRequestParams,
  StatusDidChangeNotification,
  StatusShowHelpMessageRequest,
  StatusIgnoredIssuesEditRequest,
  StatusIgnoredIssuesEditParams,
  StatusIssuesName,
  InlineCompletionTriggerMode,
} from "./protocol";
import type { Feature } from "./feature";
import { getLogger } from "../logger";
import type { AgentIssue } from "../Agent";
import { TabbyAgent } from "../TabbyAgent";
import "../ArrayExt";

export class StatusProvider extends EventEmitter implements Feature {
  private readonly logger = getLogger("StatusProvider");
  // FIXME(@icycodes): extract http client status
  private isConnecting: boolean = false;
  private isFetching: boolean = false;
  private clientInlineCompletionTriggerMode?: InlineCompletionTriggerMode;

  constructor(private readonly agent: TabbyAgent) {
    super();
    this.agent.on("connectingStateUpdated", async (isConnecting: boolean) => {
      this.isConnecting = isConnecting;
      this.update();
    });
    this.agent.on("fetchingStateUpdated", async (isFetching: boolean) => {
      this.isFetching = isFetching;
      this.update();
    });
    this.agent.on("statusChanged", async () => {
      this.update();
    });
    this.agent.on("issuesUpdated", async () => {
      this.update();
    });
  }

  private async update() {
    const status = await this.getStatus({});
    this.emit("updated", status);
  }

  // FIXME(@icycodes): move to listen to config
  setClientInlineCompletionTriggerMode(triggerMode: InlineCompletionTriggerMode): void {
    this.clientInlineCompletionTriggerMode = triggerMode;
  }

  setup(connection: Connection, clientCapabilities: ClientCapabilities): ServerCapabilities {
    connection.onRequest(StatusRequest.type, async (params) => {
      return this.getStatus(params);
    });
    connection.onRequest(StatusShowHelpMessageRequest.type, async () => {
      return this.showStatusHelpMessage(connection);
    });
    connection.onRequest(StatusIgnoredIssuesEditRequest.type, async (params) => {
      return this.editStatusIgnoredIssues(params);
    });
    if (clientCapabilities.tabby?.statusDidChangeListener) {
      this.on("updated", (status: StatusInfo) => {
        connection.sendNotification(StatusDidChangeNotification.type, status);
      });
    }
    return {};
  }

  async getStatus(params: StatusRequestParams): Promise<StatusInfo> {
    if (params.recheckConnection) {
      await this.agent.healthCheck();
    }
    const statusInfo = this.buildStatusInfo();
    this.fillToolTip(statusInfo);
    return statusInfo;
  }

  async showStatusHelpMessage(connection: Connection): Promise<boolean | null> {
    let params: ShowMessageRequestParams;
    let issue: StatusIssuesName | undefined = undefined;

    const detail = this.agent.getIssueDetail({ index: 0 });
    if (detail?.name === "connectionFailed") {
      params = {
        type: MessageType.Error,
        message: "Connect to Server Failed.\n" + this.buildHelpMessage(detail, "plaintext") ?? "",
        actions: [{ title: "OK" }],
      };
    } else if (detail?.name === "highCompletionTimeoutRate" || detail?.name === "slowCompletionResponseTime") {
      params = {
        type: MessageType.Info,
        message: this.buildHelpMessage(detail, "plaintext") ?? "",
        actions: [{ title: "OK" }, { title: "Never Show Again" }],
      };
      issue = "completionResponseSlow";
    } else {
      return false;
    }
    const helpMessage = this.buildHelpMessage(detail, "plaintext");
    if (!helpMessage) {
      return false;
    }
    const result = await connection.sendRequest(ShowMessageRequest.type, params);
    switch (result?.title) {
      case "Never Show Again":
        if (issue) {
          await this.editStatusIgnoredIssues({ operation: "add", issues: [issue] });
          await this.update();
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
    const dataStore = this.agent.getDataStore();
    switch (params.operation) {
      case "add":
        if (dataStore) {
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
        break;
      case "remove":
        if (dataStore) {
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
        break;
      case "removeAll":
        if (dataStore) {
          dataStore.data.statusIgnoredIssues = [];
          this.logger.debug("Removing all ignored issues.");
          await dataStore.save();
          return true;
        }
        break;
      default:
        break;
    }
    return false;
  }

  buildHelpMessage(issueDetail: AgentIssue, format?: "plaintext" | "markdown" | "html"): string | undefined {
    const outputFormat = format ?? "plaintext";

    // "connectionFailed"
    if (issueDetail.name == "connectionFailed") {
      if (outputFormat == "html") {
        return issueDetail.message?.replace(/\n/g, "<br/>");
      } else {
        return issueDetail.message;
      }
    }

    // "slowCompletionResponseTime" or "highCompletionTimeoutRate"
    let statsMessage = "";
    if (issueDetail.name == "slowCompletionResponseTime") {
      const stats = issueDetail.completionResponseStats;
      if (stats && stats["responses"] && stats["averageResponseTime"]) {
        statsMessage = `The average response time of recent ${stats["responses"]} completion requests is ${Number(
          stats["averageResponseTime"],
        ).toFixed(0)}ms.<br/><br/>`;
      }
    }

    if (issueDetail.name == "highCompletionTimeoutRate") {
      const stats = issueDetail.completionResponseStats;
      if (stats && stats["total"] && stats["timeouts"]) {
        statsMessage = `${stats["timeouts"]} of ${stats["total"]} completion requests timed out.<br/><br/>`;
      }
    }

    let helpMessageForRunningLargeModelOnCPU = "";
    const serverHealthState = this.agent.getServerHealthState();
    if (serverHealthState?.device === "cpu" && serverHealthState?.model?.match(/[0-9.]+B$/)) {
      helpMessageForRunningLargeModelOnCPU +=
        `Your Tabby server is running model <i>${serverHealthState?.model}</i> on CPU. ` +
        "This model may be performing poorly due to its large parameter size, please consider trying smaller models or switch to GPU. " +
        "You can find a list of recommend models in the <a href='https://tabby.tabbyml.com/'>online documentation</a>.<br/>";
    }
    let commonHelpMessage = "";
    if (helpMessageForRunningLargeModelOnCPU.length == 0) {
      commonHelpMessage += `<li>The running model <i>${
        serverHealthState?.model ?? ""
      }</i> may be performing poorly due to its large parameter size. `;
      commonHelpMessage +=
        "Please consider trying smaller models. You can find a list of recommend models in the <a href='https://tabby.tabbyml.com/'>online documentation</a>.</li>";
    }
    const host = new URL(this.agent.getConfig().server.endpoint ?? "http://localhost:8080").host;
    if (!(host.startsWith("localhost") || host.startsWith("127.0.0.1") || host.startsWith("0.0.0.0"))) {
      commonHelpMessage += "<li>A poor network connection. Please check your network and proxy settings.</li>";
      commonHelpMessage += "<li>Server overload. Please contact your Tabby server administrator for assistance.</li>";
    }
    let helpMessage = "";
    if (helpMessageForRunningLargeModelOnCPU.length > 0) {
      helpMessage += helpMessageForRunningLargeModelOnCPU + "<br/>";
      if (commonHelpMessage.length > 0) {
        helpMessage += "Other possible causes of this issue: <br/><ul>" + commonHelpMessage + "</ul>";
      }
    } else {
      // commonHelpMessage should not be empty here
      helpMessage += "Possible causes of this issue: <br/><ul>" + commonHelpMessage + "</ul>";
    }

    if (outputFormat == "html") {
      return statsMessage + helpMessage;
    }
    if (outputFormat == "markdown") {
      return (statsMessage + helpMessage)
        .replace(/<br\/>/g, " \n")
        .replace(/<i>(.*?)<\/i>/g, "*$1*")
        .replace(/<a\s+(?:[^>]*?\s+)?href=["']([^"']+)["'][^>]*>([^<]+)<\/a>/g, "[$2]($1)")
        .replace(/<ul[^>]*>(.*?)<\/ul>/g, "$1")
        .replace(/<li[^>]*>(.*?)<\/li>/g, "- $1 \n");
    } else {
      return (statsMessage + helpMessage)
        .replace(/<br\/>/g, " \n")
        .replace(/<i>(.*?)<\/i>/g, "$1")
        .replace(/<a[^>]*>(.*?)<\/a>/g, "$1")
        .replace(/<ul[^>]*>(.*?)<\/ul>/g, "$1")
        .replace(/<li[^>]*>(.*?)<\/li>/g, "- $1 \n");
    }
  }

  private buildStatusInfo(): StatusInfo {
    const agentStatus = this.agent.getStatus();
    switch (agentStatus) {
      case "notInitialized":
        return {
          status: this.isConnecting ? "connecting" : "notInitialized",
        };
      case "finalized":
        return {
          status: "finalized",
        };
      case "unauthorized":
        return {
          status: this.isConnecting ? "connecting" : "unauthorized",
        };
      case "disconnected":
        return {
          status: this.isConnecting ? "connecting" : "disconnected",
          command: {
            title: "Detail",
            command: "tabby/status/showHelpMessage",
            arguments: [{}],
          },
        };
      case "ready": {
        const serverHealth = this.agent.getServerHealthState();
        let currentIssue: StatusIssuesName | null = null;
        const agentIssue = this.agent.getIssues();
        if (
          (agentIssue.length > 0 && agentIssue[0] === "highCompletionTimeoutRate") ||
          agentIssue[0] === "slowCompletionResponseTime"
        ) {
          currentIssue = "completionResponseSlow";
        }
        const dataStore = this.agent.getDataStore();
        const ignored = dataStore?.data.statusIgnoredIssues ?? [];
        if (currentIssue && !ignored.includes(currentIssue)) {
          return {
            status: "completionResponseSlow",
            serverHealth: serverHealth ?? undefined,
            command: {
              title: "Detail",
              command: "tabby/status/showHelpMessage",
              arguments: [{}],
            },
          };
        }
        if (this.isFetching) {
          return {
            status: "fetching",
            serverHealth: serverHealth ?? undefined,
          };
        }
        switch (this.clientInlineCompletionTriggerMode) {
          case "auto":
            return {
              status: "readyForAutoTrigger",
              serverHealth: serverHealth ?? undefined,
            };
          case "manual":
            return {
              status: "readyForManualTrigger",
              serverHealth: serverHealth ?? undefined,
            };
          default:
            return {
              status: "ready",
              serverHealth: serverHealth ?? undefined,
            };
        }
      }
      default:
        return {
          status: "notInitialized",
        };
    }
  }

  private fillToolTip(statusInfo: StatusInfo) {
    switch (statusInfo.status) {
      case "notInitialized":
        statusInfo.tooltip = "Tabby: Initializing";
        break;
      case "finalized":
        statusInfo.tooltip = "Tabby";
        break;
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
      default:
        break;
    }
  }
}
