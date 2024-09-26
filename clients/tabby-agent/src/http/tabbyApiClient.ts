import type { paths as TabbyApi, components as TabbyApiComponents } from "tabby-openapi/compatible";
import type { ParseAs } from "openapi-fetch";
import type { Readable } from "readable-stream";
import type { Configurations } from "../config";
import type { AnonymousUsageLogger } from "../telemetry";
import type { ConfigData } from "../config/type";
import type { ProxyConfig } from "./proxy";
import type { ClientInfo } from "../protocol";
import type { CompletionStats } from "../codeCompletion/statistics";
import { EventEmitter } from "events";
import createClient from "openapi-fetch";
import { v4 as uuid } from "uuid";
import deepEqual from "deep-equal";
import * as semver from "semver";
import { name as agentName, version as agentVersion } from "../../package.json";
import { isBrowser } from "../env";
import { createProxyForUrl } from "./proxy";
import { getLogger } from "../logger";
import { isBlank } from "../utils/string";
import { readChatStream } from "./stream";
import { abortSignalFromAnyOf } from "../utils/signal";
import {
  errorToString,
  HttpError,
  MutexAbortError,
  isUnauthorizedError,
  isCanceledError,
  isTimeoutError,
} from "../utils/error";
import { RequestStats } from "./statistics";

export type TabbyApiClientStatus = "noConnection" | "unauthorized" | "ready";

export type TabbyServerProvidedConfig = {
  disable_client_side_telemetry?: boolean;
};

export class TabbyApiClient extends EventEmitter {
  private readonly logger = getLogger("TabbyApiClient");

  private userAgentString: string | undefined = undefined;
  private api: ReturnType<typeof createClient<TabbyApi>> | undefined;
  private endpoint: string | undefined = undefined;

  private status: TabbyApiClientStatus = "noConnection";
  private connecting: boolean = false;
  private fetchingCompletion: boolean = false;

  private readonly completionRequestStats = new RequestStats();
  private completionResponseIssue: "highTimeoutRate" | "slowResponseTime" | undefined = undefined;

  private connectionErrorMessage: string | undefined = undefined;
  private serverHealth: TabbyApiComponents["schemas"]["HealthState"] | undefined = undefined;

  private healthCheckMutexAbortController: AbortController | undefined = undefined;

  private reconnectTimer: ReturnType<typeof setInterval> | undefined = undefined;

  constructor(
    private readonly configurations: Configurations,
    private readonly anonymousUsageLogger: AnonymousUsageLogger,
  ) {
    super();
  }

  async initialize(clientInfo: ClientInfo | undefined) {
    this.userAgentString = this.buildUserAgentString(clientInfo);
    this.api = this.createApiClient();
    this.connect(); // no await

    this.configurations.on("updated", (config: ConfigData, oldConfig: ConfigData) => {
      const isServerConnectionChanged = !(
        deepEqual(config.server, oldConfig.server) && deepEqual(config.proxy, oldConfig.proxy)
      );
      if (isServerConnectionChanged) {
        this.logger.debug("Server configurations updated, reconnecting...");
        this.updateStatus("noConnection");
        this.updateCompletionResponseIssue(undefined);
        this.connectionErrorMessage = undefined;
        this.serverHealth = undefined;
        this.completionRequestStats.reset();
        this.api = this.createApiClient();
        this.connect(); // no await
      }
    });

    const reconnectInterval = 1000 * 30; // 30s
    this.reconnectTimer = setInterval(async () => {
      if (this.status === "noConnection" || this.status === "unauthorized") {
        this.logger.debug("Trying to reconnect...");
        await this.connect();
      }
    }, reconnectInterval);
  }

  async shutdown() {
    if (this.reconnectTimer) {
      clearInterval(this.reconnectTimer);
    }
  }

  private buildUserAgentString(clientInfo: ClientInfo | undefined): string {
    const envInfo = isBrowser ? navigator?.userAgent : `Node.js/${process.version}`;
    const tabbyAgentInfo = `${agentName}/${agentVersion}`;
    const ideName = clientInfo?.name.replace(/ /g, "-");
    const ideVersion = clientInfo?.version;
    const ideInfo = ideName ? `${ideName}/${ideVersion}` : "";
    const tabbyPluginName = clientInfo?.tabbyPlugin?.name;
    const tabbyPluginVersion = clientInfo?.tabbyPlugin?.version;
    const tabbyPluginInfo = tabbyPluginName ? `${tabbyPluginName}/${tabbyPluginVersion}` : "";
    return `${envInfo} ${tabbyAgentInfo} ${ideInfo} ${tabbyPluginInfo}`.trim();
  }

  private createApiClient() {
    const config = this.configurations.getMergedConfig();
    const endpoint = config.server.endpoint;
    this.endpoint = endpoint;
    const auth = !isBlank(config.server.token) ? `Bearer ${config.server.token}` : undefined;
    const proxyConfigs: ProxyConfig[] = [{ fromEnv: true }];
    if (!isBlank(config.proxy.url)) {
      proxyConfigs.unshift(config.proxy);
    }
    return createClient<TabbyApi>({
      baseUrl: endpoint,
      headers: {
        Authorization: auth,
        "User-Agent": this.userAgentString,
        ...config.server.requestHeaders,
      },
      /** dispatcher do not exist in {@link RequestInit} in browser env. */
      /* @ts-expect-error TS-2353 */
      dispatcher: createProxyForUrl(endpoint, proxyConfigs),
    });
  }

  private createTimeOutAbortSignal(): AbortSignal {
    const config = this.configurations.getMergedConfig();
    const timeout = Math.min(0x7fffffff, config.server.requestTimeout);
    return AbortSignal.timeout(timeout);
  }

  private updateStatus(status: TabbyApiClientStatus) {
    if (this.status != status) {
      this.status = status;
      this.logger.info(`Status updated: ${status}.`);
      this.emit("statusUpdated", status);
    }
  }

  private updateIsConnecting(isConnecting: boolean) {
    if (this.connecting != isConnecting) {
      this.connecting = isConnecting;
      this.emit("isConnectingUpdated", isConnecting);
    }
  }

  private updateIsFetchingCompletion(isFetchingCompletion: boolean) {
    if (this.fetchingCompletion != isFetchingCompletion) {
      this.fetchingCompletion = isFetchingCompletion;
      this.emit("isFetchingCompletionUpdated", isFetchingCompletion);
    }
  }

  private updateCompletionResponseIssue(issue: "highTimeoutRate" | "slowResponseTime" | undefined) {
    if (this.completionResponseIssue != issue) {
      this.completionResponseIssue = issue;
      if (issue) {
        this.logger.info(`Completion response issue detected: ${issue}.`);
      }
      this.emit("hasCompletionResponseTimeIssueUpdated", !!issue);
    }
  }

  getCompletionRequestStats(): RequestStats {
    return this.completionRequestStats;
  }

  getStatus(): TabbyApiClientStatus {
    return this.status;
  }

  hasHelpMessage(): boolean {
    return this.status === "noConnection" || this.completionResponseIssue !== undefined;
  }

  getHelpMessage(format?: "plaintext" | "markdown" | "html"): string | undefined {
    if (this.status === "noConnection") {
      const message = "Connect to Server Failed.\n" + this.connectionErrorMessage;
      if (format == "html") {
        return message?.replace(/\n/g, "<br/>");
      } else {
        return message;
      }
    } else if (this.completionResponseIssue) {
      return this.buildHelpMessage(format);
    }
    return;
  }

  isConnecting(): boolean {
    return this.connecting;
  }

  isFetchingCompletion(): boolean {
    return this.fetchingCompletion;
  }

  hasCompletionResponseTimeIssue(): boolean {
    return !!this.completionResponseIssue;
  }

  getServerHealth(): TabbyApiComponents["schemas"]["HealthState"] | undefined {
    return this.serverHealth;
  }

  isCodeCompletionApiAvailable(): boolean {
    const health = this.serverHealth;
    return !!(health && health["model"]);
  }

  isChatApiAvailable(): boolean {
    const health = this.serverHealth;
    return !!(health && health["chat_model"]);
  }

  async connect(): Promise<void> {
    await this.healthCheck();
    if (this.status === "ready") {
      await this.updateServerProvidedConfig();
      await this.anonymousUsageLogger.uniqueEvent("AgentConnected", this.serverHealth);
    }
  }

  private async healthCheck(signal?: AbortSignal, method: "GET" | "POST" = "GET"): Promise<void> {
    if (this.healthCheckMutexAbortController && !this.healthCheckMutexAbortController.signal.aborted) {
      this.healthCheckMutexAbortController.abort(new MutexAbortError());
    }
    this.healthCheckMutexAbortController = new AbortController();

    const requestId = uuid();
    const requestPath = "/v1/health";
    const requestDescription = `${method} ${this.endpoint + requestPath}`;
    const requestOptions = {
      signal: abortSignalFromAnyOf([
        signal,
        this.healthCheckMutexAbortController?.signal,
        this.createTimeOutAbortSignal(),
      ]),
    };
    try {
      if (!this.api) {
        throw new Error("http client not initialized");
      }
      this.logger.debug(`Health check request: ${requestDescription}. [${requestId}]`);
      this.updateIsConnecting(true);
      let response;
      if (method === "POST") {
        response = await this.api.POST(requestPath, requestOptions);
      } else {
        response = await this.api.GET(requestPath, requestOptions);
      }
      this.updateIsConnecting(false);
      this.logger.debug(`Health check response status: ${response.response.status}. [${requestId}]`);
      if (response.error || !response.response.ok) {
        throw new HttpError(response.response);
      }
      this.logger.trace(`Health check response data: [${requestId}]`, response.data);
      this.connectionErrorMessage = undefined;
      this.serverHealth = response.data;
      this.updateStatus("ready");
    } catch (error) {
      this.updateIsConnecting(false);
      this.serverHealth = undefined;
      if (error instanceof HttpError && error.status == 405 && method !== "POST") {
        return await this.healthCheck(signal, "POST");
      } else if (isUnauthorizedError(error)) {
        this.updateStatus("unauthorized");
      } else {
        if (isCanceledError(error)) {
          this.logger.debug(`Health check request canceled. [${requestId}]`);
          this.connectionErrorMessage = `${requestDescription} canceled.`;
        } else if (isTimeoutError(error)) {
          this.logger.error(`Health check request timed out. [${requestId}]`, error);
          this.connectionErrorMessage = `${requestDescription} timed out.`;
        } else {
          this.logger.error(`Health check request failed. [${requestId}]`, error);
          const message = error instanceof Error ? errorToString(error) : JSON.stringify(error);
          this.connectionErrorMessage = `${requestDescription} failed: \n${message}`;
        }
        this.updateStatus("noConnection");
      }
    }
  }

  private async updateServerProvidedConfig(): Promise<void> {
    const serverVersion = semver.coerce(this.serverHealth?.version.git_describe);
    if (serverVersion && semver.lt(serverVersion, "0.9.0")) {
      this.logger.debug(`Skip fetching server provided config due to server version: ${serverVersion}.`);
      return;
    }
    const requestId = uuid();
    const requestPath = "/v1beta/server_setting";
    const requestDescription = `GET ${this.endpoint + requestPath}`;
    const requestOptions = {
      signal: this.createTimeOutAbortSignal(),
    };
    try {
      if (!this.api) {
        throw new Error("http client not initialized");
      }
      this.logger.debug(`Fetch server provided config request: ${requestDescription}. [${requestId}]`);
      const response = await this.api.GET(requestPath, requestOptions);
      this.logger.debug(`Fetch server provided config response status: ${response.response.status}. [${requestId}]`);
      if (response.error || !response.response.ok) {
        throw new HttpError(response.response);
      }
      this.logger.trace(`Fetch server provided config response data: [${requestId}]`, response.data);
      const fetchedConfig = response.data;
      await this.configurations.updateServerProvidedConfig(fetchedConfig, true);
    } catch (error) {
      if (isUnauthorizedError(error)) {
        this.logger.debug(`Fetch server provided config request failed due to unauthorized. [${requestId}]`);
      } else if (!(error instanceof HttpError)) {
        this.logger.error(`Fetch server provided config request failed. [${requestId}]`, error);
      }
    }
  }

  async fetchCompletion(
    request: TabbyApiComponents["schemas"]["CompletionRequest"],
    signal?: AbortSignal,
    stats?: CompletionStats,
  ): Promise<TabbyApiComponents["schemas"]["CompletionResponse"]> {
    const requestId = uuid();
    const requestPath = "/v1/completions";
    const requestDescription = `POST ${this.endpoint + requestPath}`;
    const requestOptions = {
      body: request,
      signal: abortSignalFromAnyOf([signal, this.createTimeOutAbortSignal()]),
    };

    const requestStartedAt = performance.now();
    const statsData = {
      latency: NaN,
      canceled: false,
      timeout: false,
      notAvailable: false,
    };

    try {
      if (!this.api) {
        throw new Error("http client not initialized");
      }
      this.logger.debug(`Completion request: ${requestDescription}. [${requestId}]`);
      this.logger.trace(`Completion request body: [${requestId}]`, requestOptions.body);
      this.updateIsFetchingCompletion(true);
      const response = await this.api.POST(requestPath, requestOptions);
      this.updateIsFetchingCompletion(false);
      this.logger.debug(`Completion response status: ${response.response.status}. [${requestId}]`);
      if (response.error || !response.response.ok) {
        throw new HttpError(response.response);
      }
      this.logger.trace(`Completion response data: [${requestId}]`, response.data);
      statsData.latency = performance.now() - requestStartedAt;
      return response.data;
    } catch (error) {
      this.updateIsFetchingCompletion(false);
      if (isCanceledError(error)) {
        this.logger.debug(`Completion request canceled. [${requestId}]`);
        statsData.canceled = true;
      } else if (isTimeoutError(error)) {
        this.logger.debug(`Completion request timed out. [${requestId}]`);
        statsData.timeout = true;
      } else if (isUnauthorizedError(error)) {
        this.logger.debug(`Completion request failed due to unauthorized. [${requestId}]`);
        statsData.notAvailable = true;
        this.connect(); // schedule a reconnection
      } else {
        this.logger.error(`Completion request failed. [${requestId}]`, error);
        statsData.notAvailable = true;
        this.connect(); // schedule a reconnection
      }
      throw error; // rethrow error
    } finally {
      if (!statsData.notAvailable) {
        stats?.addRequestStatsEntry(statsData);
      }
      if (!statsData.notAvailable && !statsData.canceled) {
        this.completionRequestStats.add(statsData.latency);
        const statsResult = this.completionRequestStats.stats();
        const checkResult = RequestStats.check(statsResult);
        switch (checkResult) {
          case "healthy":
            this.updateCompletionResponseIssue(undefined);
            break;
          case "highTimeoutRate":
            this.updateCompletionResponseIssue("highTimeoutRate");
            break;
          case "slowResponseTime":
            this.updateCompletionResponseIssue("slowResponseTime");
            break;
        }
      }
    }
  }

  // FIXME(@icycodes): use openai for nodejs instead of tabby-openapi schema
  async fetchChatStream(
    request: TabbyApiComponents["schemas"]["ChatCompletionRequest"],
    signal?: AbortSignal,
    useBetaVersion: boolean = false,
  ): Promise<Readable | null> {
    const requestId = uuid();
    const requestPath = useBetaVersion ? "/v1beta/chat/completions" : "/v1/chat/completions";
    try {
      if (!this.api) {
        throw new Error("http client not initialized");
      }
      const requestOptions = {
        body: request,
        signal: abortSignalFromAnyOf([signal, this.createTimeOutAbortSignal()]),
        parseAs: "stream" as ParseAs,
      };
      const requestDescription = `POST ${this.endpoint + requestPath}`;
      this.logger.debug(`Chat request: ${requestDescription}. [${requestId}]`);
      this.logger.trace(`Chat request body: [${requestId}]`, requestOptions.body);
      const response = await this.api.POST(requestPath, requestOptions);
      this.logger.debug(`Chat response status: ${response.response.status}. [${requestId}]`);
      if (response.error || !response.response.ok) {
        throw new HttpError(response.response);
      }
      if (!response.response.body) {
        return null;
      }
      const readableStream = readChatStream(response.response.body, requestOptions.signal);
      return readableStream;
    } catch (error) {
      if (error instanceof HttpError && error.status == 404 && !useBetaVersion) {
        return await this.fetchChatStream(request, signal, true);
      }
      if (isCanceledError(error)) {
        this.logger.debug(`Chat request canceled. [${requestId}]`);
      } else if (isUnauthorizedError(error)) {
        this.logger.debug(`Chat request failed due to unauthorized. [${requestId}]`);
        this.connect(); // schedule a reconnection
      } else {
        this.logger.error(`Chat request failed. [${requestId}]`, error);
        this.connect(); // schedule a reconnection
      }
      throw error; // rethrow error
    }
  }

  async postEvent(
    request: TabbyApiComponents["schemas"]["LogEventRequest"] & { select_kind?: "line" },
    signal?: AbortSignal,
  ): Promise<boolean> {
    const requestId = uuid();
    const requestPath = "/v1/events";
    const requestDescription = `POST ${this.endpoint + requestPath}`;
    const requestOptions = {
      body: request,
      params: {
        query: {
          select_kind: request.select_kind,
        },
      },
      signal: abortSignalFromAnyOf([signal, this.createTimeOutAbortSignal()]),
      parseAs: "text" as ParseAs,
    };

    try {
      if (!this.api) {
        throw new Error("http client not initialized");
      }
      this.logger.debug(`Event request: ${requestDescription}. [${requestId}]`);
      this.logger.trace(`Event request body: [${requestId}]`, requestOptions.body);
      const response = await this.api.POST(requestPath, requestOptions);
      this.logger.debug(`Event response status: ${response.response.status}. [${requestId}]`);
      if (response.error || !response.response.ok) {
        throw new HttpError(response.response);
      }
      this.logger.trace(`Event response data: [${requestId}]`, response.data);
      return true;
    } catch (error) {
      if (isUnauthorizedError(error)) {
        this.logger.debug(`Completion request failed due to unauthorized. [${requestId}]`);
      } else {
        this.logger.error(`Event request failed. [${requestId}]`, error);
      }
      this.connect(); // schedule a reconnection
      return false;
    }
  }

  private buildHelpMessage(format?: "plaintext" | "markdown" | "html"): string | undefined {
    const outputFormat = format ?? "plaintext";
    let statsMessage = "";
    if (this.completionResponseIssue == "slowResponseTime") {
      const stats = this.completionRequestStats.stats().stats;
      if (stats && stats["responses"] && stats["averageResponseTime"]) {
        statsMessage = `The average response time of recent ${stats["responses"]} completion requests is ${Number(
          stats["averageResponseTime"],
        ).toFixed(0)}ms.<br/><br/>`;
      }
    }

    if (this.completionResponseIssue == "highTimeoutRate") {
      const stats = this.completionRequestStats.stats().stats;
      if (stats && stats["total"] && stats["timeouts"]) {
        statsMessage = `${stats["timeouts"]} of ${stats["total"]} completion requests timed out.<br/><br/>`;
      }
    }

    let helpMessageForRunningLargeModelOnCPU = "";
    const serverHealthState = this.serverHealth;
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
    const host = new URL(this.endpoint ?? "http://localhost:8080").host;
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
}
