import type { paths as TabbyApi, components as TabbyApiComponents } from "tabby-openapi/compatible";
import type { ParseAs } from "openapi-fetch";
import type { Readable } from "readable-stream";
import type { Configurations } from "../config";
import type { AnonymousUsageLogger } from "../telemetry";
import type { ConfigData } from "../config/type";
import type { ProxyConfig } from "./proxy";
import type { ClientInfo } from "../protocol";
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
  formatErrorMessage,
  HttpError,
  MutexAbortError,
  isUnauthorizedError,
  isCanceledError,
  isTimeoutError,
} from "../utils/error";

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

  private connectionErrorMessage: string | undefined = undefined;
  private serverHealth: TabbyApiComponents["schemas"]["HealthState"] | undefined = undefined;

  private healthCheckMutexAbortController: AbortController | undefined = undefined;

  private reconnectTimer: ReturnType<typeof setInterval> | undefined = undefined;
  private heartbeatTimer: ReturnType<typeof setInterval> | undefined = undefined;

  constructor(
    private readonly configurations: Configurations,
    private readonly anonymousUsageLogger: AnonymousUsageLogger,
  ) {
    super();
  }

  async initialize(clientInfo: ClientInfo | undefined) {
    this.userAgentString = this.buildUserAgentString(clientInfo);
    this.connect(); // no await

    this.configurations.on("updated", (config: ConfigData, oldConfig: ConfigData) => {
      const isServerConnectionChanged = !(
        deepEqual(config.server, oldConfig.server) && deepEqual(config.proxy, oldConfig.proxy)
      );
      if (isServerConnectionChanged) {
        this.logger.debug("Server configurations updated, reconnecting...");
        this.connect(); // no await
      }
    });

    const reconnectInterval = 1000 * 30; // 30s
    this.reconnectTimer = setInterval(async () => {
      if (this.status === "noConnection" || this.status === "unauthorized") {
        this.logger.debug("Trying to reconnect...");
        await this.connect({ skipReset: true });
      }
    }, reconnectInterval);

    const heartBeatInterval = 1000 * 60; // 1m
    this.heartbeatTimer = setInterval(async () => {
      if (this.status === "ready") {
        this.logger.trace("Heartbeat...");
        await this.healthCheck({ background: true });
      }
    }, heartBeatInterval);
  }

  async shutdown() {
    if (this.reconnectTimer) {
      clearInterval(this.reconnectTimer);
    }
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
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

  getStatus(): TabbyApiClientStatus {
    return this.status;
  }

  getHelpMessage(format?: "plaintext" | "markdown" | "html"): string | undefined {
    if (this.status === "noConnection") {
      const message = "Connect to Server Failed.\n" + this.connectionErrorMessage;
      if (format == "html") {
        return message?.replace(/\n/g, "<br/>");
      } else {
        return message;
      }
    }
    return undefined;
  }

  isConnecting(): boolean {
    return this.connecting;
  }

  getServerHealth(): TabbyApiComponents["schemas"]["HealthState"] | undefined {
    return this.serverHealth;
  }

  async connect(options: { skipReset?: boolean } = {}): Promise<void> {
    if (!options.skipReset) {
      this.connectionErrorMessage = undefined;
      this.serverHealth = undefined;
      this.updateStatus("noConnection");

      const config = this.configurations.getMergedConfig();
      const endpoint = config.server.endpoint;
      this.endpoint = endpoint;

      const auth = !isBlank(config.server.token) ? `Bearer ${config.server.token}` : undefined;
      const proxyConfigs: ProxyConfig[] = [{ fromEnv: true }];
      if (!isBlank(config.proxy.url)) {
        proxyConfigs.unshift(config.proxy);
      }
      this.api = createClient<TabbyApi>({
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
    await this.healthCheck();
    if (this.status === "ready") {
      await this.updateServerProvidedConfig();
      await this.anonymousUsageLogger.uniqueEvent("AgentConnected", this.serverHealth);
    }
  }

  private async healthCheck(options?: {
    signal?: AbortSignal;
    method?: "GET" | "POST";
    background?: boolean;
  }): Promise<void> {
    const signal = options?.signal;
    const method = options?.method ?? "GET";
    const background = options?.background;

    if (this.healthCheckMutexAbortController && !this.healthCheckMutexAbortController.signal.aborted) {
      if (background) {
        // there is a running check, skip background check
        return;
      }
      this.healthCheckMutexAbortController.abort(new MutexAbortError());
    }
    const abortController = new AbortController();
    this.healthCheckMutexAbortController = abortController;
    if (!background) {
      this.updateIsConnecting(true);
    }

    const requestId = uuid();
    const requestPath = "/v1/health";
    const requestDescription = `${method} ${this.endpoint + requestPath}`;
    const requestOptions = {
      signal: abortSignalFromAnyOf([signal, abortController.signal, this.createTimeOutAbortSignal()]),
    };
    try {
      if (!this.api) {
        throw new Error("http client not initialized");
      }
      this.logger.debug(`Health check request: ${requestDescription}. [${requestId}]`);
      let response;
      if (method === "POST") {
        response = await this.api.POST(requestPath, requestOptions);
      } else {
        response = await this.api.GET(requestPath, requestOptions);
      }
      this.logger.debug(`Health check response status: ${response.response.status}. [${requestId}]`);
      if (response.error || !response.response.ok) {
        throw new HttpError(response.response);
      }
      this.logger.trace(`Health check response data: [${requestId}]`, response.data);
      this.connectionErrorMessage = undefined;
      this.serverHealth = response.data;
      this.updateStatus("ready");
    } catch (error) {
      if (isCanceledError(error)) {
        this.logger.debug(`Health check request canceled. [${requestId}]`);
      } else if (error instanceof HttpError && error.status == 405 && method !== "POST") {
        return await this.healthCheck({ signal, method: "POST" });
      } else if (isUnauthorizedError(error)) {
        this.serverHealth = undefined;
        this.updateStatus("unauthorized");
      } else if (isTimeoutError(error)) {
        this.logger.error(`Health check request timed out. [${requestId}]`, error);
        this.serverHealth = undefined;
        this.connectionErrorMessage = `${requestDescription} timed out.`;
        this.updateStatus("noConnection");
      } else {
        this.logger.error(`Health check request failed. [${requestId}]`, error);
        this.serverHealth = undefined;
        this.connectionErrorMessage = `${requestDescription} failed: \n${formatErrorMessage(error)}`;
        this.updateStatus("noConnection");
      }
    } finally {
      if (this.healthCheckMutexAbortController === abortController) {
        this.healthCheckMutexAbortController = undefined;
        if (!background) {
          this.updateIsConnecting(false);
        }
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
    // set to track latency, the properties in latencyStats object will be updated in this function
    latencyStats?: {
      latency?: number; // ms, undefined means no data, timeout or canceled
      canceled?: boolean;
      timeout?: boolean;
    },
  ): Promise<TabbyApiComponents["schemas"]["CompletionResponse"]> {
    const requestId = uuid();
    const requestPath = "/v1/completions";
    const requestDescription = `POST ${this.endpoint + requestPath}`;
    const requestOptions = {
      body: request,
      signal: abortSignalFromAnyOf([signal, this.createTimeOutAbortSignal()]),
    };

    const requestStartedAt = performance.now();
    try {
      if (!this.api) {
        throw new Error("http client not initialized");
      }
      this.logger.debug(`Completion request: ${requestDescription}. [${requestId}]`);
      this.logger.trace(`Completion request body: [${requestId}]`, requestOptions.body);
      const response = await this.api.POST(requestPath, requestOptions);
      this.logger.debug(`Completion response status: ${response.response.status}. [${requestId}]`);
      if (response.error || !response.response.ok) {
        throw new HttpError(response.response);
      }
      this.logger.trace(`Completion response data: [${requestId}]`, response.data);
      if (latencyStats) {
        latencyStats.latency = performance.now() - requestStartedAt;
      }
      return response.data;
    } catch (error) {
      if (isCanceledError(error)) {
        this.logger.debug(`Completion request canceled. [${requestId}]`);
        if (latencyStats) {
          latencyStats.canceled = true;
        }
      } else if (isTimeoutError(error)) {
        this.logger.debug(`Completion request timed out. [${requestId}]`);
        if (latencyStats) {
          latencyStats.timeout = true;
        }
      } else if (isUnauthorizedError(error)) {
        this.logger.debug(`Completion request failed due to unauthorized. [${requestId}]`);
        this.healthCheck();
      } else {
        this.logger.error(`Completion request failed. [${requestId}]`, error);
        this.healthCheck();
      }

      if (latencyStats) {
        latencyStats.latency = undefined;
      }
      throw error; // rethrow error
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
        this.healthCheck();
      } else {
        this.logger.error(`Chat request failed. [${requestId}]`, error);
        this.healthCheck();
      }
      throw error; // rethrow error
    }
  }

  async postEvent(
    request: TabbyApiComponents["schemas"]["LogEventRequest"] & { select_kind?: "line" },
    signal?: AbortSignal,
  ): Promise<void> {
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
    } catch (error) {
      if (isCanceledError(error)) {
        this.logger.debug(`Event request canceled. [${requestId}]`);
      }
      if (isUnauthorizedError(error)) {
        this.logger.debug(`Event request failed due to unauthorized. [${requestId}]`);
        this.healthCheck();
      } else {
        this.logger.error(`Event request failed. [${requestId}]`, error);
        this.healthCheck();
      }
      throw error; // rethrow error
    }
  }
}
