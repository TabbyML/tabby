import { EventEmitter } from "events";
import { v4 as uuid } from "uuid";
import deepEqual from "deep-equal";
import { deepmerge } from "deepmerge-ts";
import { getProperty, setProperty, deleteProperty } from "dot-prop";
import createClient from "openapi-fetch";
import type { ParseAs } from "openapi-fetch";
import * as semver from "semver";
import { Readable } from "readable-stream";
import type { paths as TabbyApi, components as TabbyApiComponents } from "tabby-openapi/compatible";
import type {
  Agent,
  AgentStatus,
  AgentIssue,
  AgentEvent,
  ClientProperties,
  AgentInitOptions,
  AbortSignalOption,
  ServerHealthState,
  CompletionRequest,
  CompletionResponse,
  LogEventRequest,
} from "./Agent";
import { dataStore as defaultDataStore, DataStore } from "./dataStore";
import {
  isBlank,
  abortSignalFromAnyOf,
  HttpError,
  isTimeoutError,
  isCanceledError,
  isUnauthorizedError,
  errorToString,
  stringToRegExp,
} from "./utils";
import { readChatStream, parseChatResponse } from "./stream";
import { Auth } from "./Auth";
import { AgentConfig, PartialAgentConfig, defaultAgentConfig } from "./AgentConfig";
import { configFile } from "./configFile";
import { CompletionCache } from "./CompletionCache";
import { CompletionDebounce } from "./CompletionDebounce";
import { CompletionStats, RequestStats } from "./CompletionStats";
import { CompletionContext } from "./CompletionContext";
import { CompletionSolution, CompletionItem, emptyInlineCompletionList } from "./CompletionSolution";
import { preCacheProcess, postCacheProcess } from "./postprocess";
import { getLogger, logDestinations, fileLogger } from "./logger";
import { AnonymousUsageLogger } from "./AnonymousUsageLogger";
import { loadTlsCaCerts } from "./loadCaCerts";

export class TabbyAgent extends EventEmitter implements Agent {
  private readonly logger = getLogger("TabbyAgent");
  private anonymousUsageLogger = new AnonymousUsageLogger();
  private config: AgentConfig = defaultAgentConfig;
  private userConfig: PartialAgentConfig = {}; // config from `~/.tabby-client/agent/config.toml`
  private clientConfig: PartialAgentConfig = {}; // config from `initialize` and `updateConfig` method
  private serverProvidedConfig: PartialAgentConfig = {}; // config fetched from server and saved in dataStore
  private status: AgentStatus = "notInitialized";
  private issues: AgentIssue["name"][] = [];
  private serverHealthState?: ServerHealthState;
  private connectionErrorMessage?: string;
  private dataStore?: DataStore;
  private api?: ReturnType<typeof createClient<TabbyApi>>;
  private auth?: Auth;
  private completionCache = new CompletionCache();
  private completionDebounce = new CompletionDebounce();
  private completionMutexAbortController?: AbortController;
  private completionStats = new CompletionStats();
  private completionRequestStats = new RequestStats();
  static readonly tryConnectInterval = 1000 * 30; // 30s
  private tryingConnectTimer: ReturnType<typeof setInterval>;
  static readonly submitStatsInterval = 1000 * 60 * 60 * 24; // 24h
  private submitStatsTimer: ReturnType<typeof setInterval>;

  constructor() {
    super();

    this.tryingConnectTimer = setInterval(async () => {
      if (this.status === "disconnected") {
        this.logger.debug("Trying to connect...");
        await this.healthCheck();
      }
    }, TabbyAgent.tryConnectInterval);

    this.submitStatsTimer = setInterval(async () => {
      await this.submitStats();
    }, TabbyAgent.submitStatsInterval);
  }

  private async applyConfig() {
    this.logger.info("Applying updated config...");
    const oldConfig = this.config;
    const oldStatus = this.status;

    this.config = deepmerge(
      defaultAgentConfig,
      this.userConfig,
      this.clientConfig,
      this.serverProvidedConfig,
    ) as AgentConfig;
    this.config.server.endpoint = this.config.server.endpoint.replace(/\/+$/, ""); // remove trailing slash
    this.logger.trace("Updated config:", this.config);

    if (fileLogger) {
      fileLogger.level = this.config.logs.level;
    }
    this.anonymousUsageLogger.disabled = this.config.anonymousUsageTracking.disable;

    await loadTlsCaCerts(this.config.tls);

    if (isBlank(this.config.server.token) && this.config.server.requestHeaders["Authorization"] === undefined) {
      if (this.config.server.endpoint !== this.auth?.endpoint) {
        this.auth = new Auth(this.config.server.endpoint);
        await this.auth.init({ dataStore: this.dataStore });
        this.auth.on("updated", () => {
          this.setupApi();
        });
      }
    } else {
      // If auth token is provided, use it directly.
      this.auth = undefined;
    }

    // If server config changed, clear server related state
    if (!deepEqual(oldConfig.server, this.config.server)) {
      this.serverHealthState = undefined;
      this.completionRequestStats.reset();
      this.popIssue("slowCompletionResponseTime");
      this.popIssue("highCompletionTimeoutRate");
      this.popIssue("connectionFailed");
      this.connectionErrorMessage = undefined;
    }

    if (!this.api || !deepEqual(oldConfig.server, this.config.server)) {
      await this.setupApi();
    }

    this.logger.info("Completed applying updated config.");
    const event: AgentEvent = { event: "configUpdated", config: this.config };
    super.emit("configUpdated", event);
    if (
      !deepEqual(oldConfig.server, this.config.server) &&
      oldStatus === "unauthorized" &&
      this.status === "unauthorized"
    ) {
      // If server config changed and status remain `unauthorized`, we want to emit `authRequired` again.
      // but `changeStatus` will not emit `authRequired` if status is not changed, so we emit it manually here.
      this.emitAuthRequired();
    }
  }

  private async setupApi() {
    const auth = !isBlank(this.config.server.token)
      ? `Bearer ${this.config.server.token}`
      : this.auth?.token
        ? `Bearer ${this.auth.token}`
        : undefined;
    this.api = createClient<TabbyApi>({
      baseUrl: this.config.server.endpoint,
      headers: {
        Authorization: auth,
        ...this.config.server.requestHeaders,
      },
    });
    await this.healthCheck();
  }

  private changeStatus(status: AgentStatus) {
    if (this.status != status) {
      this.status = status;
      this.logger.info(`Status changed: ${status}.`);
      const event: AgentEvent = { event: "statusChanged", status };
      super.emit("statusChanged", event);
      if (this.status === "unauthorized") {
        this.emitAuthRequired();
      }
    }
  }

  private issueFromName(issueName: AgentIssue["name"]): AgentIssue {
    switch (issueName) {
      case "highCompletionTimeoutRate":
        return {
          name: "highCompletionTimeoutRate",
          completionResponseStats: this.completionRequestStats.stats().stats,
        };
      case "slowCompletionResponseTime":
        return {
          name: "slowCompletionResponseTime",
          completionResponseStats: this.completionRequestStats.stats().stats,
        };
      case "connectionFailed":
        return {
          name: "connectionFailed",
          message: this.connectionErrorMessage,
        };
    }
  }

  private pushIssue(issue: AgentIssue["name"]) {
    if (!this.issues.includes(issue)) {
      this.issues.push(issue);
      this.logger.info(`Issue created: ${issue}.`);
      this.emitIssueUpdated();
    }
  }

  private popIssue(issue: AgentIssue["name"]) {
    const index = this.issues.indexOf(issue);
    if (index >= 0) {
      this.issues.splice(index, 1);
      this.logger.info(`Issue removed: ${issue}.`);
      this.emitIssueUpdated();
    }
  }

  private emitAuthRequired() {
    const event: AgentEvent = { event: "authRequired", server: this.config.server };
    super.emit("authRequired", event);
  }

  private emitIssueUpdated() {
    const event: AgentEvent = { event: "issuesUpdated", issues: this.issues };
    super.emit("issuesUpdated", event);
  }

  private async submitStats() {
    const stats = this.completionStats.stats();
    if (stats["completion_request"]["count"] > 0) {
      await this.anonymousUsageLogger.event("AgentStats", { stats });
      this.completionStats.reset();
    }
  }

  private createAbortSignal(options?: { signal?: AbortSignal; timeout?: number }): AbortSignal {
    const timeout = Math.min(0x7fffffff, options?.timeout || this.config.server.requestTimeout);
    return abortSignalFromAnyOf([AbortSignal.timeout(timeout), options?.signal]);
  }

  private async healthCheck(options?: { signal?: AbortSignal; method?: "GET" | "POST" }): Promise<void> {
    const requestId = uuid();
    const requestPath = "/v1/health";
    const requestDescription = `${options?.method || "GET"} ${this.config.server.endpoint + requestPath}`;
    const requestOptions = {
      signal: this.createAbortSignal({ signal: options?.signal }),
    };
    try {
      if (!this.api) {
        throw new Error("http client not initialized");
      }
      this.logger.debug(`Health check request: ${requestDescription}. [${requestId}]`);
      let response;
      if (options?.method === "POST") {
        response = await this.api.POST(requestPath, requestOptions);
      } else {
        response = await this.api.GET(requestPath, requestOptions);
      }
      this.logger.debug(`Health check response status: ${response.response.status}. [${requestId}]`);
      if (response.error || !response.response.ok) {
        throw new HttpError(response.response);
      }
      this.logger.trace(`Health check response data: [${requestId}]`, response.data);
      const healthState = response.data;
      this.popIssue("connectionFailed");
      this.connectionErrorMessage = undefined;
      if (
        typeof healthState === "object" &&
        healthState["model"] !== undefined &&
        healthState["device"] !== undefined
      ) {
        this.serverHealthState = healthState;
        this.anonymousUsageLogger.uniqueEvent("AgentConnected", healthState);

        // schedule fetch server config later, no await
        this.fetchServerProvidedConfig();
      }
      this.changeStatus("ready");
    } catch (error) {
      this.serverHealthState = undefined;
      if (error instanceof HttpError && error.status == 405 && options?.method !== "POST") {
        return await this.healthCheck({ method: "POST" });
      } else if (isUnauthorizedError(error)) {
        this.changeStatus("unauthorized");
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
        this.pushIssue("connectionFailed");
        this.changeStatus("disconnected");
      }
    }
  }

  private async fetchServerProvidedConfig(): Promise<void> {
    const serverVersion = semver.coerce(this.serverHealthState?.version.git_describe);
    if (serverVersion && semver.lt(serverVersion, "0.9.0")) {
      this.logger.debug(`Skip fetching server provided config due to server version: ${serverVersion}.`);
      return;
    }
    const requestId = uuid();
    try {
      if (!this.api) {
        throw new Error("http client not initialized");
      }
      const requestPath = "/v1beta/server_setting";
      const requestDescription = `GET ${this.config.server.endpoint + requestPath}`;
      this.logger.debug(`Fetch server provided config request: ${requestDescription}. [${requestId}]`);
      const response = await this.api.GET(requestPath);
      this.logger.debug(`Fetch server provided config response status: ${response.response.status}. [${requestId}]`);
      if (response.error || !response.response.ok) {
        throw new HttpError(response.response);
      }
      this.logger.trace(`Fetch server provided config response data: [${requestId}]`, response.data);
      const fetchedConfig = response.data;
      const serverProvidedConfig: PartialAgentConfig = {};
      if (fetchedConfig.disable_client_side_telemetry) {
        serverProvidedConfig.anonymousUsageTracking = {
          disable: true,
        };
      }

      if (!deepEqual(serverProvidedConfig, this.serverProvidedConfig)) {
        this.serverProvidedConfig = serverProvidedConfig;
        await this.applyConfig();
        if (this.dataStore) {
          if (!this.dataStore.data.serverConfig) {
            this.dataStore.data.serverConfig = {};
          }
          this.dataStore.data.serverConfig[this.config.server.endpoint] = this.serverProvidedConfig;
          try {
            await this.dataStore.save();
          } catch (error) {
            this.logger.error("Failed to save server provided config.", error);
          }
        }
      }
    } catch (error) {
      if (isUnauthorizedError(error)) {
        this.logger.debug(`Fetch server provided config request failed due to unauthorized. [${requestId}]`);
      } else if (!(error instanceof HttpError)) {
        this.logger.error(`Fetch server provided config request failed. [${requestId}]`, error);
      }
    }
  }

  public async initialize(options: AgentInitOptions): Promise<boolean> {
    // initialize loggers
    if (options.loggers) {
      logDestinations.attach(...options.loggers);
    }
    if (configFile) {
      await configFile.load();
      this.userConfig = configFile.config;
      configFile.on("updated", async (config) => {
        this.userConfig = config;
        await this.applyConfig();
      });
      configFile.watch();
    }
    if (options.config) {
      this.clientConfig = options.config;
    }
    if (fileLogger) {
      fileLogger.level = this.clientConfig.logs?.level ?? this.userConfig.logs?.level ?? this.config.logs.level;
    }

    this.logger.info("Initializing...");
    this.logger.trace("Initialization options:", options);

    this.dataStore = options.dataStore ?? defaultDataStore;
    if (this.dataStore) {
      try {
        await this.dataStore.load();
        if ("watch" in this.dataStore && typeof this.dataStore.watch === "function") {
          this.dataStore.watch();
        }
      } catch (error) {
        this.logger.error("Failed to load stored data.", error);
      }
    }
    await this.anonymousUsageLogger.init({ dataStore: this.dataStore });
    if (options.clientProperties) {
      if (options.clientProperties.session) {
        Object.entries(options.clientProperties.session).forEach(([key, value]) => {
          this.anonymousUsageLogger.setSessionProperties(key, value);
        });
      }
      if (options.clientProperties.user) {
        Object.entries(options.clientProperties.user).forEach(([key, value]) => {
          this.anonymousUsageLogger.setUserProperties(key, value);
        });
      }
    }
    if (this.dataStore) {
      const localConfig = deepmerge(defaultAgentConfig, this.userConfig, this.clientConfig) as AgentConfig;
      this.serverProvidedConfig = this.dataStore?.data.serverConfig?.[localConfig.server.endpoint] ?? {};
      if (this.dataStore instanceof EventEmitter) {
        this.dataStore.on("updated", async () => {
          const localConfig = deepmerge(defaultAgentConfig, this.userConfig, this.clientConfig) as AgentConfig;
          const storedServerConfig = defaultDataStore?.data.serverConfig?.[localConfig.server.endpoint];
          if (!deepEqual(storedServerConfig, this.serverProvidedConfig)) {
            this.serverProvidedConfig = storedServerConfig ?? {};
            await this.applyConfig();
          }
        });
      }
    }
    await this.applyConfig();
    await this.anonymousUsageLogger.uniqueEvent("AgentInitialized");
    this.logger.info("Initialized.");
    return this.status !== "notInitialized";
  }

  public async finalize(): Promise<boolean> {
    if (this.status === "finalized") {
      return false;
    }
    this.logger.info(`Finalizing...`);

    await this.submitStats();

    if (this.tryingConnectTimer) {
      clearInterval(this.tryingConnectTimer);
    }
    if (this.submitStatsTimer) {
      clearInterval(this.submitStatsTimer);
    }
    this.changeStatus("finalized");
    this.logger.info(`Finalized.`);
    return true;
  }

  public async updateClientProperties(type: keyof ClientProperties, key: string, value: any): Promise<boolean> {
    this.logger.trace(`Client properties updated.`, { type, key, value });
    switch (type) {
      case "session":
        this.anonymousUsageLogger.setSessionProperties(key, value);
        break;
      case "user":
        this.anonymousUsageLogger.setUserProperties(key, value);
        break;
    }
    return true;
  }

  public async updateConfig(key: string, value: any): Promise<boolean> {
    this.logger.trace(`Config updated.`, { key, value });
    const current = getProperty(this.clientConfig, key);
    if (!deepEqual(current, value)) {
      if (value === undefined) {
        deleteProperty(this.clientConfig, key);
      } else {
        setProperty(this.clientConfig, key, value);
      }
      await this.applyConfig();
    }
    return true;
  }

  public async clearConfig(key: string): Promise<boolean> {
    return await this.updateConfig(key, undefined);
  }

  public getConfig(): AgentConfig {
    return this.config;
  }

  public getStatus(): AgentStatus {
    return this.status;
  }

  public getIssues(): AgentIssue["name"][] {
    return this.issues;
  }

  public getIssueDetail<T extends AgentIssue>(options: { index?: number; name?: T["name"] }): T | null {
    const issues = this.getIssues();
    if (options.index !== undefined && options.index < issues.length) {
      return this.issueFromName(issues[options.index]!) as T;
    } else if (options.name !== undefined && this.issues.includes(options.name)) {
      return this.issueFromName(options.name) as T;
    } else {
      return null;
    }
  }

  public getServerHealthState(): ServerHealthState | null {
    return this.serverHealthState ?? null;
  }

  // @deprecated Tabby Cloud auth
  public async requestAuthUrl(options?: AbortSignalOption): Promise<{ authUrl: string; code: string } | null> {
    if (this.status === "notInitialized") {
      throw new Error("Agent is not initialized");
    }
    await this.healthCheck(options);
    if (this.status !== "unauthorized" || !this.auth) {
      return null;
    } else {
      return await this.auth.requestAuthUrl(options);
    }
  }

  // @deprecated Tabby Cloud auth
  public async waitForAuthToken(code: string, options?: AbortSignalOption): Promise<void> {
    if (this.status === "notInitialized") {
      throw new Error("Agent is not initialized");
    }
    if (this.status !== "unauthorized" || !this.auth) {
      return;
    }
    await this.auth.pollingToken(code, options);
    await this.setupApi();
  }

  public async provideCompletions(
    request: CompletionRequest,
    options?: AbortSignalOption,
  ): Promise<CompletionResponse> {
    if (this.status === "notInitialized") {
      throw new Error("Agent is not initialized");
    }
    this.logger.debug("Function providedCompletions called.");

    // Mutex Control
    if (this.completionMutexAbortController) {
      const reason = new Error("Aborted due to new request.");
      reason.name = "AbortError";
      this.completionMutexAbortController.abort(reason);
    }
    this.completionMutexAbortController = new AbortController();
    const signal = abortSignalFromAnyOf([this.completionMutexAbortController.signal, options?.signal]);

    // Processing request
    const context = new CompletionContext(request);
    if (!context.isValid()) {
      // Early return if request is not valid
      return emptyInlineCompletionList;
    }

    let solution: CompletionSolution | undefined = undefined;
    let cachedSolution: CompletionSolution | undefined = undefined;
    if (this.completionCache.has(context.hash)) {
      cachedSolution = this.completionCache.get(context.hash);
    }

    try {
      // Resolve solution
      if (cachedSolution && (!request.manually || cachedSolution.isCompleted)) {
        // Found cached solution
        // TriggerKind is Automatic, or the solution is completed
        // Return cached solution, do not need to fetch more choices

        // Debounce before continue processing cached solution
        await this.completionDebounce.debounce(
          {
            request,
            config: this.config.completion.debounce,
            responseTime: 0,
          },
          { signal },
        );

        solution = cachedSolution.withContext(context);
        this.logger.info("Completion cache hit.");
      } else if (!request.manually) {
        // No cached solution
        // TriggerKind is Automatic
        // We need to fetch the first choice

        // Debounce before fetching
        await this.completionDebounce.debounce(
          {
            request,
            config: this.config.completion.debounce,
            responseTime: this.completionRequestStats.stats().stats.averageResponseTime,
          },
          { signal },
        );

        solution = new CompletionSolution(context);
        // Fetch the completion
        this.logger.info(`Fetching completion...`);
        try {
          const response = await this.fetchCompletion(
            context.language,
            context.buildSegments(this.config.completion.prompt),
            undefined,
            signal,
          );
          const completionItem = CompletionItem.createFromResponse(context, response);
          // postprocess: preCache
          solution.add(...(await preCacheProcess([completionItem], this.config.postprocess)));
        } catch (error) {
          if (isCanceledError(error)) {
            this.logger.info(`Fetching completion canceled.`);
            solution = undefined;
          }
        }
      } else {
        // No cached solution, or cached solution is not completed
        // TriggerKind is Manual
        // We need to fetch the more choices

        solution = cachedSolution?.withContext(context) ?? new CompletionSolution(context);
        this.logger.info(`Fetching more completions...`);

        try {
          let tries = 0;
          while (
            solution.items.length < this.config.completion.solution.maxItems &&
            tries < this.config.completion.solution.maxTries
          ) {
            tries++;
            const response = await this.fetchCompletion(
              context.language,
              context.buildSegments(this.config.completion.prompt),
              this.config.completion.solution.temperature,
              signal,
            );
            const completionItem = CompletionItem.createFromResponse(context, response);
            // postprocess: preCache
            solution.add(...(await preCacheProcess([completionItem], this.config.postprocess)));
            if (signal.aborted) {
              throw signal.reason;
            }
          }
          // Mark the solution as completed
          solution.isCompleted = true;
        } catch (error) {
          if (isCanceledError(error)) {
            this.logger.info(`Fetching completion canceled.`);
            solution = undefined;
          }
        }
      }
      // Postprocess solution
      if (solution) {
        // Update Cache
        this.completionCache.update(solution);

        // postprocess: postCache
        solution = solution.withItems(...(await postCacheProcess(solution.items, this.config.postprocess)));
        if (signal.aborted) {
          throw signal.reason;
        }
      }
    } catch (error) {
      if (!isCanceledError(error)) {
        this.logger.error(`Providing completions failed.`, error);
      }
    }
    if (solution) {
      const result = solution.toInlineCompletionList();
      this.logger.info(`Completed processing completions, choices returned: ${result.items.length}.`);
      this.logger.trace("Completion solution:", { result });
      return result;
    }
    return emptyInlineCompletionList;
  }

  private async fetchCompletion(
    language: string,
    segments: TabbyApiComponents["schemas"]["Segments"],
    temperature: number | undefined,
    signal?: AbortSignal,
  ): Promise<TabbyApiComponents["schemas"]["CompletionResponse"]> {
    const requestId = uuid();
    const stats = {
      latency: NaN,
      canceled: false,
      timeout: false,
      notAvailable: false,
    };
    const requestStartedAt = performance.now();
    try {
      if (!this.api) {
        throw new Error("http client not initialized");
      }
      const requestPath = "/v1/completions";
      const requestOptions = {
        body: {
          language,
          segments,
          temperature,
        },
        signal: this.createAbortSignal({ signal }),
      };
      const requestDescription = `POST ${this.config.server.endpoint + requestPath}`;
      this.logger.debug(`Completion request: ${requestDescription}. [${requestId}]`);
      this.logger.trace(`Completion request body: [${requestId}]`, requestOptions.body);
      const response = await this.api.POST(requestPath, requestOptions);
      this.logger.debug(`Completion response status: ${response.response.status}. [${requestId}]`);
      if (response.error || !response.response.ok) {
        throw new HttpError(response.response);
      }
      this.logger.trace(`Completion response data: [${requestId}]`, response.data);
      stats.latency = performance.now() - requestStartedAt;
      return response.data;
    } catch (error) {
      if (isCanceledError(error)) {
        this.logger.debug(`Completion request canceled. [${requestId}]`);
        stats.canceled = true;
      } else if (isTimeoutError(error)) {
        this.logger.debug(`Completion request timed out. [${requestId}]`);
        stats.timeout = true;
      } else if (isUnauthorizedError(error)) {
        this.logger.debug(`Completion request failed due to unauthorized. [${requestId}]`);
        stats.notAvailable = true;
        this.healthCheck(); // schedule a health check
      } else {
        this.logger.error(`Completion request failed. [${requestId}]`, error);
        stats.notAvailable = true;
        this.healthCheck(); // schedule a health check
      }
      // rethrow error
      throw error;
    } finally {
      if (!stats.notAvailable) {
        this.completionStats.addRequestStatsEntry(stats);
      }
      if (!stats.notAvailable && !stats.canceled) {
        this.completionRequestStats.add(stats.latency);
        const statsResult = this.completionRequestStats.stats();
        const checkResult = RequestStats.check(statsResult);
        switch (checkResult) {
          case "healthy":
            this.popIssue("slowCompletionResponseTime");
            this.popIssue("highCompletionTimeoutRate");
            break;
          case "highTimeoutRate":
            this.popIssue("slowCompletionResponseTime");
            this.pushIssue("highCompletionTimeoutRate");
            break;
          case "slowResponseTime":
            this.popIssue("highCompletionTimeoutRate");
            this.pushIssue("slowCompletionResponseTime");
            break;
        }
      }
    }
  }

  public async postEvent(request: LogEventRequest, options?: AbortSignalOption): Promise<boolean> {
    if (this.status === "notInitialized") {
      throw new Error("Agent is not initialized");
    }
    this.completionStats.addEvent(request.type);
    const requestId = uuid();
    try {
      if (!this.api) {
        throw new Error("http client not initialized");
      }
      const requestPath = "/v1/events";
      const requestOptions = {
        body: request,
        params: {
          query: {
            select_kind: request.select_kind,
          },
        },
        signal: this.createAbortSignal(options),
        parseAs: "text" as ParseAs,
      };
      const requestDescription = `POST ${this.config.server.endpoint + requestPath}`;
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
      this.healthCheck(); // schedule a health check
      return false;
    }
  }

  public async generateCommitMessage(
    diff: string | string[],
    options?: AbortSignalOption & { useBetaVersion?: boolean },
  ): Promise<string> {
    if (this.status === "notInitialized") {
      throw new Error("Agent is not initialized");
    }

    // select diffs from the list to generate a prompt under the prompt size limit
    const { maxDiffLength, promptTemplate, responseMatcher } = this.config.chat.generateCommitMessage;
    let splitDiffs: string[];
    if (typeof diff === "string") {
      splitDiffs = diff.split(/\n(?=diff)/);
    } else {
      splitDiffs = diff;
    }
    let selectedDiff = "";
    for (const item of splitDiffs) {
      if (selectedDiff.length + item.length < maxDiffLength) {
        selectedDiff += item + "\n";
      }
    }
    if (isBlank(selectedDiff)) {
      // This may happen when all separated diffs are larger than the limit.
      if (typeof diff === "string") {
        selectedDiff = diff.substring(0, maxDiffLength);
      } else {
        selectedDiff = diff.join("\n").substring(0, maxDiffLength);
      }
    }
    if (isBlank(selectedDiff)) {
      // early return if selectedDiff is still empty
      return "";
    }

    // request chat api
    const requestId = uuid();
    try {
      if (!this.api) {
        throw new Error("http client not initialized");
      }
      const requestPath = options?.useBetaVersion ? "/v1beta/chat/completions" : "/v1/chat/completions";
      const messages = [
        {
          role: "user",
          content: promptTemplate.replace("{{diff}}", selectedDiff),
        },
      ];
      const requestOptions = {
        body: { messages },
        signal: this.createAbortSignal(options),
        parseAs: "stream" as ParseAs,
      };
      const requestDescription = `POST ${this.config.server.endpoint + requestPath}`;
      this.logger.debug(`Chat request: ${requestDescription}. [${requestId}]`);
      this.logger.trace(`Chat request body: [${requestId}]`, requestOptions.body);
      const response = await this.api.POST(requestPath, requestOptions);
      this.logger.debug(`Chat response status: ${response.response.status}. [${requestId}]`);
      if (response.error || !response.response.ok) {
        throw new HttpError(response.response);
      }
      const responseMessage = await parseChatResponse(response.response, requestOptions.signal);
      this.logger.trace(`Chat response message: [${requestId}]`, { responseMessage });
      const matcherReg = stringToRegExp(responseMatcher);
      const match = matcherReg.exec(responseMessage);
      const commitMessage = (match ? match[0] : responseMessage).trim();
      this.logger.trace(`Extracted commit message:`, { commitMessage });
      return commitMessage;
    } catch (error) {
      if (error instanceof HttpError && error.status == 404 && !options?.useBetaVersion) {
        return await this.generateCommitMessage(diff, { ...options, useBetaVersion: true });
      }
      if (isCanceledError(error)) {
        this.logger.debug(`Chat request canceled. [${requestId}]`);
      } else if (isUnauthorizedError(error)) {
        this.logger.debug(`Chat request failed due to unauthorized. [${requestId}]`);
      } else {
        this.logger.error(`Chat request failed. [${requestId}]`, error);
      }
      this.healthCheck(); // schedule a health check
    }
    return "";
  }

  // selection.start equals selection.end means the cursor position with no selection
  public async provideChatEdit(
    document: string,
    selection: { start: number; end: number },
    filepath: string,
    insertMode = false,
    command: string,
    languageId = "",
    options?: AbortSignalOption & { useBetaVersion?: boolean },
  ): Promise<Readable | null> {
    if (this.status === "notInitialized") {
      throw new Error("Agent is not initialized");
    }

    const documentMaxChars = this.config.chat.edit.documentMaxChars;
    if (selection.end - selection.start > documentMaxChars) {
      throw new Error("Document to edit is too long");
    }
    if (command.length > this.config.chat.edit.commandMaxChars) {
      throw new Error("Command is too long");
    }

    let promptTemplate: string;
    let userCommand: string;
    const presetCommand = /^\/\w+\b/g.exec(command)?.[0];
    const presetConfig = presetCommand && this.config.chat.edit.presetCommands[presetCommand];
    if (presetConfig) {
      promptTemplate = presetConfig.promptTemplate;
      userCommand = command.substring(presetCommand.length);
    } else {
      promptTemplate = insertMode
        ? this.config.chat.edit.promptTemplate.insert
        : this.config.chat.edit.promptTemplate.replace;
      userCommand = command;
    }
    // Extract the selected text and the surrounding context
    const documentSelection = document.substring(selection.start, selection.end);
    let documentPrefix = document.substring(0, selection.start);
    let documentSuffix = document.substring(selection.end);
    if (document.length > documentMaxChars) {
      const charsRemain = documentMaxChars - documentSelection.length;
      if (documentPrefix.length < charsRemain / 2) {
        documentSuffix = documentSuffix.substring(0, charsRemain - documentPrefix.length);
      } else if (documentSuffix.length < charsRemain / 2) {
        documentPrefix = documentPrefix.substring(documentPrefix.length - charsRemain + documentSuffix.length);
      } else {
        documentPrefix = documentPrefix.substring(documentPrefix.length - charsRemain / 2);
        documentSuffix = documentSuffix.substring(0, charsRemain / 2);
      }
    }
    // request chat api
    const requestId = uuid();
    try {
      if (!this.api) {
        throw new Error("http client not initialized");
      }
      const requestPath = options?.useBetaVersion ? "/v1beta/chat/completions" : "/v1/chat/completions";
      const messages = [
        {
          role: "user",
          content: promptTemplate.replace(
            /{{filepath}}|{{documentPrefix}}|{{document}}|{{documentSuffix}}|{{command}}|{{languageId}}/g,
            (pattern: string) => {
              switch (pattern) {
                case "{{filepath}}":
                  return filepath;
                case "{{documentPrefix}}":
                  return documentPrefix;
                case "{{document}}":
                  return documentSelection;
                case "{{documentSuffix}}":
                  return documentSuffix;
                case "{{command}}":
                  return userCommand;
                case "{{languageId}}":
                  return languageId;
                default:
                  return "";
              }
            },
          ),
        },
      ];
      const requestOptions = {
        body: { messages },
        signal: this.createAbortSignal(options),
        parseAs: "stream" as ParseAs,
      };
      const requestDescription = `POST ${this.config.server.endpoint + requestPath}`;
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
      if (error instanceof HttpError && error.status == 404 && !options?.useBetaVersion) {
        return await this.provideChatEdit(document, selection, filepath, insertMode, command, languageId, {
          ...options,
          useBetaVersion: true,
        });
      }
      if (isCanceledError(error)) {
        this.logger.debug(`Chat request canceled. [${requestId}]`);
      } else if (isUnauthorizedError(error)) {
        this.logger.debug(`Chat request failed due to unauthorized. [${requestId}]`);
      } else {
        this.logger.error(`Chat request failed. [${requestId}]`, error);
      }
      this.healthCheck(); // schedule a health check
      return null;
    }
  }
}
