import { EventEmitter } from "events";
import { v4 as uuid } from "uuid";
import deepEqual from "deep-equal";
import { deepmerge } from "deepmerge-ts";
import { getProperty, setProperty, deleteProperty } from "dot-prop";
import createClient from "openapi-fetch";
import { paths as TabbyApi } from "./types/tabbyApi";
import { splitLines, isBlank, abortSignalFromAnyOf, HttpError, isTimeoutError, isCanceledError } from "./utils";
import type {
  Agent,
  AgentStatus,
  AgentIssue,
  AgentEvent,
  AgentInitOptions,
  AbortSignalOption,
  ServerHealthState,
  CompletionRequest,
  CompletionResponse,
  LogEventRequest,
} from "./Agent";
import { Auth } from "./Auth";
import { AgentConfig, PartialAgentConfig, defaultAgentConfig, userAgentConfig } from "./AgentConfig";
import { CompletionCache } from "./CompletionCache";
import { CompletionDebounce } from "./CompletionDebounce";
import { DataStore } from "./dataStore";
import { postprocess, preCacheProcess } from "./postprocess";
import { rootLogger, allLoggers } from "./logger";
import { AnonymousUsageLogger } from "./AnonymousUsageLogger";
import { ResponseStats, completionResponseTimeStatsStrategy } from "./ResponseStats";

/**
 * Different from AgentInitOptions or AgentConfig, this may contain non-serializable objects,
 * so it is not suitable for cli, but only used when imported as module by other js project.
 */
export type TabbyAgentOptions = {
  dataStore?: DataStore;
};

export class TabbyAgent extends EventEmitter implements Agent {
  private readonly logger = rootLogger.child({ component: "TabbyAgent" });
  private anonymousUsageLogger: AnonymousUsageLogger;
  private config: AgentConfig = defaultAgentConfig;
  private userConfig: PartialAgentConfig = {}; // config from `~/.tabby-client/agent/config.toml`
  private clientConfig: PartialAgentConfig = {}; // config from `initialize` and `updateConfig` method
  private status: AgentStatus = "notInitialized";
  private issues: AgentIssue["name"][] = [];
  private serverHealthState: ServerHealthState | null = null;
  private api: ReturnType<typeof createClient<TabbyApi>>;
  private auth: Auth;
  private dataStore: DataStore | null = null;
  private completionCache: CompletionCache = new CompletionCache();
  private completionDebounce: CompletionDebounce = new CompletionDebounce();
  private nonParallelProvideCompletionAbortController: AbortController | null = null;
  private completionResponseStats: ResponseStats = new ResponseStats(completionResponseTimeStatsStrategy);
  static readonly tryConnectInterval = 1000 * 30; // 30s
  private tryingConnectTimer: ReturnType<typeof setInterval> | null = null;

  private constructor() {
    super();

    this.tryingConnectTimer = setInterval(async () => {
      if (this.status === "disconnected") {
        this.logger.debug("Trying to connect...");
        await this.healthCheck();
      }
    }, TabbyAgent.tryConnectInterval);

    this.completionResponseStats.on("healthy", () => {
      this.popIssue("slowCompletionResponseTime");
      this.popIssue("highCompletionTimeoutRate");
    });
    this.completionResponseStats.on("highTimeoutRate", () => {
      if (this.status === "ready" || this.status === "issuesExist") {
        this.popIssue("slowCompletionResponseTime");
        this.pushIssue("highCompletionTimeoutRate");
      }
    });
    this.completionResponseStats.on("slowResponseTime", () => {
      if (this.status === "ready" || this.status === "issuesExist") {
        this.popIssue("highCompletionTimeoutRate");
        this.pushIssue("slowCompletionResponseTime");
      }
    });
  }

  static async create(options?: TabbyAgentOptions): Promise<TabbyAgent> {
    const agent = new TabbyAgent();
    agent.dataStore = options?.dataStore;
    agent.anonymousUsageLogger = await AnonymousUsageLogger.create({ dataStore: options?.dataStore });
    return agent;
  }

  private async applyConfig() {
    this.config = deepmerge(defaultAgentConfig, this.userConfig, this.clientConfig);
    allLoggers.forEach((logger) => (logger.level = this.config.logs.level));
    this.anonymousUsageLogger.disabled = this.config.anonymousUsageTracking.disable;
    if (this.config.server.requestHeaders["Authorization"] === undefined) {
      if (this.config.server.endpoint !== this.auth?.endpoint) {
        this.auth = await Auth.create({ endpoint: this.config.server.endpoint, dataStore: this.dataStore });
        this.auth.on("updated", this.setupApi.bind(this));
      }
    } else {
      // If `Authorization` request header is provided, use it directly.
      this.auth = null;
    }
    await this.setupApi();
  }

  private async setupApi() {
    this.api = createClient<TabbyApi>({
      baseUrl: this.config.server.endpoint.replace(/\/+$/, ""), // remove trailing slash
      headers: {
        Authorization: this.auth?.token ? `Bearer ${this.auth.token}` : undefined,
        ...this.config.server.requestHeaders,
      },
    });
    await this.healthCheck();
  }

  private changeStatus(status: AgentStatus) {
    if (this.status != status) {
      this.status = status;
      const event: AgentEvent = { event: "statusChanged", status };
      this.logger.debug({ event }, "Status changed");
      super.emit("statusChanged", event);
      if (this.status === "unauthorized") {
        this.emitAuthRequired();
      }
    }
  }

  private issueWithDetails(issue: AgentIssue["name"]): AgentIssue {
    switch (issue) {
      case "highCompletionTimeoutRate":
        return {
          name: "highCompletionTimeoutRate",
          completionResponseStats: this.completionResponseStats.stats(),
        };
      case "slowCompletionResponseTime":
        return {
          name: "slowCompletionResponseTime",
          completionResponseStats: this.completionResponseStats.stats(),
        };
    }
  }

  private pushIssue(issue: AgentIssue["name"]) {
    if (this.issues.indexOf(issue) === -1) {
      this.issues.push(issue);
      this.changeStatus("issuesExist");
      const event: AgentEvent = { event: "newIssue", issue: this.issueWithDetails(issue) };
      this.logger.debug({ event }, "New issue");
      super.emit("newIssue", event);
    }
  }

  private popIssue(issue: AgentIssue["name"]) {
    this.issues = this.issues.filter((i) => i !== issue);
    if (this.issues.length === 0 && this.status === "issuesExist") {
      this.changeStatus("ready");
    }
  }

  private emitAuthRequired() {
    const event: AgentEvent = { event: "authRequired", server: this.config.server };
    super.emit("authRequired", event);
  }

  private async post<T extends Parameters<typeof this.api.POST>[0]>(
    path: T,
    requestOptions: Parameters<typeof this.api.POST<T>>[1],
    abortOptions?: { signal?: AbortSignal; timeout?: number },
  ): Promise<Awaited<ReturnType<typeof this.api.POST<T>>>["data"]> {
    const requestId = uuid();
    this.logger.debug({ requestId, path, requestOptions, abortOptions }, "API request");
    try {
      const timeout = Math.min(0x7fffffff, abortOptions?.timeout || this.config.server.requestTimeout);
      const signal = abortSignalFromAnyOf([AbortSignal.timeout(timeout), abortOptions?.signal]);
      const response = await this.api.POST(path, { ...requestOptions, signal });
      if (response.error) {
        throw new HttpError(response.response);
      }
      this.logger.debug({ requestId, path, response: response.data }, "API response");
      if (this.status !== "issuesExist") {
        this.changeStatus("ready");
      }
      return response.data;
    } catch (error) {
      if (isTimeoutError(error)) {
        this.logger.debug({ requestId, path, error }, "API request timeout");
      } else if (isCanceledError(error)) {
        this.logger.debug({ requestId, path, error }, "API request canceled");
      } else if (
        error instanceof HttpError &&
        [401, 403, 405].indexOf(error.status) !== -1 &&
        new URL(this.config.server.endpoint).hostname.endsWith("app.tabbyml.com") &&
        this.config.server.requestHeaders["Authorization"] === undefined
      ) {
        this.logger.debug({ requestId, path, error }, "API unauthorized");
        this.changeStatus("unauthorized");
      } else if (error instanceof HttpError) {
        this.logger.error({ requestId, path, error }, "API error");
        this.changeStatus("disconnected");
      } else {
        this.logger.error({ requestId, path, error }, "API request failed with unknown error");
        this.changeStatus("disconnected");
      }
      throw error;
    }
  }

  private async healthCheck(options?: AbortSignalOption): Promise<any> {
    try {
      const healthState = await this.post("/v1/health", {}, options);
      if (
        typeof healthState === "object" &&
        healthState["model"] !== undefined &&
        healthState["device"] !== undefined
      ) {
        this.serverHealthState = healthState;
        if (this.status === "ready") {
          this.anonymousUsageLogger.uniqueEvent("AgentConnected", healthState);
        }
      }
    } catch (_) {
      // ignore
    }
  }

  private createSegments(request: CompletionRequest): { prefix: string; suffix: string } {
    // max lines in prefix and suffix configurable
    const maxPrefixLines = this.config.completion.prompt.maxPrefixLines;
    const maxSuffixLines = this.config.completion.prompt.maxSuffixLines;
    const prefix = request.text.slice(0, request.position);
    const prefixLines = splitLines(prefix);
    const suffix = request.text.slice(request.position);
    const suffixLines = splitLines(suffix);
    return {
      prefix: prefixLines.slice(Math.max(prefixLines.length - maxPrefixLines, 0)).join(""),
      suffix: suffixLines.slice(0, maxSuffixLines).join(""),
    };
  }

  public async initialize(options: AgentInitOptions): Promise<boolean> {
    if (options.client || options.clientProperties) {
      // Client info is only used in logging for now
      // `pino.Logger.setBindings` is not present in the browser
      allLoggers.forEach((logger) => logger.setBindings?.({ client: options.client, ...options.clientProperties }));
      this.anonymousUsageLogger.addProperties({ client: options.client, ...options.clientProperties });
    }
    if (userAgentConfig) {
      await userAgentConfig.load();
      this.userConfig = userAgentConfig.config;
      userAgentConfig.on("updated", async (config) => {
        this.userConfig = config;
        await this.applyConfig();
      });
      userAgentConfig.watch();
    }
    if (options.config) {
      this.clientConfig = options.config;
    }
    await this.applyConfig();
    await this.anonymousUsageLogger.uniqueEvent("AgentInitialized");
    this.logger.debug({ options }, "Initialized");
    return this.status !== "notInitialized";
  }

  public async updateConfig(key: string, value: any): Promise<boolean> {
    const current = getProperty(this.clientConfig, key);
    if (!deepEqual(current, value)) {
      if (value === undefined) {
        deleteProperty(this.clientConfig, key);
      } else {
        setProperty(this.clientConfig, key, value);
      }
      const prevStatus = this.status;
      await this.applyConfig();
      // If server config changed, clear server health state
      if (key.startsWith("server")) {
        this.serverHealthState = null;
      }
      // If status unchanged, `authRequired` will not be emitted when `applyConfig`,
      // so we need to emit it manually.
      if (key.startsWith("server") && prevStatus === "unauthorized" && this.status === "unauthorized") {
        this.emitAuthRequired();
      }
      const event: AgentEvent = { event: "configUpdated", config: this.config };
      this.logger.debug({ event }, "Config updated");
      super.emit("configUpdated", event);
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

  public getIssues(): AgentIssue[] {
    return this.issues.map((issue) => this.issueWithDetails(issue));
  }

  public getServerHealthState(): ServerHealthState | null {
    return this.serverHealthState;
  }

  public async requestAuthUrl(options?: AbortSignalOption): Promise<{ authUrl: string; code: string } | null> {
    if (this.status === "notInitialized") {
      throw new Error("Agent is not initialized");
    }
    await this.healthCheck(options);
    if (this.status !== "unauthorized") {
      return null;
    } else {
      return await this.auth.requestAuthUrl(options);
    }
  }

  public async waitForAuthToken(code: string, options?: AbortSignalOption): Promise<void> {
    if (this.status === "notInitialized") {
      throw new Error("Agent is not initialized");
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
    if (this.nonParallelProvideCompletionAbortController) {
      this.nonParallelProvideCompletionAbortController.abort();
    }
    this.nonParallelProvideCompletionAbortController = new AbortController();
    const signal = abortSignalFromAnyOf([this.nonParallelProvideCompletionAbortController.signal, options?.signal]);
    let completionResponse: CompletionResponse | null = null;
    if (this.completionCache.has(request)) {
      // Hit cache
      this.logger.debug({ request }, "Completion cache hit");
      await this.completionDebounce.debounce(
        {
          request,
          config: this.config.completion.debounce,
          responseTime: 0,
        },
        { signal },
      );
      completionResponse = this.completionCache.get(request);
    } else {
      // No cache
      const segments = this.createSegments(request);
      if (isBlank(segments.prefix)) {
        // Empty prompt
        this.logger.debug("Segment prefix is blank, returning empty completion response");
        completionResponse = {
          id: "agent-" + uuid(),
          choices: [],
        };
      } else {
        // Request server
        await this.completionDebounce.debounce(
          {
            request,
            config: this.config.completion.debounce,
            responseTime: this.completionResponseStats.stats()["averageResponseTime"],
          },
          options,
        );

        const requestStartedAt = performance.now();
        const apiPath = "/v1/completions";
        try {
          completionResponse = await this.post(
            apiPath,
            {
              body: {
                language: request.language,
                segments,
                user: this.auth?.user,
              },
            },
            {
              signal,
              timeout: request.manually ? this.config.completion.timeout.manually : this.config.completion.timeout.auto,
            },
          );
          this.completionResponseStats.push({
            name: apiPath,
            status: 200,
            responseTime: performance.now() - requestStartedAt,
          });
        } catch (error) {
          // record timed out request in stats, do not record canceled request
          if (isTimeoutError(error)) {
            this.completionResponseStats.push({
              name: apiPath,
              status: error.status,
              responseTime: performance.now() - requestStartedAt,
              error,
            });
          }
        }
        completionResponse = await preCacheProcess(request, completionResponse);
        if (options?.signal?.aborted) {
          throw options.signal.reason;
        }
        this.completionCache.set(request, completionResponse);
      }
    }
    completionResponse = await postprocess(request, completionResponse);
    if (options?.signal?.aborted) {
      throw options.signal.reason;
    }
    return completionResponse;
  }

  public async postEvent(request: LogEventRequest, options?: AbortSignalOption): Promise<boolean> {
    if (this.status === "notInitialized") {
      throw new Error("Agent is not initialized");
    }
    await this.post("/v1/events", { body: request, parseAs: "text" }, options);
    return true;
  }
}
