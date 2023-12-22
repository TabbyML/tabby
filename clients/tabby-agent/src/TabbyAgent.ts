import { EventEmitter } from "events";
import { v4 as uuid } from "uuid";
import deepEqual from "deep-equal";
import { deepmerge } from "deepmerge-ts";
import { getProperty, setProperty, deleteProperty } from "dot-prop";
import createClient from "openapi-fetch";
import type { ParseAs } from "openapi-fetch";
import type { paths as TabbyApi } from "./types/tabbyApi";
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
import type { DataStore } from "./dataStore";
import { isBlank, abortSignalFromAnyOf, HttpError, isTimeoutError, isCanceledError, errorToString } from "./utils";
import { Auth } from "./Auth";
import { AgentConfig, PartialAgentConfig, defaultAgentConfig } from "./AgentConfig";
import { configFile } from "./configFile";
import { CompletionCache } from "./CompletionCache";
import { CompletionDebounce } from "./CompletionDebounce";
import { CompletionContext } from "./CompletionContext";
import { preCacheProcess, postCacheProcess, calculateReplaceRange } from "./postprocess";
import { rootLogger, allLoggers } from "./logger";
import { AnonymousUsageLogger } from "./AnonymousUsageLogger";
import { CompletionProviderStats, CompletionProviderStatsEntry } from "./CompletionProviderStats";

export class TabbyAgent extends EventEmitter implements Agent {
  private readonly logger = rootLogger.child({ component: "TabbyAgent" });
  private anonymousUsageLogger = new AnonymousUsageLogger();
  private config: AgentConfig = defaultAgentConfig;
  private userConfig: PartialAgentConfig = {}; // config from `~/.tabby-client/agent/config.toml`
  private clientConfig: PartialAgentConfig = {}; // config from `initialize` and `updateConfig` method
  private status: AgentStatus = "notInitialized";
  private issues: AgentIssue["name"][] = [];
  private serverHealthState?: ServerHealthState;
  private connectionErrorMessage?: string;
  private dataStore?: DataStore;
  private api?: ReturnType<typeof createClient<TabbyApi>>;
  private auth?: Auth;
  private completionCache = new CompletionCache();
  private completionDebounce = new CompletionDebounce();
  private nonParallelProvideCompletionAbortController?: AbortController;
  private completionProviderStats = new CompletionProviderStats();
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
    const oldConfig = this.config;
    const oldStatus = this.status;

    this.config = deepmerge(defaultAgentConfig, this.userConfig, this.clientConfig) as AgentConfig;
    allLoggers.forEach((logger) => (logger.level = this.config.logs.level));
    this.anonymousUsageLogger.disabled = this.config.anonymousUsageTracking.disable;

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
      this.completionProviderStats.resetWindowed();
      this.popIssue("slowCompletionResponseTime");
      this.popIssue("highCompletionTimeoutRate");
      this.popIssue("connectionFailed");
      this.connectionErrorMessage = undefined;
    }

    await this.setupApi();

    if (!deepEqual(oldConfig.server, this.config.server)) {
      // If server config changed and status remain `unauthorized`, we want to emit `authRequired` again.
      // but `changeStatus` will not emit `authRequired` if status is not changed, so we emit it manually here.
      if (oldStatus === "unauthorized" && this.status === "unauthorized") {
        this.emitAuthRequired();
      }
    }

    if (oldConfig.completion.timeout !== this.config.completion.timeout) {
      this.completionProviderStats.updateConfigByRequestTimeout(this.config.completion.timeout);
      this.popIssue("slowCompletionResponseTime");
      this.popIssue("highCompletionTimeoutRate");
    }

    const event: AgentEvent = { event: "configUpdated", config: this.config };
    this.logger.debug({ event }, "Config updated");
    super.emit("configUpdated", event);
  }

  private async setupApi() {
    const auth = !isBlank(this.config.server.token)
      ? `Bearer ${this.config.server.token}`
      : this.auth?.token
        ? `Bearer ${this.auth.token}`
        : undefined;
    this.api = createClient<TabbyApi>({
      baseUrl: this.config.server.endpoint.replace(/\/+$/, ""), // remove trailing slash
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
      const event: AgentEvent = { event: "statusChanged", status };
      this.logger.debug({ event }, "Status changed");
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
          completionResponseStats: this.completionProviderStats.windowed().stats,
        };
      case "slowCompletionResponseTime":
        return {
          name: "slowCompletionResponseTime",
          completionResponseStats: this.completionProviderStats.windowed().stats,
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
      this.logger.debug({ issue }, "Issues Pushed");
      this.emitIssueUpdated();
    }
  }

  private popIssue(issue: AgentIssue["name"]) {
    const index = this.issues.indexOf(issue);
    if (index >= 0) {
      this.issues.splice(index, 1);
      this.logger.debug({ issue }, "Issues Popped");
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
    const stats = this.completionProviderStats.stats();
    if (stats.completion_request.count > 0) {
      await this.anonymousUsageLogger.event("AgentStats", { stats });
      this.completionProviderStats.reset();
      this.logger.debug({ stats }, "Stats submitted");
    }
  }

  private createAbortSignal(options?: { signal?: AbortSignal; timeout?: number }): AbortSignal {
    const timeout = Math.min(0x7fffffff, options?.timeout || this.config.server.requestTimeout);
    return abortSignalFromAnyOf([AbortSignal.timeout(timeout), options?.signal]);
  }

  private async healthCheck(options?: AbortSignalOption): Promise<void> {
    const requestId = uuid();
    const requestPath = "/v1/health";
    const requestUrl = this.config.server.endpoint + requestPath;
    const requestOptions = {
      signal: this.createAbortSignal(options),
    };
    try {
      if (!this.api) {
        throw new Error("http client not initialized");
      }
      this.logger.debug({ requestId, requestOptions, url: requestUrl }, "Health check request");
      const response = await this.api.GET(requestPath, requestOptions);
      if (response.error || !response.response.ok) {
        throw new HttpError(response.response);
      }
      this.logger.debug({ requestId, response }, "Health check response");
      this.changeStatus("ready");
      this.popIssue("connectionFailed");
      this.connectionErrorMessage = undefined;
      const healthState = response.data;
      if (
        typeof healthState === "object" &&
        healthState["model"] !== undefined &&
        healthState["device"] !== undefined
      ) {
        this.serverHealthState = healthState;
        this.anonymousUsageLogger.uniqueEvent("AgentConnected", healthState);
      }
    } catch (error) {
      this.serverHealthState = undefined;
      if (error instanceof HttpError && [401, 403, 405].includes(error.status)) {
        this.logger.debug({ requestId, error }, "Health check error: unauthorized");
        this.changeStatus("unauthorized");
      } else {
        if (isTimeoutError(error)) {
          this.logger.debug({ requestId, error }, "Health check error: timeout");
          this.connectionErrorMessage = `GET ${requestUrl}: Timed out.`;
        } else if (isCanceledError(error)) {
          this.logger.debug({ requestId, error }, "Health check error: canceled");
          this.connectionErrorMessage = `GET ${requestUrl}: Canceled.`;
        } else {
          this.logger.error({ requestId, error }, "Health check error: unknown error");
          const message = error instanceof Error ? errorToString(error) : JSON.stringify(error);
          this.connectionErrorMessage = `GET ${requestUrl}: Request failed: \n${message}`;
        }
        this.pushIssue("connectionFailed");
        this.changeStatus("disconnected");
      }
    }
  }

  private createSegments(context: CompletionContext): { prefix: string; suffix: string; clipboard?: string } {
    // max lines in prefix and suffix configurable
    const maxPrefixLines = this.config.completion.prompt.maxPrefixLines;
    const maxSuffixLines = this.config.completion.prompt.maxSuffixLines;
    const { prefixLines, suffixLines } = context;
    const prefix = prefixLines.slice(Math.max(prefixLines.length - maxPrefixLines, 0)).join("");
    let suffix;
    if (this.config.completion.prompt.experimentalStripAutoClosingCharacters && context.mode !== "fill-in-line") {
      suffix = "\n" + suffixLines.slice(1, maxSuffixLines).join("");
    } else {
      suffix = suffixLines.slice(0, maxSuffixLines).join("");
    }

    let clipboard = undefined;
    const clipboardConfig = this.config.completion.prompt.clipboard;
    if (context.clipboard.length >= clipboardConfig.minChars && context.clipboard.length <= clipboardConfig.maxChars) {
      clipboard = context.clipboard;
    }
    return { prefix, suffix, clipboard };
  }

  public async initialize(options: AgentInitOptions): Promise<boolean> {
    this.dataStore = options?.dataStore;
    await this.anonymousUsageLogger.init({ dataStore: this.dataStore });
    if (options.clientProperties) {
      const { user: userProp, session: sessionProp } = options.clientProperties;
      allLoggers.forEach((logger) => logger.setBindings?.({ ...sessionProp }));
      if (sessionProp) {
        Object.entries(sessionProp).forEach(([key, value]) => {
          this.anonymousUsageLogger.setSessionProperties(key, value);
        });
      }
      if (userProp) {
        Object.entries(userProp).forEach(([key, value]) => {
          this.anonymousUsageLogger.setUserProperties(key, value);
        });
      }
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
    await this.applyConfig();
    await this.anonymousUsageLogger.uniqueEvent("AgentInitialized");
    this.logger.debug({ options }, "Initialized");
    return this.status !== "notInitialized";
  }

  public async finalize(): Promise<boolean> {
    if (this.status === "finalized") {
      return false;
    }

    await this.submitStats();

    if (this.tryingConnectTimer) {
      clearInterval(this.tryingConnectTimer);
    }
    if (this.submitStatsTimer) {
      clearInterval(this.submitStatsTimer);
    }
    this.changeStatus("finalized");
    return true;
  }

  public async updateClientProperties(type: keyof ClientProperties, key: string, value: any): Promise<boolean> {
    switch (type) {
      case "session":
        allLoggers.forEach((logger) => logger.setBindings?.(setProperty({}, key, value)));
        this.anonymousUsageLogger.setSessionProperties(key, value);
        break;
      case "user":
        this.anonymousUsageLogger.setUserProperties(key, value);
        break;
    }
    return true;
  }

  public async updateConfig(key: string, value: any): Promise<boolean> {
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
    this.logger.trace({ request }, "Call provideCompletions");
    if (this.nonParallelProvideCompletionAbortController) {
      this.nonParallelProvideCompletionAbortController.abort();
    }
    this.nonParallelProvideCompletionAbortController = new AbortController();
    const signal = abortSignalFromAnyOf([this.nonParallelProvideCompletionAbortController.signal, options?.signal]);

    let completionResponse: CompletionResponse;
    let stats: CompletionProviderStatsEntry | undefined = {
      triggerMode: request.manually ? "manual" : "auto",
      cacheHit: false,
      aborted: false,
      requestSent: false,
      requestLatency: 0,
      requestCanceled: false,
      requestTimeout: false,
    };
    let requestStartedAt: number | undefined;

    const context = new CompletionContext(request);
    try {
      if (this.completionCache.has(context)) {
        // Cache hit
        stats.cacheHit = true;
        this.logger.debug({ context }, "Completion cache hit");
        // Debounce before returning cached response
        await this.completionDebounce.debounce(
          {
            request,
            config: this.config.completion.debounce,
            responseTime: 0,
          },
          { signal },
        );

        completionResponse = this.completionCache.get(context)!;
      } else {
        // Cache miss
        stats.cacheHit = false;
        const segments = this.createSegments(context);
        if (isBlank(segments.prefix)) {
          // Empty prompt
          stats = undefined; // no need to record stats for empty prompt
          this.logger.debug("Segment prefix is blank, returning empty completion response");
          completionResponse = {
            id: "agent-" + uuid(),
            choices: [],
          };
        } else {
          // Debounce before sending request
          await this.completionDebounce.debounce(
            {
              request,
              config: this.config.completion.debounce,
              responseTime: this.completionProviderStats.windowed().stats.averageResponseTime,
            },
            options,
          );

          // Send http request
          const requestId = uuid();
          stats.requestSent = true;
          requestStartedAt = performance.now();
          try {
            if (!this.api) {
              throw new Error("http client not initialized");
            }
            const requestPath = "/v1/completions";
            const requestOptions = {
              body: {
                language: request.language,
                segments,
                user: this.auth?.user,
              },
              signal: this.createAbortSignal({
                signal,
                timeout: this.config.completion.timeout,
              }),
            };
            this.logger.debug(
              { requestId, requestOptions, url: this.config.server.endpoint + requestPath },
              "Completion request",
            );
            const response = await this.api.POST(requestPath, requestOptions);
            if (response.error || !response.response.ok) {
              throw new HttpError(response.response);
            }
            this.logger.debug({ requestId, response }, "Completion response");
            const responseData = response.data;
            stats.requestLatency = performance.now() - requestStartedAt;
            completionResponse = {
              id: responseData.id,
              choices: responseData.choices.map((choice) => {
                return {
                  index: choice.index,
                  text: choice.text,
                  replaceRange: {
                    start: request.position,
                    end: request.position,
                  },
                };
              }),
            };
          } catch (error) {
            if (isCanceledError(error)) {
              this.logger.debug({ requestId, error }, "Completion request canceled");
              stats.requestCanceled = true;
              stats.requestLatency = performance.now() - requestStartedAt;
            } else if (isTimeoutError(error)) {
              this.logger.debug({ requestId, error }, "Completion request timeout");
              stats.requestTimeout = true;
              stats.requestLatency = NaN;
            } else {
              this.logger.error({ requestId, error }, "Completion request failed with unknown error");
              // schedule a health check
              this.healthCheck();
            }
            // rethrow error
            throw error;
          }
          // Postprocess (pre-cache)
          completionResponse = await preCacheProcess(context, this.config.postprocess, completionResponse);
          if (signal.aborted) {
            throw signal.reason;
          }
          // Build cache
          this.completionCache.buildCache(context, JSON.parse(JSON.stringify(completionResponse)));
        }
      }
      // Postprocess (post-cache)
      completionResponse = await postCacheProcess(context, this.config.postprocess, completionResponse);
      if (signal.aborted) {
        throw signal.reason;
      }
      // Calculate replace range
      completionResponse = await calculateReplaceRange(context, this.config.postprocess, completionResponse);
      if (signal.aborted) {
        throw signal.reason;
      }
    } catch (error) {
      if (isCanceledError(error) || isTimeoutError(error)) {
        if (stats) {
          stats.aborted = true;
        }
      } else {
        // unexpected error
        stats = undefined;
      }
      // rethrow error
      throw error;
    } finally {
      if (stats) {
        this.completionProviderStats.add(stats);

        if (stats.requestSent && !stats.requestCanceled) {
          const windowedStats = this.completionProviderStats.windowed();
          const checkResult = this.completionProviderStats.check(windowedStats);
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
    this.logger.trace({ context, completionResponse }, "Return from provideCompletions");
    return completionResponse;
  }

  public async postEvent(request: LogEventRequest, options?: AbortSignalOption): Promise<boolean> {
    if (this.status === "notInitialized") {
      throw new Error("Agent is not initialized");
    }
    this.completionProviderStats.addEvent(request.type);
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
      this.logger.debug({ requestId, requestOptions, url: this.config.server.endpoint + requestPath }, "Event request");
      const response = await this.api.POST(requestPath, requestOptions);
      if (response.error || !response.response.ok) {
        throw new HttpError(response.response);
      }
      this.logger.debug({ requestId, response }, "Event response");
      return true;
    } catch (error) {
      if (isTimeoutError(error)) {
        this.logger.debug({ requestId, error }, "Event request timeout");
      } else if (isCanceledError(error)) {
        this.logger.debug({ requestId, error }, "Event request canceled");
      } else {
        this.logger.error({ requestId, error }, "Event request failed with unknown error");
      }
      return false;
    }
  }
}
