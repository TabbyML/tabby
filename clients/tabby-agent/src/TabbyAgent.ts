import { EventEmitter } from "events";
import { v4 as uuid } from "uuid";
import deepEqual from "deep-equal";
import { deepmerge } from "deepmerge-ts";
import { getProperty, setProperty, deleteProperty } from "dot-prop";
import { TabbyApi, CancelablePromise } from "./generated";
import { cancelable, splitLines, isBlank } from "./utils";
import {
  Agent,
  AgentStatus,
  AgentIssue,
  AgentEvent,
  AgentInitOptions,
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
  private api: TabbyApi;
  private auth: Auth;
  private dataStore: DataStore | null = null;
  private completionCache: CompletionCache = new CompletionCache();
  private CompletionDebounce: CompletionDebounce = new CompletionDebounce();
  static readonly tryConnectInterval = 1000 * 30; // 30s
  private tryingConnectTimer: ReturnType<typeof setInterval> | null = null;
  private completionResponseStats: ResponseStats = new ResponseStats(completionResponseTimeStatsStrategy);

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
      this.auth = null;
    }
    await this.setupApi();
  }

  private async setupApi() {
    this.api = new TabbyApi({
      BASE: this.config.server.endpoint.replace(/\/+$/, ""), // remove trailing slash
      TOKEN: this.auth?.token,
      HEADERS: this.config.server.requestHeaders,
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

  private callApi<Request, Response>(
    api: (request: Request) => CancelablePromise<Response>,
    request: Request,
    options: { timeout?: number } = { timeout: this.config.server.requestTimeout },
  ): CancelablePromise<Response> {
    return new CancelablePromise((resolve, reject, onCancel) => {
      const requestId = uuid();
      this.logger.debug({ requestId, api: api.name, request }, "API request");
      let timeout: ReturnType<typeof setTimeout> | null = null;
      let timeoutCancelled = false;
      const apiRequest = api.call(this.api.v1, request);
      const requestStartedAt = performance.now();
      apiRequest
        .then((response: Response) => {
          this.logger.debug({ requestId, api: api.name, response }, "API response");
          if (this.status !== "issuesExist") {
            this.changeStatus("ready");
          }
          if (api.name === "completion") {
            this.completionResponseStats.push({
              name: api.name,
              status: 200,
              responseTime: performance.now() - requestStartedAt,
            });
          }
          if (timeout) {
            clearTimeout(timeout);
          }
          resolve(response);
        })
        .catch((error) => {
          if (
            (!!error.isCancelled && timeoutCancelled) ||
            (!error.isCancelled && error.code === "ECONNABORTED") ||
            (error.name === "ApiError" && [408, 499].indexOf(error.status) !== -1)
          ) {
            error.isTimeoutError = true;
            this.logger.debug({ requestId, api: api.name, error }, "API request timeout");
          } else if (!!error.isCancelled) {
            this.logger.debug({ requestId, api: api.name, error }, "API request cancelled");
          } else if (
            error.name === "ApiError" &&
            [401, 403, 405].indexOf(error.status) !== -1 &&
            new URL(this.config.server.endpoint).hostname.endsWith("app.tabbyml.com") &&
            this.config.server.requestHeaders["Authorization"] === undefined
          ) {
            this.logger.debug({ requestId, api: api.name, error }, "API unauthorized");
            this.changeStatus("unauthorized");
          } else if (error.name === "ApiError") {
            this.logger.error({ requestId, api: api.name, error }, "API error");
            this.changeStatus("disconnected");
          } else {
            this.logger.error({ requestId, api: api.name, error }, "API request failed with unknown error");
            this.changeStatus("disconnected");
          }
          // don't record cancelled request in stats
          if (api.name === "completion" && (error.isTimeoutError || !error.isCancelled)) {
            this.completionResponseStats.push({
              name: api.name,
              status: error.status,
              responseTime: performance.now() - requestStartedAt,
              error,
            });
          }
          if (timeout) {
            clearTimeout(timeout);
          }
          reject(error);
        });
      // It seems that openapi-typescript-codegen does not provide timeout options passing to axios,
      // Just use setTimeout to cancel the request manually.
      if (options.timeout && options.timeout > 0) {
        timeout = setTimeout(
          () => {
            this.logger.debug({ api: api.name, timeout: options.timeout }, "Cancel API request due to timeout");
            timeoutCancelled = true;
            apiRequest.cancel();
          },
          Math.min(options.timeout, 0x7fffffff),
        );
      }
      onCancel(() => {
        if (timeout) {
          clearTimeout(timeout);
        }
        apiRequest.cancel();
      });
    });
  }

  private healthCheck(): Promise<any> {
    return this.callApi(this.api.v1.health, {})
      .then((healthState) => {
        this.serverHealthState = healthState;
        if (this.status === "ready") {
          this.anonymousUsageLogger.uniqueEvent("AgentConnected", healthState);
        }
      })
      .catch(() => {});
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
    if (options.client) {
      // Client info is only used in logging for now
      // `pino.Logger.setBindings` is not present in the browser
      allLoggers.forEach((logger) => logger.setBindings?.({ client: options.client }));
      this.anonymousUsageLogger.addProperties({ client: options.client });
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

  public requestAuthUrl(): CancelablePromise<{ authUrl: string; code: string } | null> {
    if (this.status === "notInitialized") {
      return cancelable(Promise.reject("Agent is not initialized"), () => {});
    }
    return new CancelablePromise(async (resolve, reject, onCancel) => {
      let request: CancelablePromise<{ authUrl: string; code: string }>;
      onCancel(() => {
        request?.cancel();
      });
      await this.healthCheck();
      if (onCancel.isCancelled) return;
      if (this.status === "unauthorized") {
        request = this.auth.requestAuthUrl();
        resolve(request);
      } else {
      }
      resolve(null);
    });
  }

  public waitForAuthToken(code: string): CancelablePromise<any> {
    if (this.status === "notInitialized") {
      return cancelable(Promise.reject("Agent is not initialized"), () => {});
    }
    const polling = this.auth.pollingToken(code);
    return cancelable(
      polling.then(() => {
        return this.setupApi();
      }),
      () => {
        polling.cancel();
      },
    );
  }

  public provideCompletions(request: CompletionRequest): CancelablePromise<CompletionResponse> {
    if (this.status === "notInitialized") {
      return cancelable(Promise.reject("Agent is not initialized"), () => {});
    }
    const cancelableList: CancelablePromise<any>[] = [];
    return cancelable(
      Promise.resolve(null)
        // From cache
        .then(async (response: CompletionResponse | null) => {
          if (response) return response;
          if (this.completionCache.has(request)) {
            this.logger.debug({ request }, "Completion cache hit");
            const debounce = this.CompletionDebounce.debounce(request, this.config.completion.debounce, 0);
            cancelableList.push(debounce);
            await debounce;
            return this.completionCache.get(request);
          }
          return null;
        })
        // From api
        .then(async (response: CompletionResponse | null) => {
          if (response) return response;
          const segments = this.createSegments(request);
          if (isBlank(segments.prefix)) {
            this.logger.debug("Segment prefix is blank, returning empty completion response");
            return {
              id: "agent-" + uuid(),
              choices: [],
            };
          }
          const debounce = this.CompletionDebounce.debounce(
            request,
            this.config.completion.debounce,
            this.completionResponseStats.stats()["averageResponseTime"],
          );
          cancelableList.push(debounce);
          await debounce;
          const apiRequest = this.callApi(
            this.api.v1.completion,
            {
              language: request.language,
              segments,
              user: this.auth?.user,
            },
            {
              timeout: request.manually ? this.config.completion.timeout.manually : this.config.completion.timeout.auto,
            },
          );
          cancelableList.push(apiRequest);
          let res = await apiRequest;
          res = await preCacheProcess(request, res);
          this.completionCache.set(request, res);
          return res;
        })
        // Postprocess
        .then(async (response: CompletionResponse | null) => {
          return postprocess(request, response);
        }),
      () => {
        cancelableList.forEach((cancelable) => cancelable.cancel());
      },
    );
  }

  public postEvent(request: LogEventRequest): CancelablePromise<boolean> {
    if (this.status === "notInitialized") {
      return cancelable(Promise.reject("Agent is not initialized"), () => {});
    }
    return this.callApi(this.api.v1.event, request);
  }
}
