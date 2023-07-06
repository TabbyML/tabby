import { EventEmitter } from "events";
import { v4 as uuid } from "uuid";
import deepEqual from "deep-equal";
import deepMerge from "deepmerge";
import { TabbyApi, CancelablePromise } from "./generated";
import { cancelable, splitLines, isBlank } from "./utils";
import {
  Agent,
  AgentStatus,
  AgentEvent,
  AgentInitOptions,
  CompletionRequest,
  CompletionResponse,
  LogEventRequest,
} from "./Agent";
import { Auth } from "./Auth";
import { AgentConfig, defaultAgentConfig, userAgentConfig } from "./AgentConfig";
import { CompletionCache } from "./CompletionCache";
import { DataStore } from "./dataStore";
import { postprocess } from "./postprocess";
import { rootLogger, allLoggers } from "./logger";
import { AnonymousUsageLogger } from "./AnonymousUsageLogger";

/**
 * Different from AgentInitOptions or AgentConfig, this may contain non-serializable objects,
 * so it is not suitable for cli, but only used when imported as module by other js project.
 */
export type TabbyAgentOptions = {
  dataStore: DataStore;
};

export class TabbyAgent extends EventEmitter implements Agent {
  private readonly logger = rootLogger.child({ component: "TabbyAgent" });
  private anonymousUsageLogger: AnonymousUsageLogger;
  private config: AgentConfig = defaultAgentConfig;
  private userConfig: Partial<AgentConfig> = {}; // config from `~/.tabby/agent/config.toml`
  private clientConfig: Partial<AgentConfig> = {}; // config from `initialize` and `updateConfig` method
  private status: AgentStatus = "notInitialized";
  private api: TabbyApi;
  private auth: Auth;
  private dataStore: DataStore | null = null;
  private completionCache: CompletionCache = new CompletionCache();
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
  }

  static async create(options?: Partial<TabbyAgentOptions>): Promise<TabbyAgent> {
    const agent = new TabbyAgent();
    agent.dataStore = options?.dataStore;
    agent.anonymousUsageLogger = await AnonymousUsageLogger.create({ dataStore: options?.dataStore });
    return agent;
  }

  private async applyConfig() {
    this.config = deepMerge.all<AgentConfig>([defaultAgentConfig, this.userConfig, this.clientConfig]);
    allLoggers.forEach((logger) => (logger.level = this.config.logs.level));
    this.anonymousUsageLogger.disabled = this.config.anonymousUsageTracking.disable;
    if (this.config.server.endpoint !== this.auth?.endpoint) {
      this.auth = await Auth.create({ endpoint: this.config.server.endpoint, dataStore: this.dataStore });
      this.auth.on("updated", this.setupApi.bind(this));
    }
    await this.setupApi();
  }

  private async setupApi() {
    this.api = new TabbyApi({
      BASE: this.config.server.endpoint.replace(/\/+$/, ""), // remove trailing slash
      TOKEN: this.auth?.token,
    });
    await this.healthCheck();
  }

  private changeStatus(status: AgentStatus) {
    if (this.status != status) {
      this.status = status;
      const event: AgentEvent = { event: "statusChanged", status };
      this.logger.debug({ event }, "Status changed");
      super.emit("statusChanged", event);
    }
  }

  private callApi<Request, Response>(
    api: (request: Request) => CancelablePromise<Response>,
    request: Request
  ): CancelablePromise<Response> {
    this.logger.debug({ api: api.name, request }, "API request");
    const promise = api.call(this.api.v1, request);
    return cancelable(
      promise
        .then((response: Response) => {
          this.logger.debug({ api: api.name, response }, "API response");
          this.changeStatus("ready");
          return response;
        })
        .catch((error) => {
          if (!!error.isCancelled) {
            this.logger.debug({ api: api.name, error }, "API request canceled");
          } else if (error.name === "ApiError" && [401, 403, 405].indexOf(error.status) !== -1) {
            this.logger.debug({ api: api.name, error }, "API unauthorized");
            this.changeStatus("unauthorized");
          } else if (error.name === "ApiError") {
            this.logger.error({ api: api.name, error }, "API error");
            this.changeStatus("disconnected");
          } else {
            this.logger.error({ api: api.name, error }, "API request failed with unknown error");
            this.changeStatus("disconnected");
          }
          throw error;
        }),
      () => {
        promise.cancel();
      }
    );
  }

  private healthCheck(): Promise<any> {
    return this.callApi(this.api.v1.health, {}).catch(() => {});
  }

  private createSegments(request: CompletionRequest): { prefix: string; suffix: string } {
    // max lines in prefix and suffix configurable
    const maxPrefixLines = request.maxPrefixLines ?? this.config.completion.maxPrefixLines;
    const maxSuffixLines = request.maxSuffixLines ?? this.config.completion.maxSuffixLines;
    const prefix = request.text.slice(0, request.position);
    const prefixLines = splitLines(prefix);
    const suffix = request.text.slice(request.position);
    const suffixLines = splitLines(suffix);
    return {
      prefix: prefixLines.slice(Math.max(prefixLines.length - maxPrefixLines, 0)).join(""),
      suffix: suffixLines.slice(0, maxSuffixLines).join(""),
    };
  }

  public async initialize(options: Partial<AgentInitOptions>): Promise<boolean> {
    if (options.client) {
      // Client info is only used in logging for now
      // `pino.Logger.setBindings` is not present in the browser
      allLoggers.forEach((logger) => logger.setBindings?.({ client: options.client }));
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
      this.clientConfig = deepMerge(this.clientConfig, options.config);
    }
    await this.applyConfig();
    if (this.status === "unauthorized") {
      const event: AgentEvent = { event: "authRequired", server: this.config.server };
      super.emit("authRequired", event);
    }
    await this.anonymousUsageLogger.event("AgentInitialized", {
      client: options.client,
    });
    this.logger.debug({ options }, "Initialized");
    return this.status !== "notInitialized";
  }

  public async updateConfig(config: Partial<AgentConfig>): Promise<boolean> {
    const mergedConfig = deepMerge(this.clientConfig, config);
    if (!deepEqual(this.clientConfig, mergedConfig)) {
      const serverUpdated = !deepEqual(this.config.server, mergedConfig.server);
      this.clientConfig = mergedConfig;
      await this.applyConfig();
      const event: AgentEvent = { event: "configUpdated", config: this.config };
      this.logger.debug({ event }, "Config updated");
      super.emit("configUpdated", event);
      if (serverUpdated && this.status === "unauthorized") {
        const event: AgentEvent = { event: "authRequired", server: this.config.server };
        super.emit("authRequired", event);
      }
    }
    return true;
  }

  public getConfig(): AgentConfig {
    return this.config;
  }

  public getStatus(): AgentStatus {
    return this.status;
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
      }
    );
  }

  public getCompletions(request: CompletionRequest): CancelablePromise<CompletionResponse> {
    if (this.status === "notInitialized") {
      return cancelable(Promise.reject("Agent is not initialized"), () => {});
    }
    if (this.completionCache.has(request)) {
      this.logger.debug({ request }, "Completion cache hit");
      return new CancelablePromise((resolve) => {
        resolve(this.completionCache.get(request));
      });
    }
    const segments = this.createSegments(request);
    if (isBlank(segments.prefix)) {
      this.logger.debug("Segment prefix is blank, returning empty completion response");
      return new CancelablePromise((resolve) => {
        resolve({
          id: "agent-" + uuid(),
          choices: [],
        });
      });
    }
    const promise = this.callApi(this.api.v1.completion, {
      language: request.language,
      segments,
      user: this.auth?.user,
    });
    return cancelable(
      promise
        .then((response) => {
          this.completionCache.set(request, response);
          return response;
        })
        .then((response) => {
          return postprocess(request, response);
        }),
      () => {
        promise.cancel();
      }
    );
  }

  public postEvent(request: LogEventRequest): CancelablePromise<boolean> {
    if (this.status === "notInitialized") {
      return cancelable(Promise.reject("Agent is not initialized"), () => {});
    }
    return this.callApi(this.api.v1.event, request);
  }
}
