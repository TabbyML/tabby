import axios from "axios";
import { EventEmitter } from "events";
import { v4 as uuid } from "uuid";
import deepEqual from "deep-equal";
import deepMerge from "deepmerge";
import { TabbyApi, CancelablePromise, ApiError, ChoiceEvent, CompletionEvent } from "./generated";
import { sleep, cancelable, splitLines, isBlank } from "./utils";
import { Agent, AgentEvent, AgentInitOptions, CompletionRequest, CompletionResponse } from "./Agent";
import { AgentConfig, defaultAgentConfig } from "./AgentConfig";
import { CompletionCache } from "./CompletionCache";
import { rootLogger, allLoggers } from "./logger";

export class TabbyAgent extends EventEmitter implements Agent {
  private readonly logger = rootLogger.child({ component: "TabbyAgent" });
  private config: AgentConfig = defaultAgentConfig;
  private status: "connecting" | "ready" | "disconnected" = "connecting";
  private api: TabbyApi;
  private completionCache: CompletionCache = new CompletionCache();

  constructor() {
    super();
    this.onConfigUpdated();
  }

  private onConfigUpdated() {
    allLoggers.forEach((logger) => (logger.level = this.config.logs.level));
    this.api = new TabbyApi({ BASE: this.config.server.endpoint });
    this.ping();
  }

  private changeStatus(status: "connecting" | "ready" | "disconnected") {
    if (this.status != status) {
      this.status = status;
      const event: AgentEvent = { event: "statusChanged", status };
      this.logger.debug({ event }, "Status changed");
      super.emit("statusChanged", event);
    }
  }

  private async ping(tries: number = 0): Promise<boolean> {
    try {
      await axios.get(this.config.server.endpoint);
      this.changeStatus("ready");
      return true;
    } catch (e) {
      if (tries > 5) {
        this.changeStatus("disconnected");
        return false;
      }
      this.changeStatus("connecting");
      const pingRetryDelay = 1000;
      await sleep(pingRetryDelay);
      return this.ping(tries + 1);
    }
  }

  private callApi<Request, Response>(
    api: (request: Request) => CancelablePromise<Response>,
    request: Request
  ): CancelablePromise<Response> {
    this.logger.debug({ api: api.name, request }, "API request");
    const promise = api.call(this.api.default, request);
    return cancelable(
      promise
        .then((response: Response) => {
          this.logger.debug({ api: api.name, response }, "API response");
          this.changeStatus("ready");
          return response;
        })
        .catch((error: ApiError) => {
          this.logger.error({ api: api.name, error }, "API error");
          this.changeStatus("disconnected");
          throw error;
        }),
      () => {
        promise.cancel();
      }
    );
  }

  private createPrompt(request: CompletionRequest): string {
    const maxLines = 20;
    const prefix = request.text.slice(0, request.position);
    const lines = splitLines(prefix);
    const cutoff = Math.max(lines.length - maxLines, 0);
    const prompt = lines.slice(cutoff).join("");
    return prompt;
  }

  public initialize(params: AgentInitOptions): boolean {
    if (params.config) {
      this.updateConfig(params.config);
    }
    if (params.client) {
      // Client info is only used in logging for now
      // `pino.Logger.setBindings` is not present in the browser
      allLoggers.forEach((logger) => logger.setBindings && logger.setBindings({ client: params.client }));
    }
    this.logger.debug({ params }, "Initialized");
    return true;
  }

  public updateConfig(config: AgentConfig): boolean {
    const mergedConfig = deepMerge(this.config, config);
    if (!deepEqual(this.config, mergedConfig)) {
      this.config = mergedConfig;
      this.onConfigUpdated();
      const event: AgentEvent = { event: "configUpdated", config: this.config };
      this.logger.debug({ event }, "Config updated");
      super.emit("configUpdated", event);
    }
    return true;
  }

  public getConfig(): AgentConfig {
    return this.config;
  }

  public getStatus(): "connecting" | "ready" | "disconnected" {
    return this.status;
  }

  public getCompletions(request: CompletionRequest): CancelablePromise<CompletionResponse> {
    if (this.completionCache.has(request)) {
      this.logger.debug({ request }, "Completion cache hit");
      return new CancelablePromise((resolve) => {
        resolve(this.completionCache.get(request));
      });
    }
    const prompt = this.createPrompt(request);
    if (isBlank(prompt)) {
      this.logger.debug("Prompt is blank, returning empty completion response");
      return new CancelablePromise((resolve) => {
        resolve({
          id: "agent-" + uuid(),
          created: new Date().getTime(),
          choices: [],
        });
      });
    }
    const promise = this.callApi(this.api.default.completionsV1CompletionsPost, {
      prompt,
      language: request.language,
    });
    return cancelable(
      promise.then((response: CompletionResponse) => {
        this.completionCache.set(request, response);
        return response;
      }),
      () => {
        promise.cancel();
      }
    );
  }

  public postEvent(request: ChoiceEvent | CompletionEvent): CancelablePromise<boolean> {
    return this.callApi(this.api.default.eventsV1EventsPost, request);
  }
}
