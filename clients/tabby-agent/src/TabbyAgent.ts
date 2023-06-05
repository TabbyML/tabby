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

export class TabbyAgent extends EventEmitter implements Agent {
  private config: AgentConfig = defaultAgentConfig;
  private status: "connecting" | "ready" | "disconnected" = "connecting";
  private api: TabbyApi;
  private completionCache: CompletionCache = new CompletionCache();

  constructor() {
    super();
    this.onConfigUpdated();
  }

  private onConfigUpdated() {
    this.api = new TabbyApi({ BASE: this.config.server.endpoint });
    this.ping();
  }

  private changeStatus(status: "connecting" | "ready" | "disconnected") {
    if (this.status != status) {
      this.status = status;
      const event: AgentEvent = { event: "statusChanged", status };
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

  private wrapApiPromise<T>(promise: CancelablePromise<T>): CancelablePromise<T> {
    return cancelable(
      promise
        .then((resolved: T) => {
          this.changeStatus("ready");
          return resolved;
        })
        .catch((err: ApiError) => {
          this.changeStatus("disconnected");
          throw err;
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
    return true;
  }

  public updateConfig(config: AgentConfig): boolean {
    if (!deepEqual(this.config, config)) {
      this.config = deepMerge(this.config, config);
      this.onConfigUpdated();
      const event: AgentEvent = { event: "configUpdated", config: this.config };
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
      return new CancelablePromise((resolve) => {
        resolve(this.completionCache.get(request));
      });
    }
    const prompt = this.createPrompt(request);
    if (isBlank(prompt)) {
      // Create a empty completion response
      return new CancelablePromise((resolve) => {
        resolve({
          id: "agent-" + uuid(),
          created: new Date().getTime(),
          choices: [],
        });
      });
    }
    const promise = this.wrapApiPromise(
      this.api.default.completionsV1CompletionsPost({
        prompt,
        language: request.language,
      })
    );
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
    return this.wrapApiPromise(this.api.default.eventsV1EventsPost(request));
  }
}
