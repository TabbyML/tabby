import axios from "axios";
import { EventEmitter } from "events";
import { strict as assert } from "assert";
import { CompletionCache } from "./CompletionCache";
import { sleep, cancelable } from "./utils";
import { Agent, AgentEvent } from "./types";
import {
  TabbyApi,
  CancelablePromise,
  ApiError,
  CompletionRequest,
  CompletionResponse,
  ChoiceEvent,
  CompletionEvent,
} from "./generated";

export class TabbyAgent extends EventEmitter implements Agent {
  private serverUrl: string = "http://127.0.0.1:5000";
  private status: "connecting" | "ready" | "disconnected" = "connecting";
  private api: TabbyApi;
  private completionCache: CompletionCache;

  constructor() {
    super();
    this.ping();
    this.api = new TabbyApi({ BASE: this.serverUrl });
    this.completionCache = new CompletionCache();
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
      const response = await axios.get(`${this.serverUrl}/`);
      assert(response.status == 200);
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

  public setServerUrl(serverUrl: string): string {
    this.serverUrl = serverUrl.replace(/\/$/, ""); // Remove trailing slash
    this.ping();
    this.api = new TabbyApi({ BASE: this.serverUrl });
    return this.serverUrl;
  }

  public getServerUrl(): string {
    return this.serverUrl;
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
    const promise = this.wrapApiPromise(this.api.default.completionsV1CompletionsPost(request));
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

  public postEvent(request: ChoiceEvent | CompletionEvent): CancelablePromise<any> {
    return this.wrapApiPromise(this.api.default.eventsV1EventsPost(request));
  }
}
