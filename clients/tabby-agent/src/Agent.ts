import axios from "axios";
import { sleep } from "./utils";
import { EventEmitter } from "node:events";
import { strict as assert } from "node:assert";
import {
  TabbyApi,
  CancelablePromise,
  CancelError,
  ApiError,
  CompletionRequest,
  CompletionResponse,
  ChoiceEvent,
  CompletionEvent,
} from "./generated";

export class Agent extends EventEmitter {
  private serverUrl: string = "http://127.0.0.1:5000";
  private status: "connecting" | "ready" | "disconnected" = "connecting";
  private api: TabbyApi;

  constructor() {
    super();
    this.ping();
    this.api = new TabbyApi({ BASE: this.serverUrl });
  }

  private changeStatus(status: "connecting" | "ready" | "disconnected") {
    if (this.status != status) {
      this.status = status;
      this.emit("statusChanged", status);
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
    return new CancelablePromise((resolve, reject, onCancel) => {
      promise
        .then((resp: T) => {
          this.changeStatus("ready");
          resolve(resp);
        })
        .catch((err: CancelError) => {
          reject(err);
        })
        .catch((err: ApiError) => {
          this.changeStatus("disconnected");
          reject(err);
        })
        .catch((err: Error) => {
          reject(err);
        });
      onCancel(() => {
        promise.cancel();
      });
    });
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

  public getCompletions(req: CompletionRequest): CancelablePromise<CompletionResponse> {
    const promise = this.api.default.completionsV1CompletionsPost(req);
    return this.wrapApiPromise(promise);
  }

  public postEvent(req: ChoiceEvent | CompletionEvent): CancelablePromise<any> {
    const promise = this.api.default.eventsV1EventsPost(req);
    return this.wrapApiPromise(promise);
  }
}
