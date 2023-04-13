import axios from "axios";
import { sleep } from "./utils";
import { EventEmitter } from "node:events";
import { strict as assert } from "node:assert";
import { Tabby as TabbyApi } from "./generated";

export class TabbyClient extends EventEmitter {
  private static instance: TabbyClient;
  static getInstance(): TabbyClient {
    if (!TabbyClient.instance) {
      TabbyClient.instance = new TabbyClient();
    }
    return TabbyClient.instance;
  }

  private serverUrl: string = "http://127.0.0.1:5000";
  private status: "connecting" | "ready" | "disconnected" = "connecting";
  api: TabbyApi;

  constructor() {
    super();
    this.ping();
    this.api = new TabbyApi({ BASE: this.serverUrl });
  }

  public changeStatus(status: "connecting" | "ready" | "disconnected") {
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

  public setServerUrl(serverUrl: string): boolean {
    this.serverUrl = serverUrl.replace(/\/$/, ''); // Remove trailing slash
    this.ping();
    this.api = new TabbyApi({ BASE: this.serverUrl });
    return true;
  }

  public getServerUrl(): string {
    return this.serverUrl;
  }
}
