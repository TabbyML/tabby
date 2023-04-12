import { workspace } from "vscode";
import axios from "axios";
import { sleep } from "./utils";
import { EventEmitter } from "node:events";
import { strict as assert } from "node:assert";
import { Tabby as TabbyClient } from "tabby-client";

export class Tabby extends EventEmitter {
  private static instance: Tabby;
  static getInstance(): Tabby {
    if (!Tabby.instance) {
      Tabby.instance = new Tabby();
    }
    return Tabby.instance;
  }

  private serverUrl: string = "";
  status: "connecting" | "ready" | "disconnected" = "connecting";
  api: TabbyClient;

  constructor() {
    super();

    this.updateConfiguration();
    this.api = new TabbyClient({ BASE: this.serverUrl });

    workspace.onDidChangeConfiguration((event) => {
      if (event.affectsConfiguration("tabby")) {
        this.updateConfiguration();
        this.api = new TabbyClient({ BASE: this.serverUrl });
      }
    });
  }

  private updateConfiguration() {
    const configuration = workspace.getConfiguration("tabby");
    this.serverUrl = configuration.get("serverUrl", "http://127.0.0.1:5000");
    this.serverUrl = this.serverUrl.replace(/\/$/, ''); // Remove trailing slash
    this.ping();
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

  public async healthCheck(force = false): Promise<boolean> {
    if (force || (this.status == "disconnected")) {
      return this.ping();
    }
    return true;
  }
}
