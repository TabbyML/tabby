import { workspace } from "vscode";
import axios from "axios";
import { sleep } from "./utils";
import { EventEmitter } from "node:events";
import { strict as assert } from "node:assert";

const logAxios = false;
if (logAxios) {
  axios.interceptors.request.use((request) => {
    console.debug("Starting Request: ", request);
    return request;
  });
  axios.interceptors.response.use((response) => {
    console.debug("Response: ", response);
    return response;
  });
}

export interface TabbyCompletion {
  id?: string;
  created?: number;
  choices?: Array<{
    index: number;
    text: string;
  }>;
}

export enum EventType {
  InlineCompletionDisplayed = "view",
  InlineCompletionAccepted = "select",
}

export interface Event {
  type: EventType;
  id?: string;
  index?: number;
}

export class TabbyClient extends EventEmitter {
  private static instance: TabbyClient;
  static getInstance(): TabbyClient {
    if (!TabbyClient.instance) {
      TabbyClient.instance = new TabbyClient();
    }
    return TabbyClient.instance;
  }

  private tabbyServerUrl: string = "";
  status: "connecting" | "ready" | "disconnected" = "connecting";

  constructor() {
    super();

    this.updateConfiguration();
    workspace.onDidChangeConfiguration((event) => {
      if (event.affectsConfiguration("tabby")) {
        this.updateConfiguration();
      }
    });
  }

  private updateConfiguration() {
    const configuration = workspace.getConfiguration("tabby");
    this.tabbyServerUrl = configuration.get("serverUrl", "http://127.0.0.1:5000");
    this.ping();
  }

  private changeStatus(status: "connecting" | "ready" | "disconnected") {
    if (this.status != status) {
      this.status = status;
      this.emit("statusChanged", status);
    }
  }

  private async ping(tries: number = 0) {
    try {
      const response = await axios.get(`${this.tabbyServerUrl}/`);
      assert(response.status == 200);
      this.changeStatus("ready");
    } catch (e) {
      if (tries > 5) {
        this.changeStatus("disconnected");
        return;
      }
      this.changeStatus("connecting");
      const pingRetryDelay = 1000;
      await sleep(pingRetryDelay);
      this.ping(tries + 1);
    }
  }

  public async getCompletion(prompt: string): Promise<TabbyCompletion | null> {
    if (this.status == "disconnected") {
      this.ping();
    }
    try {
      const response = await axios.post<TabbyCompletion>(`${this.tabbyServerUrl}/v1/completions`, {
        prompt,
      });
      assert(response.status == 200);
      return response.data;
    } catch (e) {
      this.ping();
      return null;
    }
  }

  public async postEvent(event: Event) {
    if (this.status == "disconnected") {
      this.ping();
    }
    try {
      const response = await axios.post(`${this.tabbyServerUrl}/v1/events`, {
        type: event.type,
        completion_id: event.id,
        choice_index: event.index
      });
      assert(response.status == 200);
    } catch (e) {
      this.ping();
    }
  }
}
