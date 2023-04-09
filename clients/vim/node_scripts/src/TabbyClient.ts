import axios from "axios";
import { sleep } from "./utils";
import { EventEmitter } from "node:events";
import { strict as assert } from "node:assert";

export interface TabbyCompletionRequest {
  prompt: string;
  language?: string;
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

  private tabbyServerUrl: string = "http://127.0.0.1:5000";
  private status: "connecting" | "ready" | "disconnected" = "connecting";

  constructor() {
    super();
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
      const response = await axios.get(`${this.tabbyServerUrl}/`);
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

  public async setServerUrl(serverUrl: string): Promise<boolean> {
    this.tabbyServerUrl = serverUrl;
    this.ping();
    return true;
  }

  public async getServerUrl(): Promise<string> {
    return this.tabbyServerUrl;
  }

  public async getCompletion(
    request: TabbyCompletionRequest
  ): Promise<TabbyCompletion | null> {
    if (this.status == "disconnected") {
      this.ping();
    }
    try {
      const response = await axios.post<TabbyCompletion>(
        `${this.tabbyServerUrl}/v1/completions`,
        request
      );
      assert(response.status == 200);
      return response.data;
    } catch (e) {
      this.ping();
      return null;
    }
  }

  public async postEvent(event: Event): Promise<boolean> {
    if (this.status == "disconnected") {
      this.ping();
    }
    try {
      const response = await axios.post(`${this.tabbyServerUrl}/v1/events`, {
        type: event.type,
        completion_id: event.id,
        choice_index: event.index,
      });
      assert(response.status == 200);
      return true;
    } catch (e) {
      this.ping();
      return false;
    }
  }
}
