import { workspace } from "vscode";
import axios from "axios";

export enum EventType {
  InlineCompletionDisplayed,
  InlineCompletionAccepted,
}

export interface Event {
  type: EventType,
  id?: string,
  index?: number,
}

export class EventHandler {
  private tabbyServerUrl: string = "";

  constructor() {
    this.updateConfiguration();
    workspace.onDidChangeConfiguration((event) => {
      if (event.affectsConfiguration("tabby")) {
        this.updateConfiguration();
      }
    });
  }

  handle(event: Event) {
    console.debug("Event: ", event);
    switch (event.type) {
      case EventType.InlineCompletionDisplayed:
        axios.post(`${this.tabbyServerUrl}/v1/completions/${event.id}/choices/${event.index}/view`);
        break;
      case EventType.InlineCompletionAccepted:
        axios.post(`${this.tabbyServerUrl}/v1/completions/${event.id}/choices/${event.index}/select`);
        break;
    }
  }

  private updateConfiguration() {
    const configuration = workspace.getConfiguration("tabby");
    this.tabbyServerUrl = configuration.get("serverUrl", "http://127.0.0.1:5000");
  }
}