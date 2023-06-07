import {
  CancelablePromise,
  LogEventRequest as ApiLogEventRequest,
  CompletionResponse as ApiCompletionResponse,
} from "./generated";

import { AgentConfig } from "./AgentConfig";

export type AgentInitOptions = {
  config?: AgentConfig;
  client?: string;
};

export type CompletionRequest = {
  filepath: string;
  language: string;
  text: string;
  position: number;
};

export type CompletionResponse = ApiCompletionResponse;

export type LogEventRequest = ApiLogEventRequest;

export interface AgentFunction {
  initialize(options?: AgentInitOptions): boolean;
  updateConfig(config: AgentConfig): boolean;
  getConfig(): AgentConfig;
  getStatus(): "connecting" | "ready" | "disconnected";
  getCompletions(request: CompletionRequest): CancelablePromise<CompletionResponse>;
  postEvent(event: LogEventRequest): CancelablePromise<boolean>;
}

export type StatusChangedEvent = {
  event: "statusChanged";
  status: "connecting" | "ready" | "disconnected";
};
export type ConfigUpdatedEvent = {
  event: "configUpdated";
  config: AgentConfig;
};

export type AgentEvent = StatusChangedEvent | ConfigUpdatedEvent;
export const agentEventNames: AgentEvent["event"][] = ["statusChanged", "configUpdated"];

export interface AgentEventEmitter {
  on<T extends AgentEvent>(eventName: T["event"], callback: (event: T) => void): this;
}

export type Agent = AgentFunction & AgentEventEmitter;
