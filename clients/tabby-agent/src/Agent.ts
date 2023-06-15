import {
  CancelablePromise,
  LogEventRequest as ApiLogEventRequest,
  CompletionResponse as ApiCompletionResponse,
} from "./generated";

import { AgentConfig } from "./AgentConfig";

export type AgentInitOptions = {
  config: Partial<AgentConfig>;
  client: string;
};

export type CompletionRequest = {
  filepath: string;
  language: string;
  text: string;
  position: number;
};

export type CompletionResponse = ApiCompletionResponse;

export type LogEventRequest = ApiLogEventRequest;

export type AgentStatus = "notInitialized" | "ready" | "disconnected" | "unauthorized";

export interface AgentFunction {
  initialize(options: Partial<AgentInitOptions>): Promise<boolean>;
  updateConfig(config: Partial<AgentConfig>):  Promise<boolean>;
  getConfig(): AgentConfig;
  getStatus(): AgentStatus;

  /**
   * @returns string auth url if AgentStatus is `unauthorized`, null otherwise
   * @throws Error if agent is not initialized
   */
  startAuth(): CancelablePromise<string | null>;

  /**
   * @param request 
   * @returns
   * @throws Error if agent is not initialized
   */
  getCompletions(request: CompletionRequest): CancelablePromise<CompletionResponse>;

  /**
   * @param event 
   * @returns
   * @throws Error if agent is not initialized
   */
  postEvent(event: LogEventRequest): CancelablePromise<boolean>;
}

export type StatusChangedEvent = {
  event: "statusChanged";
  status: AgentStatus;
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
