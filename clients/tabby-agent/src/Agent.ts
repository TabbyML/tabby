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
  maxPrefixLines?: number;
  maxSuffixLines?: number;
};

export type CompletionResponse = ApiCompletionResponse;

export type LogEventRequest = ApiLogEventRequest;

export type AgentStatus = "notInitialized" | "ready" | "disconnected" | "unauthorized";

export interface AgentFunction {
  initialize(options: Partial<AgentInitOptions>): Promise<boolean>;
  updateConfig(config: Partial<AgentConfig>): Promise<boolean>;

  /**
   * @returns the current config
   *
   * Configuration precedence:
   * 1. Default config
   * 2. User config file `~/.tabby/agent/config.toml` (not available in browser)
   * 3. Agent `initialize` and `updateConfig` methods
   */
  getConfig(): AgentConfig;

  /**
   * @returns the current status
   */
  getStatus(): AgentStatus;

  /**
   * @returns the auth url for redirecting, and the code for next step `waitingForAuth`, only return value when
   *          `AgentStatus` is `unauthorized`, return null otherwise
   * @throws Error if agent is not initialized
   */
  requestAuthUrl(): CancelablePromise<{ authUrl: string; code: string } | null>;

  /**
   * Wait for auth token to be ready after redirecting user to auth url,
   * returns nothing, but `AgentStatus` will change to `ready` if resolved successfully
   * @param code from `requestAuthUrl`
   * @throws Error if agent is not initialized
   */
  waitForAuthToken(code: string): CancelablePromise<any>;

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
export type AuthRequiredEvent = {
  event: "authRequired";
  server: AgentConfig["server"];
};

export type AgentEvent = StatusChangedEvent | ConfigUpdatedEvent | AuthRequiredEvent;
export const agentEventNames: AgentEvent["event"][] = ["statusChanged", "configUpdated", "authRequired"];

export interface AgentEventEmitter {
  on<T extends AgentEvent>(eventName: T["event"], callback: (event: T) => void): this;
}

export type Agent = AgentFunction & AgentEventEmitter;
