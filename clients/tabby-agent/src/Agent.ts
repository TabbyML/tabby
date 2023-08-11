import {
  CancelablePromise,
  LogEventRequest as ApiLogEventRequest,
  CompletionResponse as ApiCompletionResponse,
} from "./generated";

import { AgentConfig, PartialAgentConfig } from "./AgentConfig";

export type AgentInitOptions = Partial<{
  config: PartialAgentConfig;
  client: string;
}>;

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

/**
 * `notInitialized`: When the agent is not initialized.
 * `ready`: When the agent get a valid response from the server, and is ready to use.
 * `disconnected`: When the agent failed to connect to the server.
 * `unauthorized`: When the server is set to a Tabby Cloud endpoint that requires auth,
 *   and no `Authorization` request header is provided in the agent config,
 *   and user has not completed the auth flow or the auth token is expired.
 *   See also `requestAuthUrl` and `waitForAuthToken`.
 */
export type AgentStatus = "notInitialized" | "ready" | "disconnected" | "unauthorized";

export interface AgentFunction {
  /**
   * Initialize agent. Client should call this method before calling any other methods.
   * @param options
   */
  initialize(options: AgentInitOptions): Promise<boolean>;

  /**
   * The agent configuration has the following levels, will be deep merged in the order:
   * 1. Default config
   * 2. User config file `~/.tabby/agent/config.toml` (not available in browser)
   * 3. Agent `initialize` and `updateConfig` methods
   *
   * This method will update the 3rd level config.
   * @param key the key of the config to update, can be nested with dot, e.g. `server.endpoint`
   * @param value the value to set
   */
  updateConfig(key: string, value: any): Promise<boolean>;

  /**
   * Clear the 3rd level config.
   * @param key the key of the config to clear, can be nested with dot, e.g. `server.endpoint`
   */
  clearConfig(key: string): Promise<boolean>;

  /**
   * @returns the current config
   */
  getConfig(): AgentConfig;

  /**
   * @returns the current status
   */
  getStatus(): AgentStatus;

  /**
   * Request auth url for Tabby Cloud endpoint. Only return value when the `AgentStatus` is `unauthorized`.
   * Otherwise, return null. See also `AgentStatus`.
   * @returns the auth url for redirecting, and the code for next step `waitingForAuth`
   * @throws Error if agent is not initialized
   */
  requestAuthUrl(): CancelablePromise<{ authUrl: string; code: string } | null>;

  /**
   * Wait for auth token to be ready after redirecting user to auth url,
   * returns nothing, but `AgentStatus` will change to `ready` if resolved successfully.
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
/**
 * This event is emitted when the server is set to a Tabby Cloud endpoint that requires auth,
 * and no `Authorization` request header is provided in the agent config.
 */
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
