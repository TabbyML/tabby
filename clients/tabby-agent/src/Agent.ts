import {
  CancelablePromise,
  LogEventRequest as ApiLogEventRequest,
  CompletionResponse as ApiCompletionResponse,
  HealthState,
} from "./generated";

import { AgentConfig, PartialAgentConfig } from "./AgentConfig";

export type AgentInitOptions = Partial<{
  config: PartialAgentConfig;
  client: string;
}>;

export type ServerHealthState = HealthState;

export type CompletionRequest = {
  filepath: string;
  language: string;
  text: string;
  position: number;
  manually?: boolean;
};

export type CompletionResponse = ApiCompletionResponse;

export type LogEventRequest = ApiLogEventRequest;

export type SlowCompletionResponseTimeIssue = {
  name: "slowCompletionResponseTime";
  completionResponseStats: Record<string, number>;
};
export type HighCompletionTimeoutRateIssue = {
  name: "highCompletionTimeoutRate";
  completionResponseStats: Record<string, number>;
};
export type AgentIssue = SlowCompletionResponseTimeIssue | HighCompletionTimeoutRateIssue;

/**
 * Represents the status of the agent.
 * @enum
 * @property {string} notInitialized - When the agent is not initialized.
 * @property {string} ready - When the agent gets a valid response from the server.
 * @property {string} disconnected - When the agent fails to connect to the server.
 * @property {string} unauthorized - When the server is set to a Tabby Cloud endpoint that requires auth,
 *   and no `Authorization` request header is provided in the agent config,
 *   and the user has not completed the auth flow or the auth token is expired.
 *   See also `requestAuthUrl` and `waitForAuthToken`.
 * @property {string} issuesExist - When the agent gets a valid response from the server, but still
 *   has some non-blocking issues, e.g. the average completion response time is too slow,
 *   or the timeout rate is too high.
 */
export type AgentStatus = "notInitialized" | "ready" | "disconnected" | "unauthorized" | "issuesExist";

export interface AgentFunction {
  /**
   * Initialize agent. Client should call this method before calling any other methods.
   * @param options
   */
  initialize(options: AgentInitOptions): Promise<boolean>;

  /**
   * The agent configuration has the following levels, will be deep merged in the order:
   * 1. Default config
   * 2. User config file `~/.tabby-client/agent/config.toml` (not available in browser)
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
   * @returns the current issues if AgentStatus is `issuesExist`, otherwise returns empty array
   */
  getIssues(): AgentIssue[];

  /**
   * @returns server info returned from latest server health check, returns null if not available
   */
  getServerHealthState(): ServerHealthState | null;

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
   * Provide completions for the given request. This method is debounced, calling it before the previous
   * call is resolved will cancel the previous call. The debouncing interval is automatically calculated
   * or can be set in the config.
   * @param request
   * @returns
   * @throws Error if agent is not initialized
   */
  provideCompletions(request: CompletionRequest): CancelablePromise<CompletionResponse>;

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
export type NewIssueEvent = {
  event: "newIssue";
  issue: AgentIssue;
};

export type AgentEvent = StatusChangedEvent | ConfigUpdatedEvent | AuthRequiredEvent | NewIssueEvent;
export const agentEventNames: AgentEvent["event"][] = ["statusChanged", "configUpdated", "authRequired", "newIssue"];

export interface AgentEventEmitter {
  on<T extends AgentEvent>(eventName: T["event"], callback: (event: T) => void): this;
}

export type Agent = AgentFunction & AgentEventEmitter;
