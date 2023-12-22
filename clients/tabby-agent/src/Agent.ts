import type { components as ApiComponents } from "./types/tabbyApi";
import type { AgentConfig, PartialAgentConfig } from "./AgentConfig";
import type { DataStore } from "./dataStore";
import type { CompletionRequest, CompletionResponse } from "./CompletionContext";

export type { CompletionRequest, CompletionResponse };

export type ClientProperties = Partial<{
  user: Record<string, any>;
  session: Record<string, any>;
}>;

export type AgentInitOptions = Partial<{
  config: PartialAgentConfig;
  clientProperties: ClientProperties;
  dataStore: DataStore;
}>;

export type ServerHealthState = ApiComponents["schemas"]["HealthState"];

export type LogEventRequest = ApiComponents["schemas"]["LogEventRequest"] & {
  select_kind?: "line";
};

export type AbortSignalOption = { signal: AbortSignal };

export type SlowCompletionResponseTimeIssue = {
  name: "slowCompletionResponseTime";
  completionResponseStats: Record<string, number>;
};
export type HighCompletionTimeoutRateIssue = {
  name: "highCompletionTimeoutRate";
  completionResponseStats: Record<string, number>;
};
export type ConnectionFailedIssue = {
  name: "connectionFailed";
  message?: string;
};
export type AgentIssue = SlowCompletionResponseTimeIssue | HighCompletionTimeoutRateIssue | ConnectionFailedIssue;

/**
 * Represents the status of the agent.
 * @enum
 * @property {string} notInitialized - When the agent has not been initialized.
 * @property {string} ready - When the agent gets a valid response from the server.
 * @property {string} disconnected - When the agent fails to connect to the server.
 * @property {string} unauthorized - When the server requires authentication.
 * @property {string} finalized - When the agent is finalized.
 */
export type AgentStatus = "notInitialized" | "ready" | "disconnected" | "unauthorized" | "finalized";

export interface AgentFunction {
  /**
   * Initialize agent. Client should call this method before calling any other methods.
   * @param options
   */
  initialize(options?: AgentInitOptions): Promise<boolean>;

  /**
   * Finalize agent. Client should call this method before exiting.
   */
  finalize(): Promise<boolean>;

  /**
   * Update client properties.
   * Client properties are mostly used for logging and anonymous usage statistics.
   */
  updateClientProperties(type: keyof ClientProperties, key: string, value: any): Promise<boolean>;

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
   * @returns the current issues if any exists
   */
  getIssues(): AgentIssue["name"][];

  /**
   * Get the detail of an issue by index or name.
   * @param options if `index` is provided, `name` will be ignored
   * @returns the issue detail if exists, otherwise null
   */
  getIssueDetail<T extends AgentIssue>(options: { index?: number; name?: T["name"] }): T | null;

  /**
   * @returns server info returned from latest server health check, returns null if not available
   */
  getServerHealthState(): ServerHealthState | null;

  /**
   * @deprecated Tabby Cloud auth
   *
   * Request auth url for Tabby Cloud endpoint. Only return value when the `AgentStatus` is `unauthorized`.
   * Otherwise, return null. See also `AgentStatus`.
   * @returns the auth url for redirecting, and the code for next step `waitingForAuth`
   * @throws Error if agent is not initialized
   */
  requestAuthUrl(options?: AbortSignalOption): Promise<{ authUrl: string; code: string } | null>;

  /**
   * @deprecated Tabby Cloud auth
   *
   * Wait for auth token to be ready after redirecting user to auth url,
   * returns nothing, but `AgentStatus` will change to `ready` if resolved successfully.
   * @param code from `requestAuthUrl`
   * @throws Error if agent is not initialized
   */
  waitForAuthToken(code: string, options?: AbortSignalOption): Promise<void>;

  /**
   * Provide completions for the given request. This method is debounced, calling it before the previous
   * call is resolved will cancel the previous call. The debouncing interval is automatically calculated
   * or can be set in the config.
   * @param request
   * @returns
   * @throws Error if agent is not initialized
   */
  provideCompletions(request: CompletionRequest, options?: AbortSignalOption): Promise<CompletionResponse>;

  /**
   * @param event
   * @returns
   * @throws Error if agent is not initialized
   */
  postEvent(event: LogEventRequest, options?: AbortSignalOption): Promise<boolean>;
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
 * This event is emitted when the server requires authentication.
 */
export type AuthRequiredEvent = {
  event: "authRequired";
  server: AgentConfig["server"];
};
export type IssuesUpdatedEvent = {
  event: "issuesUpdated";
  issues: AgentIssue["name"][];
};

export type AgentEvent = StatusChangedEvent | ConfigUpdatedEvent | AuthRequiredEvent | IssuesUpdatedEvent;
export const agentEventNames: AgentEvent["event"][] = [
  "statusChanged",
  "configUpdated",
  "authRequired",
  "issuesUpdated",
];

export interface AgentEventEmitter {
  on<T extends AgentEvent>(eventName: T["event"], callback: (event: T) => void): this;
}

export type Agent = AgentFunction & AgentEventEmitter;
