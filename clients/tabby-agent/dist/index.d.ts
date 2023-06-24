import { EventEmitter } from 'events';

interface OnCancel {
    readonly isResolved: boolean;
    readonly isRejected: boolean;
    readonly isCancelled: boolean;
    (cancelHandler: () => void): void;
}
declare class CancelablePromise<T> implements Promise<T> {
    #private;
    constructor(executor: (resolve: (value: T | PromiseLike<T>) => void, reject: (reason?: any) => void, onCancel: OnCancel) => void);
    get [Symbol.toStringTag](): string;
    then<TResult1 = T, TResult2 = never>(onFulfilled?: ((value: T) => TResult1 | PromiseLike<TResult1>) | null, onRejected?: ((reason: any) => TResult2 | PromiseLike<TResult2>) | null): Promise<TResult1 | TResult2>;
    catch<TResult = never>(onRejected?: ((reason: any) => TResult | PromiseLike<TResult>) | null): Promise<T | TResult>;
    finally(onFinally?: (() => void) | null): Promise<T>;
    cancel(): void;
    get isCancelled(): boolean;
}

type Choice = {
    index: number;
    text: string;
};

type CompletionResponse$1 = {
    id: string;
    choices: Array<Choice>;
};

type LogEventRequest$1 = {
    /**
     * Event type, should be `view` or `select`.
     */
    type: string;
    completion_id: string;
    choice_index: number;
};

type AgentConfig = {
    server: {
        endpoint: string;
    };
    logs: {
        level: "debug" | "error" | "silent";
    };
    anonymousUsageTracking: {
        disable: boolean;
    };
};

type AgentInitOptions = {
    config: Partial<AgentConfig>;
    client: string;
};
type CompletionRequest = {
    filepath: string;
    language: string;
    text: string;
    position: number;
    maxPrefixLines: number;
    maxSuffixLines: number;
};
type CompletionResponse = CompletionResponse$1;
type LogEventRequest = LogEventRequest$1;
type AgentStatus = "notInitialized" | "ready" | "disconnected" | "unauthorized";
interface AgentFunction {
    initialize(options: Partial<AgentInitOptions>): Promise<boolean>;
    updateConfig(config: Partial<AgentConfig>): Promise<boolean>;
    getConfig(): AgentConfig;
    getStatus(): AgentStatus;
    /**
     * @returns the auth url for redirecting, and the code for next step `waitingForAuth`, only return value when
     *          `AgentStatus` is `unauthorized`, return null otherwise
     * @throws Error if agent is not initialized
     */
    requestAuthUrl(): CancelablePromise<{
        authUrl: string;
        code: string;
    } | null>;
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
type StatusChangedEvent = {
    event: "statusChanged";
    status: AgentStatus;
};
type ConfigUpdatedEvent = {
    event: "configUpdated";
    config: AgentConfig;
};
type AuthRequiredEvent = {
    event: "authRequired";
    server: AgentConfig["server"];
};
type AgentEvent = StatusChangedEvent | ConfigUpdatedEvent | AuthRequiredEvent;
declare const agentEventNames: AgentEvent["event"][];
interface AgentEventEmitter {
    on<T extends AgentEvent>(eventName: T["event"], callback: (event: T) => void): this;
}
type Agent = AgentFunction & AgentEventEmitter;

type StoredData = {
    anonymousId: string;
    auth: {
        [endpoint: string]: {
            jwt: string;
        };
    };
};
interface DataStore {
    data: Partial<StoredData>;
    load(): PromiseLike<void>;
    save(): PromiseLike<void>;
}

/**
 * Different from AgentInitOptions or AgentConfig, this may contain non-serializable objects,
 * so it is not suitable for cli, but only used when imported as module by other js project.
 */
type TabbyAgentOptions = {
    dataStore: DataStore;
};
declare class TabbyAgent extends EventEmitter implements Agent {
    private readonly logger;
    private anonymousUsageLogger;
    private config;
    private status;
    private api;
    private auth;
    private dataStore;
    private completionCache;
    static readonly tryConnectInterval: number;
    private tryingConnectTimer;
    private constructor();
    static create(options?: Partial<TabbyAgentOptions>): Promise<TabbyAgent>;
    private applyConfig;
    private setupApi;
    private changeStatus;
    private callApi;
    private healthCheck;
    private createSegments;
    initialize(options: Partial<AgentInitOptions>): Promise<boolean>;
    updateConfig(config: Partial<AgentConfig>): Promise<boolean>;
    getConfig(): AgentConfig;
    getStatus(): AgentStatus;
    requestAuthUrl(): CancelablePromise<{
        authUrl: string;
        code: string;
    } | null>;
    waitForAuthToken(code: string): CancelablePromise<any>;
    getCompletions(request: CompletionRequest): CancelablePromise<CompletionResponse>;
    postEvent(request: LogEventRequest): CancelablePromise<boolean>;
}

export { Agent, AgentConfig, AgentEvent, AgentFunction, AgentStatus, CancelablePromise, CompletionRequest, CompletionResponse, ConfigUpdatedEvent, DataStore, LogEventRequest, StatusChangedEvent, TabbyAgent, TabbyAgentOptions, agentEventNames };
