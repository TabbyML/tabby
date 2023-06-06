import { EventEmitter } from 'events';

type ApiRequestOptions = {
    readonly method: 'GET' | 'PUT' | 'POST' | 'DELETE' | 'OPTIONS' | 'HEAD' | 'PATCH';
    readonly url: string;
    readonly path?: Record<string, any>;
    readonly cookies?: Record<string, any>;
    readonly headers?: Record<string, any>;
    readonly query?: Record<string, any>;
    readonly formData?: Record<string, any>;
    readonly body?: any;
    readonly mediaType?: string;
    readonly responseHeader?: string;
    readonly errors?: Record<number, string>;
};

declare class CancelError extends Error {
    constructor(message: string);
    get isCancelled(): boolean;
}
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

/**
 * An enumeration.
 */
declare enum EventType {
    COMPLETION = "completion",
    VIEW = "view",
    SELECT = "select"
}

type ChoiceEvent = {
    type: EventType;
    completion_id: string;
    choice_index: number;
};

type Choice = {
    index: number;
    text: string;
};

type CompletionEvent = {
    type: EventType;
    id: string;
    language: string;
    prompt: string;
    created: number;
    choices: Array<Choice>;
};

type CompletionResponse$1 = {
    id: string;
    created: number;
    choices: Array<Choice>;
};

type ApiResult = {
    readonly url: string;
    readonly ok: boolean;
    readonly status: number;
    readonly statusText: string;
    readonly body: any;
};

declare class ApiError extends Error {
    readonly url: string;
    readonly status: number;
    readonly statusText: string;
    readonly body: any;
    readonly request: ApiRequestOptions;
    constructor(request: ApiRequestOptions, response: ApiResult, message: string);
}

type ValidationError = {
    loc: Array<(string | number)>;
    msg: string;
    type: string;
};

type HTTPValidationError = {
    detail?: Array<ValidationError>;
};

type AgentConfig = {
    server?: {
        endpoint?: string;
    };
    logs?: {
        level?: "debug" | "error" | "silent";
    };
    anonymousUsageTracking?: {
        disable?: boolean;
    };
};

type AgentInitOptions = {
    config?: AgentConfig;
    client?: string;
};
type CompletionRequest = {
    filepath: string;
    language: string;
    text: string;
    position: number;
};
type CompletionResponse = CompletionResponse$1;
interface AgentFunction {
    initialize(options?: AgentInitOptions): boolean;
    updateConfig(config: AgentConfig): boolean;
    getConfig(): AgentConfig;
    getStatus(): "connecting" | "ready" | "disconnected";
    getCompletions(request: CompletionRequest): CancelablePromise<CompletionResponse>;
    postEvent(event: ChoiceEvent | CompletionEvent): CancelablePromise<boolean>;
}
type StatusChangedEvent = {
    event: "statusChanged";
    status: "connecting" | "ready" | "disconnected";
};
type ConfigUpdatedEvent = {
    event: "configUpdated";
    config: AgentConfig;
};
type AgentEvent = StatusChangedEvent | ConfigUpdatedEvent;
declare const agentEventNames: AgentEvent["event"][];
interface AgentEventEmitter {
    on<T extends AgentEvent>(eventName: T["event"], callback: (event: T) => void): this;
}
type Agent = AgentFunction & AgentEventEmitter;

declare class TabbyAgent extends EventEmitter implements Agent {
    private readonly logger;
    private config;
    private status;
    private api;
    private completionCache;
    constructor();
    private onConfigUpdated;
    private changeStatus;
    private ping;
    private callApi;
    private createPrompt;
    initialize(params: AgentInitOptions): boolean;
    updateConfig(config: AgentConfig): boolean;
    getConfig(): AgentConfig;
    getStatus(): "connecting" | "ready" | "disconnected";
    getCompletions(request: CompletionRequest): CancelablePromise<CompletionResponse>;
    postEvent(request: ChoiceEvent | CompletionEvent): CancelablePromise<boolean>;
}

export { Agent, AgentConfig, AgentEvent, AgentFunction, ApiError, CancelError, CancelablePromise, Choice, ChoiceEvent, CompletionEvent, CompletionRequest, CompletionResponse, EventType, HTTPValidationError, StatusChangedEvent, TabbyAgent, ValidationError, agentEventNames };
