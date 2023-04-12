type HttpRequestConstructor = new (config: OpenAPIConfig) => BaseHttpRequest;
export declare class Tabby {
    readonly default: DefaultService;
    readonly request: BaseHttpRequest;
    constructor(config?: Partial<OpenAPIConfig>, HttpRequest?: HttpRequestConstructor);
}

export declare class ApiError extends Error {
    readonly url: string;
    readonly status: number;
    readonly statusText: string;
    readonly body: any;
    readonly request: ApiRequestOptions;
    constructor(request: ApiRequestOptions, response: ApiResult, message: string);
}

export type ApiRequestOptions = {
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

export type ApiResult = {
    readonly url: string;
    readonly ok: boolean;
    readonly status: number;
    readonly statusText: string;
    readonly body: any;
};

export declare class AxiosHttpRequest extends BaseHttpRequest {
    constructor(config: OpenAPIConfig);
    /**
     * Request method
     * @param options The request options from the service
     * @returns CancelablePromise<T>
     * @throws ApiError
     */
    request<T>(options: ApiRequestOptions): CancelablePromise<T>;
}

export declare abstract class BaseHttpRequest {
    readonly config: OpenAPIConfig;
    constructor(config: OpenAPIConfig);
    abstract request<T>(options: ApiRequestOptions): CancelablePromise<T>;
}

export declare class CancelError extends Error {
    constructor(message: string);
    get isCancelled(): boolean;
}
export interface OnCancel {
    readonly isResolved: boolean;
    readonly isRejected: boolean;
    readonly isCancelled: boolean;
    (cancelHandler: () => void): void;
}
export declare class CancelablePromise<T> implements Promise<T> {
    #private;
    constructor(executor: (resolve: (value: T | PromiseLike<T>) => void, reject: (reason?: any) => void, onCancel: OnCancel) => void);
    get [Symbol.toStringTag](): string;
    then<TResult1 = T, TResult2 = never>(onFulfilled?: ((value: T) => TResult1 | PromiseLike<TResult1>) | null, onRejected?: ((reason: any) => TResult2 | PromiseLike<TResult2>) | null): Promise<TResult1 | TResult2>;
    catch<TResult = never>(onRejected?: ((reason: any) => TResult | PromiseLike<TResult>) | null): Promise<T | TResult>;
    finally(onFinally?: (() => void) | null): Promise<T>;
    cancel(): void;
    get isCancelled(): boolean;
}

type Resolver<T> = (options: ApiRequestOptions) => Promise<T>;
type Headers = Record<string, string>;
export type OpenAPIConfig = {
    BASE: string;
    VERSION: string;
    WITH_CREDENTIALS: boolean;
    CREDENTIALS: 'include' | 'omit' | 'same-origin';
    TOKEN?: string | Resolver<string>;
    USERNAME?: string | Resolver<string>;
    PASSWORD?: string | Resolver<string>;
    HEADERS?: Headers | Resolver<Headers>;
    ENCODE_PATH?: (path: string) => string;
};
export declare const OpenAPI: OpenAPIConfig;

/**
 * Request method
 * @param config The OpenAPI configuration object
 * @param options The request options from the service
 * @returns CancelablePromise<T>
 * @throws ApiError
 */
export declare const request: <T>(config: OpenAPIConfig, options: ApiRequestOptions) => CancelablePromise<T>;

export type Choice = {
    index: number;
    text: string;
};

export type ChoiceEvent = {
    type: EventType;
    completion_id: string;
    choice_index: number;
};

export type CompletionEvent = {
    type: EventType;
    id: string;
    language: Language;
    prompt: string;
    created: number;
    choices: Array<Choice>;
};

export type CompletionRequest = {
    /**
     * Language for completion request
     */
    language?: Language;
    /**
     * The context to generate completions for, encoded as a string.
     */
    prompt: string;
};

export type CompletionResponse = {
    id: string;
    created: number;
    choices: Array<Choice>;
};

/**
 * An enumeration.
 */
export declare enum EventType {
    COMPLETION = "completion",
    VIEW = "view",
    SELECT = "select"
}

export type HTTPValidationError = {
    detail?: Array<ValidationError>;
};

/**
 * An enumeration.
 */
export declare enum Language {
    UNKNOWN = "unknown",
    PYTHON = "python",
    JAVASCRIPT = "javascript",
    TYPESCRIPT = "typescript"
}

export type ValidationError = {
    loc: Array<(string | number)>;
    msg: string;
    type: string;
};

export declare class DefaultService {
    readonly httpRequest: BaseHttpRequest;
    constructor(httpRequest: BaseHttpRequest);
    /**
     * Completions
     * @param requestBody
     * @returns CompletionResponse Successful Response
     * @throws ApiError
     */
    completionsV1CompletionsPost(requestBody: CompletionRequest): CancelablePromise<CompletionResponse>;
    /**
     * Events
     * @param requestBody
     * @returns any Successful Response
     * @throws ApiError
     */
    eventsV1EventsPost(requestBody: (ChoiceEvent | CompletionEvent)): CancelablePromise<any>;
}
