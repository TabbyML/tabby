interface LineRange {
    start: number;
    end: number;
}
interface FileContext {
    kind: 'file';
    range: LineRange;
    filepath: string;
    content?: string;
}
type Context = FileContext;
interface FetcherOptions {
    authorization: string;
}
interface InitRequest {
    fetcherOptions: FetcherOptions;
}
interface ServerApi {
    init: (request: InitRequest) => void;
    sendMessage: (message: ChatMessage) => void;
}
interface ClientApi {
    navigate: (context: Context) => void;
}
interface ChatMessage {
    message: string;
    selectContext?: Context;
    relevantContext?: Array<Context>;
}
declare function createClient(target: HTMLIFrameElement, api: ClientApi): ServerApi;
declare function createServer(api: ServerApi): ClientApi;

export { type ChatMessage, type ClientApi, type Context, type FetcherOptions, type FileContext, type InitRequest, type LineRange, type ServerApi, createClient, createServer };
