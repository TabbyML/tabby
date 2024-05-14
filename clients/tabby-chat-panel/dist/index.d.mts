import * as _quilted_threads from '@quilted/threads';

interface LineRange {
    start: number;
    end?: number;
}
interface FileContext {
    kind: 'file';
    range: LineRange;
    filepath: string;
}
type Context = FileContext;
interface FetcherOptions {
    authorization: string;
}
interface InitRequest {
    fetcherOptions: FetcherOptions;
}
interface Api {
    init: (request: InitRequest) => void;
    sendMessage: (message: ChatMessage) => void;
}
interface ChatMessage {
    message: string;
    selectContext?: Context;
    relevantContext?: Array<Context>;
}
declare function createClient(target: HTMLIFrameElement): _quilted_threads.Thread<Record<string, never>>;
declare function createServer(api: Api): _quilted_threads.Thread<Record<string, never>>;

export { type Api, type ChatMessage, type Context, type FetcherOptions, type FileContext, type InitRequest, type LineRange, createClient, createServer };
