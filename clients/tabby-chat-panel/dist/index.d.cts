import { ThreadOptions } from '@quilted/threads';

type LineRange = {
    start: number;
    end: number;
};
type FileContext = {
    kind: 'file';
    range: LineRange;
    filename: string;
    link: string;
};
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
type CreateThreadFn = ((target: any, opts: ThreadOptions<Api>) => Record<string, any>) | ((opts: ThreadOptions<Api>) => Record<string, any>);
declare function createClient(createFn: CreateThreadFn, target: any): Api;
declare function createServer(createFn: CreateThreadFn, api: Api, target?: any): Record<string, any>;

export { type Api, type ChatMessage, type Context, type FetcherOptions, type FileContext, type InitRequest, type LineRange, createClient, createServer };
