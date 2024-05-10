import { MessageEndpoint, createEndpoint, fromIframe, fromInsideIframe } from '@remote-ui/rpc'

export interface LineRange {
  start: number
  end: number
}

export interface FileContext {
  kind: 'file'
  range: LineRange
  filename: string
  link: string
}

export type Context = FileContext

export interface FetcherOptions {
  authorization: string
}

export interface InitRequest {
  message?: string
  selectContext?: Context
  relevantContext?: Array<Context>
  fetcherOptions?: FetcherOptions
}

export interface Api {
  init: (request: InitRequest) => void
}

export function createClient(endpoint: MessageEndpoint) {
  return createEndpoint<Api>(endpoint)
}

export function createServer(endpoint: MessageEndpoint, api: Api) {
    const server = createEndpoint(endpoint)
    server.expose({
      init: api.init,
    })
    return server;
}