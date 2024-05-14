import { createThreadFromIframe, createThreadFromInsideIframe } from '@quilted/threads'

export interface LineRange {
  start: number
  end?: number
}

export interface FileContext {
  kind: 'file'
  range: LineRange
  filepath: string
}

export type Context = FileContext

export interface FetcherOptions {
  authorization: string
}

export interface InitRequest {
  fetcherOptions: FetcherOptions
}

export interface Api {
  init: (request: InitRequest) => void
  sendMessage: (message: ChatMessage) => void
}

export interface ChatMessage {
  message: string
  selectContext?: Context
  relevantContext?: Array<Context>
}

export function createClient(target: HTMLIFrameElement) {
  return createThreadFromIframe(target)
}

export function createServer(api: Api) {
  return createThreadFromInsideIframe({
    expose: {
      init: api.init,
      sendMessage: api.sendMessage,
    },
  })
}
