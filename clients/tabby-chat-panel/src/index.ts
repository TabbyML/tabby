import type { Thread, ThreadOptions } from '@quilted/threads'

export interface LineRange {
  start: number
  end?: number
}

export interface FileContext {
  kind: 'file'
  range?: LineRange
  language?: string
  path: string
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

type CreateThreadFn =
  ((target: any, opts: ThreadOptions<Api>) => Record<string, any>) |
  ((opts: ThreadOptions<Api>) => Record<string, any>)

export function createClient(createFn: CreateThreadFn, target: any) {
  return createFn(target, {
    callable: ['init', 'sendMessage'],
  }) as Api
}

export function createServer(createFn: CreateThreadFn, api: Api, target?: any) {
  const opts: ThreadOptions<Api> = {
    expose: {
      init: api.init,
      sendMessage: api.sendMessage,
    },
  }
  if (target)
    return createFn(target, opts)

  const createFnWithoutTarget = createFn as (opts: ThreadOptions<Api>) => Thread<Record<string, any>>
  return createFnWithoutTarget(opts)
}
