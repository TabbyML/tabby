import { createThreadFromIframe, createThreadFromInsideIframe } from '@quilted/threads'

export interface LineRange {
  start: number
  end: number
}

export interface FileContext {
  kind: 'file'
  range: LineRange
  filepath: string
  content: string
  git_url: string
}

export type Context = FileContext

export interface FetcherOptions {
  authorization: string
}

export interface InitRequest {
  fetcherOptions: FetcherOptions
}

export interface ErrorMessage {
  title?: string
  content: string
}

export interface NavigateOpts {
  openInEditor?: boolean
}

export interface ServerApi {
  init: (request: InitRequest) => void
  sendMessage: (message: ChatMessage) => void
  showError: (error: ErrorMessage) => void
  cleanError: () => void
  addClientSelectedContext: (context: Context) => void
}

export interface ClientApi {
  navigate: (context: Context, opts?: NavigateOpts) => void
  refresh: () => Promise<void>
  onSubmitMessage?: (msg: string, relevantContext?: Context[]) => Promise<void>
  onApplyInEditor?: (content: string) => void
}

export interface ChatMessage {
  message: string
  selectContext?: Context // Client side context - displayed in user message
  relevantContext?: Array<Context> // Client side contexts - displayed in assistant message
}

export function createClient(target: HTMLIFrameElement, api: ClientApi): ServerApi {
  return createThreadFromIframe(target, {
    expose: {
      navigate: api.navigate,
      refresh: api.refresh,
      onSubmitMessage: api.onSubmitMessage,
      onApplyInEditor: api.onApplyInEditor,
    },
  })
}

export function createServer(api: ServerApi): ClientApi {
  return createThreadFromInsideIframe({
    expose: {
      init: api.init,
      sendMessage: api.sendMessage,
      showError: api.showError,
      cleanError: api.cleanError,
      addClientSelectedContext: api.addClientSelectedContext,
    },
  })
}
