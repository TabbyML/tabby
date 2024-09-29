import { createThreadFromIframe, createThreadFromInsideIframe } from '@quilted/threads'
import { version } from '../package.json'

export const TABBY_CHAT_PANEL_API_VERSION: string = version

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
  headers?: Record<string, unknown>
}

export interface InitRequest {
  fetcherOptions: FetcherOptions
  // Workaround for vscode webview issue:
  // shortcut (cmd+a, cmd+c, cmd+v, cmd+x) not work in nested iframe in vscode webview
  // see https://github.com/microsoft/vscode/issues/129178
  useMacOSKeyboardEventHandler?: boolean
}

export interface OnLoadedParams {
  apiVersion: string
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
  addRelevantContext: (context: Context) => void
  updateTheme: (style: string, themeClass: string) => void
}

export interface ClientApi {
  navigate: (context: Context, opts?: NavigateOpts) => void
  refresh: () => Promise<void>

  onSubmitMessage: (msg: string, relevantContext?: Context[]) => Promise<void>

  onApplyInEditor: (content: string) => void

  // On current page is loaded.
  onLoaded: (params?: OnLoadedParams | undefined) => void

  // On user copy content to clipboard.
  onCopy: (content: string) => void

  onKeyboardEvent: (type: 'keydown' | 'keyup' | 'keypress', event: KeyboardEventInit) => void
}

export interface ChatMessage {
  message: string

  // Client side context - displayed in user message
  selectContext?: Context

  // Client side contexts - displayed in assistant message
  relevantContext?: Array<Context>

  // Client side active selection context - displayed in assistant message
  activeContext?: Context
}

export function createClient(target: HTMLIFrameElement, api: ClientApi): ServerApi {
  return createThreadFromIframe(target, {
    expose: {
      navigate: api.navigate,
      refresh: api.refresh,
      onSubmitMessage: api.onSubmitMessage,
      onApplyInEditor: api.onApplyInEditor,
      onLoaded: api.onLoaded,
      onCopy: api.onCopy,
      onKeyboardEvent: api.onKeyboardEvent,
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
      addRelevantContext: api.addRelevantContext,
      updateTheme: api.updateTheme,
    },
  })
}
