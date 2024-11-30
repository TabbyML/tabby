import { createThreadFromIframe, createThreadFromInsideIframe } from 'tabby-threads'
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
  updateActiveSelection: (context: Context | null) => void
}
export interface SymbolInfo {
  sourceFile: string
  sourceLine: number
  sourceCol: number
  targetFile: string
  targetLine: number
  targetCol: number
}
export interface ClientApiMethods {
  navigate: (context: Context, opts?: NavigateOpts) => void
  refresh: () => Promise<void>

  onSubmitMessage: (msg: string, relevantContext?: Context[]) => Promise<void>

  // apply content into active editor, version 1, not support smart apply
  onApplyInEditor: (content: string) => void

  // version 2, support smart apply and normal apply
  onApplyInEditorV2?: (
    content: string,
    opts?: { languageId: string, smart: boolean }
  ) => void

  // On current page is loaded.
  onLoaded: (params?: OnLoadedParams | undefined) => void

  // On user copy content to clipboard.
  onCopy: (content: string) => void

  onKeyboardEvent: (type: 'keydown' | 'keyup' | 'keypress', event: KeyboardEventInit) => void
  // navigate to lsp definition by symbol
  onNavigateSymbol: (filepaths: string[], keywords: string) => void

  // on hover symbol return symbol info if exist
  onHoverSymbol: (filepaths: string[], keyword: string) => Promise<SymbolInfo | undefined>
}

export interface ClientApi extends ClientApiMethods {
  // this is inner function cover by tabby-threads
  // the function doesn't need to expose to client but can call by client
  hasCapability: (method: keyof ClientApiMethods) => Promise<boolean>
}

export const clientApiKeys: (keyof ClientApiMethods)[] = [
  'navigate',
  'refresh',
  'onSubmitMessage',
  'onApplyInEditor',
  'onApplyInEditorV2',
  'onLoaded',
  'onCopy',
  'onKeyboardEvent',
]

export interface ChatMessage {
  message: string

  // Client side context - displayed in user message
  selectContext?: Context

  // Client side contexts - displayed in assistant message
  relevantContext?: Array<Context>

  // Client side active selection context - displayed in assistant message
  activeContext?: Context
}

export function createClient(target: HTMLIFrameElement, api: ClientApiMethods): ServerApi {
  return createThreadFromIframe(target, {
    expose: {
      navigate: api.navigate,
      refresh: api.refresh,
      onSubmitMessage: api.onSubmitMessage,
      onApplyInEditor: api.onApplyInEditor,
      onLoaded: api.onLoaded,
      onCopy: api.onCopy,
      onKeyboardEvent: api.onKeyboardEvent,
      onNavigateSymbol: api.onNavigateSymbol,
      onHoverSymbol: api.onHoverSymbol,
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
      updateActiveSelection: api.updateActiveSelection,
    },
  })
}
