import { createThreadFromIframe, createThreadFromInsideIframe } from 'tabby-threads'
import { version } from '../package.json'

export const TABBY_CHAT_PANEL_API_VERSION: string = version

/**
 * Represents a position in a file.
 */
export interface Position {
  /**
   * 1-based line number
   */
  line: number
  /**
   * 1-based character number
   */
  character: number
}

/**
 * Represents a range in a file.
 */
export interface PositionRange {
  /**
   * The start position of the range.
   */
  start: Position
  /**
   * The end position of the range.
   */
  end: Position
}

/**
 * Represents a range of lines in a file.
 */
export interface LineRange {
  /**
   * 1-based line number
   */
  start: number
  /**
   * 1-based line number
   */
  end: number
}

/**
 * Represents a location in a file.
 * It could be a 1-based line number, a line range, a position or a position range.
 */
export type Location = number | LineRange | Position | PositionRange

/**
 * Represents a client-side file context.
 * This type should only be used for sending context from client to server.
 */
export interface EditorFileContext {
  kind: 'file'

  /**
   * The filepath of the file.
   */
  filepath: Filepath

  /**
   * The range of the selected content in the file.
   * If the range is not provided, the whole file is considered.
   */
  range?: LineRange | PositionRange

  /**
   * The content of the file context.
   */
  content: string
}

/**
 * Represents a client-side context.
 * This type should only be used for sending context from client to server.
 */
export type EditorContext = EditorFileContext

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

// @deprecated
export interface ErrorMessage {
  title?: string
  content: string
}

/**
 * Represents a filepath to identify a file.
 */
export type Filepath = FilepathInGitRepository | FilepathUri

/**
 * This is used for files in a Git repository, and should be used in priority.
 */
export interface FilepathInGitRepository {
  kind: 'git'

  /**
   * A string that is a relative path in the repository.
   */
  filepath: string

  /**
   * A URL used to identify the Git repository in both the client and server.
   * The URL should be valid for use as a git remote URL, for example:
   * 1. 'https://github.com/TabbyML/tabby'
   * 2. 'git://github.com/TabbyML/tabby.git'
   */
  gitUrl: string

  /**
   * An optional Git revision which the file is at.
   */
  revision?: string
}

/**
 * This is used for files not in a Git repository.
 */
export interface FilepathUri {
  kind: 'uri'

  /**
   * A string that can be parsed as a URI, used to identify the file in the client.
   * The scheme of the URI could be:
   * - 'untitled' means a new file not saved.
   * - 'file', 'vscode-vfs' or some other protocol to access the file.
   */
  uri: string
}

/**
 * Represents a file and a location in it.
 */
export interface FileLocation {
  /**
   * The filepath of the file.
   */
  filepath: Filepath

  /**
   * The location in the file.
   * It could be a 1-based line number, a line range, a position or a position range.
   * If the location is not provided, the whole file is considered.
   */
  location?: Location
}

/**
 * Represents a hint to help find a symbol.
 */
export interface LookupSymbolHint {
  /**
   * The filepath of the file to search the symbol.
   */
  filepath?: Filepath

  /**
   * The location in the file to search the symbol.
   */
  location?: Location
}

/**
 * Includes information about a symbol returned by the {@link ClientApiMethods.lookupSymbol} method.
 */
export interface SymbolInfo {
  /**
   * Where the symbol is found.
   */
  source: FileLocation
  /**
   * The target location to navigate to when the symbol is clicked.
   */
  target: FileLocation
}

/**
 * Includes information about a git repository in workspace folder
 */
export interface GitRepository {
  url: string
}

/**
 * Predefined commands.
 * - 'explain': Explain the selected code.
 * - 'fix': Fix bugs in the selected code.
 * - 'generate-docs': Generate documentation for the selected code.
 * - 'generate-tests': Generate tests for the selected code.
 */
export type ChatCommand = 'explain' | 'fix' | 'generate-docs' | 'generate-tests'

export interface ServerApi {
  init: (request: InitRequest) => void

  /**
   * Execute a predefined command.
   * @param command The command to execute.
   */
  executeCommand: (command: ChatCommand) => Promise<void>

  // @deprecated
  showError: (error: ErrorMessage) => void
  // @deprecated
  cleanError: () => void

  addRelevantContext: (context: EditorContext) => void
  updateTheme: (style: string, themeClass: string) => void
  updateActiveSelection: (context: EditorContext | null) => void
}

export interface ClientApiMethods {
  refresh: () => Promise<void>

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

  /**
   * Find the target symbol and return the symbol information.
   * @param symbol The symbol to find.
   * @param hints The optional {@link LookupSymbolHint} list to help find the symbol. The hints should be sorted by priority.
   * @returns The symbol information if found, otherwise undefined.
   */
  lookupSymbol?: (symbol: string, hints?: LookupSymbolHint[] | undefined) => Promise<SymbolInfo | undefined>

  /**
   * Open the target file location in the editor.
   * @param target The target file location to open.
   * @returns Whether the file location is opened successfully.
   */
  openInEditor: (target: FileLocation) => Promise<boolean>

  /**
   * Open the target URL in the external browser.
   * @param url The target URL to open.
   */
  openExternal: (url: string) => Promise<void>

  // Provide all repos found in workspace folders.
  readWorkspaceGitRepositories?: () => Promise<GitRepository[]>

  /**
   * @returns The active selection of active editor.
   */
  getActiveEditorSelection: () => Promise<EditorFileContext | null>
}

export interface ClientApi extends ClientApiMethods {
  /**
   * Checks if the client supports this capability.
   * This method is designed to check capability across different clients (IDEs).
   * Note: This method should not be used to ensure compatibility across different chat panel SDK versions.
   */
  hasCapability: (method: keyof ClientApiMethods) => Promise<boolean>
}

export function createClient(target: HTMLIFrameElement, api: ClientApiMethods): ServerApi {
  return createThreadFromIframe(target, {
    expose: {
      refresh: api.refresh,
      onApplyInEditor: api.onApplyInEditor,
      onApplyInEditorV2: api.onApplyInEditorV2,
      onLoaded: api.onLoaded,
      onCopy: api.onCopy,
      onKeyboardEvent: api.onKeyboardEvent,
      lookupSymbol: api.lookupSymbol,
      openInEditor: api.openInEditor,
      openExternal: api.openExternal,
      readWorkspaceGitRepositories: api.readWorkspaceGitRepositories,
      getActiveEditorSelection: api.getActiveEditorSelection,
    },
  })
}

export function createServer(api: ServerApi): ClientApi {
  return createThreadFromInsideIframe({
    expose: {
      init: api.init,
      executeCommand: api.executeCommand,
      showError: api.showError,
      cleanError: api.cleanError,
      addRelevantContext: api.addRelevantContext,
      updateTheme: api.updateTheme,
      updateActiveSelection: api.updateActiveSelection,
    },
  })
}
