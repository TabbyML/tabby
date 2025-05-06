import type { EditorContext } from './types'

export const serverApiVersionList = ['0.8.0', '0.9.0', '0.10.0']

export type ServerApi = ServerApiV0_10

export interface ServerApiList {
  '0.8.0': ServerApiV0_8
  '0.9.0': ServerApiV0_9 | undefined
  '0.10.0': ServerApiV0_10 | undefined
}

export interface ServerApiV0_8 {
  /**
   * The initialization request sent from the client to the server.
   * @param request {@link InitRequest}
   */
  init: (request: InitRequest) => Promise<void>

  /**
   * @deprecated
   * Show an error message in the chat panel.
   * @param error {@link ErrorMessage}
   */
  showError: (error: ErrorMessage) => Promise<void>

  /**
   * @deprecated
   * Clear the current error message.
   */
  cleanError: () => Promise<void>

  /**
   * Update the style and theme.
   * @param style css style string
   * @param themeClass dark or light
   */
  updateTheme: (style: string, themeClass: 'dark' | 'light') => Promise<void>

  /**
   * Execute a predefined command.
   * @param command {@link ChatCommand}
   */
  executeCommand: (command: ChatCommand) => Promise<void>

  /**
   * Navigate to the specified view.
   * @param view {@link ChatView}
   */
  navigate: (view: ChatView) => Promise<void>

  /**
   * Add a relevant editor context.
   * @param context the {@link EditorContext} to add as the relevant context.
   */
  addRelevantContext: (context: EditorContext) => Promise<void>

  /**
   * Notify the server that the user has changed the active selection in the editor.
   * @param selection the active selected {@link EditorContext} in the editor.
   */
  updateActiveSelection: (selection: EditorContext | undefined | null) => Promise<void>
}

export interface ServerApiV0_9 extends ServerApiV0_8 {
  /**
   * Get the api version of the server.
   * @returns the version string.
   */
  getVersion: () => Promise<string>
}

export interface ServerApiV0_10 extends ServerApiV0_9 {
  /**
   * @since 0.10.0 added 'explain-terminal'
   */
  executeCommand: (command: ChatCommand) => Promise<void>

  /**
   * @since 0.10.0 added terminal context support
   */
  addRelevantContext: (context: EditorContext) => Promise<void>
}

/**
 * Predefined commands used in {@link ServerApiV0_8.executeCommand}.
 * - 'explain': Explain the selected code.
 * - 'fix': Fix bugs in the selected code.
 * - 'generate-docs': Generate documentation for the selected code.
 * - 'generate-tests': Generate tests for the selected code.
 * - 'code-review': Review the selected code and provide feedback.
 * - 'explain-terminal': Explain the selected text in the terminal. @since 0.10.0
 */
export type ChatCommand = 'explain' | 'fix' | 'generate-docs' | 'generate-tests' | 'code-review' | 'explain-terminal'

/**
 * The views used in {@link ServerApiV0_8.navigate}.
 */
export type ChatView = 'new-chat' | 'history'

/**
 * The params used in {@link ServerApiV0_8.init}.
 */
export interface InitRequest {
  fetcherOptions: {
    /**
     * The authorization token.
     */
    authorization: string

    /**
     * Optional http headers to be sent with every request.
     */
    headers?: Record<string, unknown>
  }

  /**
   * Enabled the keyboard event handler to workaround for vscode webview issue:
   * shortcut (cmd+a, cmd+c, cmd+v, cmd+x) not work in nested iframe in vscode webview
   * see https://github.com/microsoft/vscode/issues/129178
   */
  useMacOSKeyboardEventHandler?: boolean
}

/**
 * @deprecated
 * The params used in {@link ServerApiV0_8.showError}.
 */
export interface ErrorMessage {
  title?: string
  content: string
}
