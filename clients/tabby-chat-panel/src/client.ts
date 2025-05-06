import type { EditorFileContext, FileLocation, FileRange, Filepath, GitRepository, LineRange, Location, TerminalContext } from './types'

export interface ClientApi {
  /**
   * @deprecated
   * Notify the client that the server is ready.
   * @param params {@link OnLoadedParams}
   */
  onLoaded?: (params?: OnLoadedParams | undefined) => Promise<void>

  /**
   * Forces the webview to refresh.
   */
  refresh: () => Promise<void>

  /**
   * Open the target file location in the editor.
   * @param target The target {@link FileLocation} to open.
   * @returns Whether the file location is opened successfully.
   */
  openInEditor: (target: FileLocation) => Promise<boolean>

  /**
   * Open the target URL in the external browser.
   * @param url The target URL to open.
   */
  openExternal: (url: string) => Promise<void>

  /**
   * Get the active editor selection as context, or the whole file if no selection.
   * @returns The {@link EditorFileContext} of the active editor, or `null` if no active editor is found.
   */
  getActiveEditorSelection: () => Promise<EditorFileContext | null>

  /**
   * Get the active terminal selection as context.
   * @returns The {@link TerminalContext} of the active terminal, or `null` if no active terminal selection is found.
   * @since 0.10.0
   */
  getActiveTerminalSelection?: () => Promise<TerminalContext | null>

  /**
   * Copy the content to the clipboard.
   * @param content The content to copy.
   */
  onCopy: (content: string) => Promise<void>

  /**
   * Apply the content into the active editor.
   * @param content The content to apply.
   * See {@link onApplyInEditorV2} for Smart Apply support.
   */
  onApplyInEditor: (content: string) => Promise<void>

  /**
   * Apply the content into the active editor, with Smart Apply support.
   * @param content The content to apply.
   * @param options The optional {@link ApplyInEditorOptions} to control the behavior of the apply.
   */
  onApplyInEditorV2?: (content: string, options?: ApplyInEditorOptions) => Promise<void>

  /**
   * Notify the client that a keyboard event has been triggered.
   * @param type The type of the keyboard event.
   * @param event The {@link KeyboardEventInit} to handle.
   */
  onKeyboardEvent?: (type: 'keydown' | 'keyup' | 'keypress', event: KeyboardEventInit) => Promise<void>

  /**
   * Fetch all git repositories information for the current workspace.
   * @returns The list of {@link GitRepository} objects, or an empty array if no repositories are found.
   */
  readWorkspaceGitRepositories?: () => Promise<GitRepository[]>

  /**
   * Returns the content of a file within the specified range.
   * @param fileRange The {@link FileRange} to read the content from.
   * @returns The content of the file as a string, or `null` if the file or range cannot be accessed.
   */
  readFileContent?: (fileRange: FileRange) => Promise<string | null>

  /**
   * Returns a list of file information matching the specified query.
   * @param params An {@link ListFilesInWorkspaceParams} object that includes a search query and a limit for the results.
   * @returns An array of {@link ListFileItem} objects that could be empty.
   */
  listFileInWorkspace?: (params: ListFilesInWorkspaceParams) => Promise<ListFileItem[]>

  /**
   * Returns active editor symbols when no query is provided. Otherwise, returns workspace symbols that match the query.
   * @param params An {@link ListSymbolsParams} object that includes a search query and a limit for the results.
   * @returns An array of {@link ListSymbolItem} objects that could be empty.
   */
  listSymbols?: (params: ListSymbolsParams) => Promise<ListSymbolItem[]>

  /**
   * Find the target symbol and return the symbol information.
   * @param symbol The symbol to find.
   * @param hints The optional {@link LookupSymbolHint} list to help find the symbol. The hints should be sorted by priority.
   * @returns The {@link SymbolInfo} object if found, or `null` if not found.
   */
  lookupSymbol?: (symbol: string, hints?: LookupSymbolHint[] | undefined) => Promise<SymbolInfo | null>

  /**
   * Fetch the saved session state from the client.
   * When initialized, the chat panel attempts to fetch the saved session state to restore the session.
   * @param keys The keys to be fetched. If not provided, all keys will be returned.
   * @return The saved persisted state, or `null` if no state is found.
   */
  fetchSessionState?: (keys?: string[] | undefined) => Promise<Record<string, unknown> | null>

  /**
   * Save the session state of the chat panel.
   * The client is responsible for maintaining the state in case of a webview reload.
   * The saved state should be merged and updated by the record key.
   * @param state The state to save.
   */
  storeSessionState?: (state: Record<string, unknown>) => Promise<void>

  /**
   * Retrieves the Git diff changes from the client IDE.
   * @param params The parameters to filter or specify the changes to retrieve.
   * @returns A Promise resolving to an array of {@link ChangeItem} objects representing the Git diff results.
   */
  getChanges?: (params: GetChangesParams) => Promise<ChangeItem[]>

  /**
   * Run a terminal command in the IDE Terminal shell.
   * @param command The command to run in the terminal.
   * @returns void
   */
  runShell?: (command: string) => Promise<void>
}

/**
 * @deprecated
 * The params used in {@link ClientApi.onLoaded}.
 */
export interface OnLoadedParams {
  /**
   * The current version used by the server.
   */
  apiVersion: string
}

/**
 * The options used in {@link ClientApi.applyInEditorV2}.
 */
export interface ApplyInEditorOptions {
  /**
   * The language of the content to apply.
   */
  languageId?: string

  /**
   * Controls whether use smart apply.
   */
  smart?: boolean
}

/**
 * The options used in {@link ClientApi.lookupSymbol}.
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
 * The result returned by {@link ClientApi.lookupSymbol}.
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
 * The params used in {@link ClientApi.listFileInWorkspace}.
 */
export interface ListFilesInWorkspaceParams {
  /**
   * The query string to filter the files.
   * The query string could be an empty string. In this case, we do not read all files in the workspace,
   * but only list the opened files in the editor.
   */
  query: string

  /**
   * The maximum number of files to list.
   */
  limit?: number
}

/**
 * The item returned by {@link ClientApi.listFileInWorkspace}.
 */
export interface ListFileItem {
  /**
   * The filepath of the file.
   */
  filepath: Filepath

  /**
   * Source of the file, indicating whether it comes from an editor or search results.
   * When undefined, the file is assumed to be from the workspace by default.
   * This property helps determine how the file should be handled in the UI.
   */
  source?: 'openedInEditor' | 'searchResult'
}

/**
 * The params used in {@link ClientApi.listSymbols}.
 */
export interface ListSymbolsParams {
  /**
   * The query string.
   * When the query is empty, returns symbols in the current file.
   */
  query: string

  /**
   * The maximum number of items to return.
   */
  limit?: number
}

/**
 * The item returned by {@link ClientApi.listSymbols}.
 */
export interface ListSymbolItem {
  /**
   * The symbol name.
   */
  label: string

  /**
   * The filepath of the containing file.
   */
  filepath: Filepath

  /**
   * The line range of the symbol definition in the file.
   */
  range: LineRange
}

/**
 * Parameters for retrieving Git changes from the client IDE.
 * Used with {@link ClientApi.getChanges} method.
 */
export interface GetChangesParams {
  /**
   * The maximum number of characters to return in the change content.
   * If not provided, all changes will be returned without character limitation.
   */
  maxChars?: number
}
/**
 * Represents a Git change item returned by the {@link ClientApi.getChanges} method.
 * Contains information about modifications detected in the Git workspace.
 */
export interface ChangeItem {
  /**
   * The content of the change, formatted in the same Diff format as the `git diff` command.
   * This includes information about added, modified, and deleted lines in the affected files.
   */
  content: string
  /**
   * Indicates whether the change has been staged in the Git index.
   * True if the change is staged, false if it is unstaged.
   */
  staged: boolean
}
