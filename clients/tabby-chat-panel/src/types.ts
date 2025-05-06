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
 * This is used for files in the workspace, but not in a Git repository.
 */
export interface FilepathInWorkspace {
  kind: 'workspace'

  /**
   * A string that is a relative path to `baseDir`.
   */
  filepath: string

  /**
   * A string that can be parsed as a URI, used to identify the directory in the client.
   * The scheme of the URI could be 'file' or some other protocol to access the directory.
   */
  baseDir: string
}

/**
 * This is used for files not in a Git repository and not in the workspace.
 * Also used for untitled files not saved.
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
 * Represents a filepath to identify a file.
 */
export type Filepath = FilepathInGitRepository | FilepathInWorkspace | FilepathUri

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
 * Represents a client-side terminal context.
 * This type should only be used for sending context from client to server.
 * @since 0.10.0
 */
export interface TerminalContext {
  kind: 'terminal'

  /**
   * The terminal name.
   */
  name: string

  /**
   * The terminal process id
   */
  processId: number | undefined

  /**
   * The selected text in the terminal.
   */
  selection: string
}

/**
 * Represents a client-side context.
 * This type should only be used for sending context from client to server.
 * @since 0.10.0 added TerminalContext
 */
export type EditorContext = EditorFileContext | TerminalContext

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
 * Represents a file reference for retrieving file content.
 * If `range` is not provided, the entire file is considered.
 */
export interface FileRange {
  /**
   * The file path of the file.
   */
  filepath: Filepath

  /**
   * The range of the selected content in the file.
   * If the range is not provided, the whole file is considered.
   */
  range?: LineRange | PositionRange
}

/**
 * Includes information about a git repository in workspace folder
 */
export interface GitRepository {
  url: string
}
