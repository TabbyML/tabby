import type { HTMLAttributes, ReactNode } from 'react'
import type { Content, Editor, EditorEvents } from '@tiptap/react'
import type {
  ChangeItem,
  ChatCommand,
  EditorContext,
  EditorFileContext,
  FileLocation,
  FileRange,
  GetChangesParams,
  GitRepository,
  ListFileItem,
  ListFilesInWorkspaceParams,
  ListSymbolItem,
  ListSymbolsParams,
  LookupSymbolHint,
  SymbolInfo,
  TerminalContext
} from 'tabby-chat-panel'

import type { QuestionAnswerPair, SessionState } from '@/lib/types'

export interface ChatProps extends React.ComponentProps<'div'> {
  threadId: string | undefined
  setThreadId: React.Dispatch<React.SetStateAction<string | undefined>>
  api?: string
  initialMessages?: QuestionAnswerPair[]
  onLoaded?: () => void
  onThreadUpdates?: (messages: QuestionAnswerPair[]) => void
  container?: HTMLDivElement
  docQuery?: boolean
  generateRelevantQuestions?: boolean
  welcomeMessage?: string
  promptFormClassname?: string
  onCopyContent?: (value: string) => void
  onApplyInEditor?:
    | ((content: string) => void)
    | ((content: string, opts?: { languageId: string; smart: boolean }) => void)
  onLookupSymbol?: (
    symbol: string,
    hints?: LookupSymbolHint[] | undefined
  ) => Promise<SymbolInfo | null>
  openInEditor: (target: FileLocation) => Promise<boolean>
  openExternal: (url: string) => Promise<void>
  chatInputRef: React.RefObject<PromptFormRef>
  supportsOnApplyInEditorV2: boolean
  readWorkspaceGitRepositories?: () => Promise<GitRepository[]>
  getActiveEditorSelection?: () => Promise<EditorFileContext | null>
  getActiveTerminalSelection?: () => Promise<TerminalContext | null>
  fetchSessionState?: () => Promise<SessionState | null>
  storeSessionState?: (state: Partial<SessionState>) => Promise<void>
  listFileInWorkspace?: (
    params: ListFilesInWorkspaceParams
  ) => Promise<ListFileItem[]>
  listSymbols?: (param: ListSymbolsParams) => Promise<ListSymbolItem[]>
  readFileContent?: (info: FileRange) => Promise<string | null>
  setShowHistory: React.Dispatch<React.SetStateAction<boolean>>
  runShell?: (command: string) => Promise<void>
  getChanges?: (params: GetChangesParams) => Promise<ChangeItem[]>
}
export interface ChatRef {
  executeCommand: (command: ChatCommand) => Promise<void>
  stop: () => void
  isLoading: boolean
  addRelevantContext: (context: EditorContext) => void
  focus: () => void
  updateActiveSelection: (context: EditorContext | null) => void
  newChat: () => void
}

/**
 * PromptProps defines the props for the PromptForm component.
 */
export interface PromptProps
  extends Omit<HTMLAttributes<HTMLDivElement>, 'onSubmit'> {
  /**
   * A callback function that handles form submission.
   * Returns a Promise for handling async operations.
   */
  onSubmit: (value: string) => Promise<void>
  /**
   * Indicates whether the form (or chat) is in a loading/submitting state.
   */
  isLoading: boolean
  onUpdate?: (p: EditorEvents['update']) => void
}

/**
 * PromptFormRef defines the methods exposed by PromptForm via forwardRef.
 */
export interface PromptFormRef {
  /**
   * Focuses the editor within PromptForm.
   */
  focus: () => void
  /**
   * Programmatically sets the editor's text content.
   */
  setInput: (value: Content) => void
  /**
   * Returns the current text content of the editor.
   */
  input: string
  editor: Editor | null
}

/**
 * Represents a command item for use in mentions.
 */
export interface CommandItem {
  id: string
  name: string
  command: string
  description?: string
}

/**
 * Represents a file item in the workspace.
 */
export type FileItem = ListFileItem | ListSymbolItem

/**
 * Represents a source item for the mention suggestion list.
 */
export interface SourceItem {
  id: string
  name: string
  filepath?: string
  category: CategoryMenu
  isRootCategoryItem?: boolean
  fileItem?: FileItem
  icon: ReactNode
  command?: string
  description?: string
}

export interface CategoryItem {
  label: string
  categoryKind: CategoryMenu
  icon: ReactNode
}

/**
 * Raw mention data in the form editor
 */
export type EditorMentionData =
  | {
      category: 'file'
      fileItem: ListFileItem
    }
  | {
      category: 'symbol'
      fileItem: ListSymbolItem
    }
  | {
      category: 'command'
      command: string
    }

export type CategoryMenu = 'file' | 'symbol' | 'category' | 'command'

/**
 * Defines the attributes to be stored in a mention node.
 */
export interface MentionNodeAttrs {
  category: 'file' | 'symbol' | 'command'
  filepath?: string
  label: string
  fileItem?: FileItem
  command?: string
}

/**
 * Maintains the current state of the mention feature while typing.
 */
export interface MentionState {
  items: SourceItem[]
  command: ((props: MentionNodeAttrs) => void) | null
  query: string
  selectedIndex: number
}

/**
 * Represents a change item in the git diff.
 */
export interface GitChange {
  id: string
  filepath: string
  additions: number
  deletions: number
  diffContent: string
  lineStart?: number
}
