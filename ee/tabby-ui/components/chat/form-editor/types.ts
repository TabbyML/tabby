import { ListFileItem } from 'tabby-chat-panel/index'

/**
 * PromptProps defines the props for the PromptForm component.
 */
export interface PromptProps {
  /**
   * A callback function that handles form submission.
   * It returns a Promise, so you can handle async actions.
   */
  onSubmit: (value: string) => Promise<void>
  /**
   * Indicates if the form (or chat) is in a loading/submitting state.
   */
  isLoading: boolean
}

/**
 * PromptFormRef defines the methods exposed by the PromptForm via forwardRef.
 */
export interface PromptFormRef {
  /**
   * Focus the editor inside PromptForm.
   */
  focus: () => void
  /**
   * Set the content of the editor programmatically.
   */
  setInput: (value: string) => void
  /**
   * Get the current editor text content.
   */
  input: string
}

// TODO: move this into chat-panel in next iterate
/**
 * Represents a file item inside the workspace.
 */
export type FileItem = ListFileItem

/**
 * Represents a file source item for mention suggestions.
 */
export interface SourceItem {
  name: string
  filepath: string
  category: 'file'
  fileItem: FileItem
}

/**
 * The attributes stored in a mention node.
 */
export interface MentionNodeAttrs {
  id: string
  name: string
  category: 'file'
  fileItem: FileItem
}

/**
 * Stores the current state of the mention feature while typing.
 */
export interface MentionState {
  items: SourceItem[]
  command: ((props: MentionNodeAttrs) => void) | null
  query: string
  selectedIndex: number
}
