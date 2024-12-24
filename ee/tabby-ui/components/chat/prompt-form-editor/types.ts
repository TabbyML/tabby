/* eslint-disable no-console */
import { AtInfo } from 'tabby-chat-panel/index'

/**
 * Types of mention categories, such as files, symbols, or categories.
 * 'categories' serves as a root-level menu before selecting 'files' or 'symbols'.
 */
export type MentionCategory = 'files' | 'symbols' | 'categories'

/**
 * The first level pop menu.
 */
export const CATEGORIES_MENU: SourceItem[] = [
  {
    name: 'Files',
    category: 'categories',
    filepath: ''
  },
  {
    name: 'Symbols',
    category: 'categories',
    filepath: ''
  }
]

/**
 * Represents an item that can be mentioned, such as a file or symbol.
 */
export interface SourceItem {
  /**
   * Original 'AtInfo' object, It could be optional if we are in root level
   */
  atInfo?: AtInfo

  /**
   * Name derived from 'atInfo.name'.
   * Typically a filename or symbol name that will be shown in the mention list.
   */
  name: string

  /**
   * Filepath or a descriptive path to be shown in the UI.
   */
  filepath: string

  /**
   * The mention category, e.g. 'files', 'symbols', or 'categories'.
   */
  category: MentionCategory
}

/**
 * Alias of SourceItem, used for option lists.
 */
export type OptionItem = SourceItem

/**
 * Manages the current view of the mention menu.
 * For example, 'categories' or directly 'files'/'symbols'.
 */
export interface MenuState {
  view: MentionCategory
}

/**
 * Attributes for a mention node used by the Tiptap editor.
 * 'id' is required by Tiptap. 'atInfo' holds optional extra data.
 */
export interface MentionNodeAttrs {
  id: string
  name: string
  category: MentionCategory
  atInfo: AtInfo // TODO: Consider removing if not needed in final implementation
}

/**
 * Represents the suggestion popover state:
 * - items: the current list of potential mentions
 * - command: the Tiptap command function to insert the mention
 * - clientRect: a function returning the current bounding box for positioning
 * - selectedIndex: the index of the currently highlighted item
 */
export interface SuggestionState {
  items: SourceItem[]
  command: (props: MentionNodeAttrs) => void
  clientRect: () => DOMRect | null
  selectedIndex: number
}
