// types.ts
import { FileAtInfo } from 'tabby-chat-panel/index'

import { ContextSourceKind } from '@/lib/gql/generates/graphql'

export type MentionCategory = 'file' | 'symbol'

export interface SourceItem {
  id: string
  label: string
  category: MentionCategory
  type: 'source'
  data: {
    sourceKind: ContextSourceKind
    sourceId: string
    sourceName: string
  }
}

export type OptionItem = SourceItem
export type MenuView = 'categories' | 'files' | 'symbols'

export interface MenuState {
  view: MenuView
  category?: 'file' | 'symbol'
}

export interface MentionNodeAttrs {
  id: string
  label: string
  category: 'file' | 'symbol'
}

export interface SuggestionItem extends FileAtInfo {
  id: string
  label: string
  category: 'file' | 'symbol'
  type: 'source'
  data: {
    sourceId: string
    sourceName: string
    sourceKind: string
  }
}

export interface SuggestionState {
  items: SuggestionItem[]
  command: (item: MentionNodeAttrs) => void
  clientRect: () => DOMRect | null
  selectedIndex: number
}
