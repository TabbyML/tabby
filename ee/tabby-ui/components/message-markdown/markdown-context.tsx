import { createContext } from 'react'
import {
  Context,
  FileContext,
  NavigateOpts,
  SymbolInfo
} from 'tabby-chat-panel/index'

import { ContextInfo } from '@/lib/gql/generates/graphql'
import { AttachmentCodeItem } from '@/lib/types'

export type MessageMarkdownContextValue = {
  onCopyContent?: ((value: string) => void) | undefined
  onApplyInEditor?: (
    content: string,
    opts?: { languageId: string; smart: boolean }
  ) => void
  onCodeCitationClick?: (code: AttachmentCodeItem) => void
  onCodeCitationMouseEnter?: (index: number) => void
  onCodeCitationMouseLeave?: (index: number) => void
  contextInfo: ContextInfo | undefined
  fetchingContextInfo: boolean
  canWrapLongLines: boolean
  onLookupSymbol?: (
    filepaths: string[],
    keyword: string
  ) => Promise<SymbolInfo | undefined>
  onNavigateToContext?: (context: Context, opts?: NavigateOpts) => void
  supportsOnApplyInEditorV2: boolean
  activeSelection?: FileContext
}

export const MessageMarkdownContext =
  createContext<MessageMarkdownContextValue>({} as MessageMarkdownContextValue)
