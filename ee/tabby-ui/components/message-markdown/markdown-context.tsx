import { createContext } from 'react'
import { FileContext } from 'tabby-chat-panel/index'

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
  onNavigateSymbol?: (filepaths: string[], keyword: string) => void
  supportsOnApplyInEditorV2: boolean
  activeSelection?: FileContext
}

export const MessageMarkdownContext =
  createContext<MessageMarkdownContextValue>({} as MessageMarkdownContextValue)
