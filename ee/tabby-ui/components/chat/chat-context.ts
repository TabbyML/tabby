import { createContext, RefObject } from 'react'
import type {
  FileLocation,
  FileRange,
  ListFileItem,
  ListFilesInWorkspaceParams,
  ListSymbolItem,
  ListSymbolsParams,
  LookupSymbolHint,
  SymbolInfo
} from 'tabby-chat-panel'

import { RepositorySourceListQuery } from '@/lib/gql/generates/graphql'
import { Context, MessageActionType, QuestionAnswerPair } from '@/lib/types'

import { PromptFormRef } from './types'

export type ChatContextValue = {
  initialized: boolean
  threadId: string | undefined
  isLoading: boolean
  // FIXME: remove this?
  qaPairs: QuestionAnswerPair[]
  handleMessageAction: (
    userMessageId: string,
    action: MessageActionType
  ) => void
  onClearMessages: () => void
  container?: HTMLDivElement
  onCopyContent?: (value: string) => void
  onApplyInEditor?:
    | ((content: string) => void)
    | ((content: string, opts?: { languageId: string; smart: boolean }) => void)
  onLookupSymbol?: (
    symbol: string,
    hints?: LookupSymbolHint[] | undefined
  ) => Promise<SymbolInfo | undefined>
  openInEditor: (target: FileLocation) => Promise<boolean>
  openExternal: (url: string) => Promise<void>
  activeSelection: Context | null
  relevantContext: Context[]
  setRelevantContext: React.Dispatch<React.SetStateAction<Context[]>>
  chatInputRef: RefObject<PromptFormRef>
  supportsOnApplyInEditorV2: boolean
  listFileInWorkspace?: (
    params: ListFilesInWorkspaceParams
  ) => Promise<ListFileItem[]>
  listSymbols?: (param: ListSymbolsParams) => Promise<ListSymbolItem[]>
  readFileContent?: (info: FileRange) => Promise<string | null>

  // for repo select
  selectedRepoId: string | undefined
  setSelectedRepoId: React.Dispatch<React.SetStateAction<string | undefined>>
  repos: RepositorySourceListQuery['repositoryList'] | undefined
  fetchingRepos: boolean
}

export const ChatContext = createContext<ChatContextValue>(
  {} as ChatContextValue
)
