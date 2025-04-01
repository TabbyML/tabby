import { createContext, RefObject } from 'react'
import type {
  ChangeItem,
  FileLocation,
  FileRange,
  GetChangesParams,
  ListFileItem,
  ListFilesInWorkspaceParams,
  ListSymbolItem,
  ListSymbolsParams,
  LookupSymbolHint,
  SymbolInfo
} from 'tabby-chat-panel'

import {
  ContextInfo,
  RepositorySourceListQuery
} from '@/lib/gql/generates/graphql'
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
  ) => Promise<SymbolInfo | null>
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
  getChanges?: (params: GetChangesParams) => Promise<ChangeItem[]>
  // for repo select
  selectedRepoId: string | undefined
  setSelectedRepoId: React.Dispatch<React.SetStateAction<string | undefined>>
  repos: RepositorySourceListQuery['repositoryList'] | undefined
  fetchingRepos: boolean
  runShell?: (command: string) => Promise<void>
  contextInfo: ContextInfo | undefined
  fetchingContextInfo: boolean
}

export const ChatContext = createContext<ChatContextValue>(
  {} as ChatContextValue
)
