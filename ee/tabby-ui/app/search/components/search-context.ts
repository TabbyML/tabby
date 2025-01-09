import { createContext } from 'react'

import {
  ContextInfo,
  RepositorySourceListQuery
} from '@/lib/gql/generates/graphql'
import { ExtendedCombinedError } from '@/lib/types'

import { ConversationMessage } from './types'

type SearchContextValue = {
  // flag for initialize the pathname
  isPathnameInitialized: boolean
  isLoading: boolean
  onRegenerateResponse: (id: string) => void
  onSubmitSearch: (question: string) => void
  setDevPanelOpen: (v: boolean) => void
  setConversationIdForDev: (v: string | undefined) => void
  enableDeveloperMode: boolean
  contextInfo: ContextInfo | undefined
  fetchingContextInfo: boolean
  onDeleteMessage: (id: string) => void
  isThreadOwner: boolean
  onUpdateMessage: (
    message: ConversationMessage
  ) => Promise<ExtendedCombinedError | undefined>
  repositories: RepositorySourceListQuery['repositoryList'] | undefined
}

export const SearchContext = createContext<SearchContextValue>(
  {} as SearchContextValue
)
