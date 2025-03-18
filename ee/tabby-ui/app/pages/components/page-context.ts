import { createContext, Dispatch, SetStateAction } from 'react'

import { ContextInfo, MoveSectionDirection } from '@/lib/gql/generates/graphql'

type PageContextValue = {
  mode: 'edit' | 'view'
  setMode: Dispatch<SetStateAction<'view' | 'edit'>>
  isPathnameInitialized: boolean
  isLoading: boolean
  onDeleteSection: (id: string) => Promise<void>
  isPageOwner: boolean
  pendingSectionIds: Set<string>
  setPendingSectionIds: (value: SetStateAction<Set<string>>) => void
  currentSectionId: string | undefined
  onMoveSectionPosition: (
    sectionId: string,
    direction: MoveSectionDirection
  ) => Promise<void>
  pageIdFromURL: string | undefined
  isNew: boolean
  fetchingContextInfo: boolean
  contextInfo: ContextInfo | undefined
}

export const PageContext = createContext<PageContextValue>(
  {} as PageContextValue
)
