import { createContext, Dispatch, SetStateAction } from 'react'

import { ContextInfo, MoveSectionDirection } from '@/lib/gql/generates/graphql'

type PageContextValue = {
  isNew: boolean
  mode: 'edit' | 'view'
  setMode: Dispatch<SetStateAction<'view' | 'edit'>>
  isPathnameInitialized: boolean
  pageIdFromURL: string | undefined
  fetchingContextInfo: boolean
  contextInfo: ContextInfo | undefined
  isLoading: boolean
  isPageOwner: boolean
  pendingSectionIds: Set<string>
  setPendingSectionIds: (value: SetStateAction<Set<string>>) => void
  currentSectionId: string | undefined
  onDeleteSection: (id: string) => Promise<void>
  onMoveSectionPosition: (
    sectionId: string,
    direction: MoveSectionDirection
  ) => Promise<void>

  enableDeveloperMode: boolean
  devPanelOpen: boolean
  setDevPanelOpen: Dispatch<SetStateAction<boolean>>
}

export const PageContext = createContext<PageContextValue>(
  {} as PageContextValue
)
