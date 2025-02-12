import { createContext, Dispatch, SetStateAction } from 'react'

import { MoveSectionDirection } from '@/lib/gql/generates/graphql'

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
}

export const PageContext = createContext<PageContextValue>(
  {} as PageContextValue
)
