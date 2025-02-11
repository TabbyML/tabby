import { createContext, Dispatch, SetStateAction } from 'react'

type PageContextValue = {
  mode: 'edit' | 'view'
  setMode: Dispatch<SetStateAction<'view' | 'edit'>>
  isPathnameInitialized: boolean
  isLoading: boolean
  onDeleteSection: (id: string) => void
  isPageOwner: boolean
  pendingSectionIds: Set<string>
  setPendingSectionIds: (value: SetStateAction<Set<string>>) => void
  currentSectionId: string | undefined
}

export const PageContext = createContext<PageContextValue>(
  {} as PageContextValue
)
