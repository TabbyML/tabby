import { createContext, Dispatch, SetStateAction } from 'react'

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
  onUpdateSectionPosition: (
    sectionId: string,
    position: number
  ) => Promise<void>
}

export const PageContext = createContext<PageContextValue>(
  {} as PageContextValue
)
