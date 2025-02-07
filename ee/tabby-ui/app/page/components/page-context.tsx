import { createContext, Dispatch, SetStateAction } from 'react'

import { ExtendedCombinedError } from '@/lib/types'

type PageContextValue = {
  mode: 'edit' | 'view'
  setMode: Dispatch<SetStateAction<'view' | 'edit'>>
  isPathnameInitialized: boolean
  isLoading: boolean
  onAddSection: (title: string) => void
  onDeleteSection: (id: string) => void
  isPageOwner: boolean
  onUpdateSectionContent: (
    message: string
  ) => Promise<ExtendedCombinedError | undefined>
}

export const PageContext = createContext<PageContextValue>(
  {} as PageContextValue
)
