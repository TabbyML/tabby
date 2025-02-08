import { create } from 'zustand'
import { persist } from 'zustand/middleware'

const excludeFromState: string[] = []

export interface PageState {
  pendingThreadId: string | undefined
}

const initialState: PageState = {
  pendingThreadId: undefined
}

export const usePageStore = create<PageState>()(
  persist(
    () => ({
      ...initialState
    }),
    {
      name: 'tabby-page-storage',
      partialize: state =>
        Object.fromEntries(
          Object.entries(state).filter(
            ([key]) => !excludeFromState.includes(key)
          )
        )
    }
  )
)

const set = usePageStore.setState

export const updatePendingThreadId = (threadId: string | undefined) => {
  set(() => ({ pendingThreadId: threadId }))
}
