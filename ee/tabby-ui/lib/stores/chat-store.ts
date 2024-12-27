import { create } from 'zustand'
import { persist } from 'zustand/middleware'

import { nanoid } from '@/lib/utils'

const excludeFromState = ['activeChatId']

export interface ChatState {
  activeChatId: string | undefined
  selectedModel: string | undefined
  selectedRepoSourceId: string | undefined
  enableActiveSelection: boolean
}

const initialState: ChatState = {
  activeChatId: nanoid(),
  selectedModel: undefined,
  selectedRepoSourceId: undefined,
  enableActiveSelection: true
}

export const useChatStore = create<ChatState>()(
  persist(
    () => ({
      ...initialState
    }),
    {
      name: 'tabby-chat-storage',
      partialize: state =>
        Object.fromEntries(
          Object.entries(state).filter(
            ([key]) => !excludeFromState.includes(key)
          )
        ),
      // version for breaking change
      version: 1
    }
  )
)
