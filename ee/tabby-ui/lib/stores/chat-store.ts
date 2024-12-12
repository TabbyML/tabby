import { create } from 'zustand'
import { persist } from 'zustand/middleware'

import { nanoid } from '@/lib/utils'

const excludeFromState = ['activeChatId']

export interface ChatState {
  activeChatId: string | undefined
  selectedModel: string | undefined
  enableActiveSelection: boolean
  enableIndexedRepository: boolean
}

const initialState: ChatState = {
  activeChatId: nanoid(),
  selectedModel: undefined,
  enableActiveSelection: true,
  enableIndexedRepository: true
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
