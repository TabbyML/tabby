import { create } from 'zustand'
import { persist } from 'zustand/middleware'

import { nanoid } from '@/lib/utils'

import { ThreadRunContexts } from '../types'

const excludeFromState = ['activeChatId', 'pendingUserMessage']

export interface ChatState {
  activeChatId: string | undefined
  selectedModel: string | undefined
  selectedRepoSourceId: string | undefined
  enableActiveSelection: boolean
  // question from homepage
  pendingUserMessage:
    | {
        content?: string
        context: ThreadRunContexts | undefined
      }
    | undefined
}

const initialState: ChatState = {
  activeChatId: nanoid(),
  selectedModel: undefined,
  selectedRepoSourceId: undefined,
  enableActiveSelection: true,
  pendingUserMessage: undefined
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
