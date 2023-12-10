import { create } from 'zustand'
import { persist } from 'zustand/middleware'

import { Chat } from '@/lib/types'
import { nanoid } from '@/lib/utils'

const excludeFromState = ['_hasHydrated', 'setHasHydrated', 'activeChatId']

export interface ChatState {
  chats: Chat[] | undefined
  activeChatId: string | undefined
  _hasHydrated: boolean
  setHasHydrated: (state: boolean) => void
}

const initialState: Omit<ChatState, 'setHasHydrated' | 'deleteChat'> = {
  _hasHydrated: false,
  chats: undefined,
  activeChatId: nanoid()
}

export const useChatStore = create<ChatState>()(
  persist(
    set => {
      return {
        ...initialState,
        setHasHydrated: (state: boolean) => {
          set({
            _hasHydrated: state
          })
        }
      }
    },
    {
      name: 'tabby-chat-storage',
      partialize: state =>
        Object.fromEntries(
          Object.entries(state).filter(
            ([key]) => !excludeFromState.includes(key)
          )
        ),
      onRehydrateStorage() {
        return state => {
          if (state) {
            state.setHasHydrated(true)
          }
        }
      }
    }
  )
)
