import { create } from 'zustand'
import { persist } from 'zustand/middleware'

import { Chat } from '@/lib/types'

export const getChatById = (
  chats: Chat[] | undefined,
  chatId: string | undefined
): Chat | undefined => {
  if (!Array.isArray(chats) || !chatId) return undefined
  return chats.find(c => c.id === chatId)
}

// @reference:
// https://github.com/pmndrs/zustand/blob/main/docs/integrations/persisting-store-data.md#hydration-and-asynchronous-storages
export const createStoreWithHydrated = ({
  initialState,
  storeName,
  excludeFromState
}: {
  initialState: Record<string, any>
  storeName: string
  excludeFromState?: string[]
}) => {
  return create<typeof initialState>()(
    persist(
      set => {
        return {
          ...initialState,
          _hasHydrated: false,
          setHasHydrated: (state: boolean) => {
            set({
              _hasHydrated: state
            })
          }
        }
      },
      {
        name: storeName,
        partialize: state => {
          const defaultExcludeFromState = ['_hasHydrated', 'setHasHydrated']
          const myExcludeFromState = excludeFromState
            ? defaultExcludeFromState.concat(excludeFromState)
            : defaultExcludeFromState
          return Object.fromEntries(
            Object.entries(state).filter(
              ([key]) => !myExcludeFromState.includes(key)
            )
          )
        },
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
}
