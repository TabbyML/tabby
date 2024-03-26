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
export const createStoreWithHydrated = <T extends Record<string, any>>({
  initialState,
  storeName,
  excludeFromState
}: {
  initialState: T;
  storeName: string;
  excludeFromState?: string[]
}) => {
  
  return create<T & {
    _hasHydrated: boolean;
    setHasHydrated: (state: boolean) => void;
  }>()(
    persist(
      set => {
        return {
          ...initialState,
          _hasHydrated: false as boolean,
          setHasHydrated: (newState: boolean) => {
            set((state) => ({
              ...state,
              _hasHydrated: newState
            }));
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
