import { create } from 'zustand'
import { persist } from 'zustand/middleware'

const excludeFromState = ['_hasHydrated', 'setHasHydrated', 'activeChatId']

export interface ChatState {
  homePage: number | undefined
  _hasHydrated: boolean
  setHasHydrated: (state: boolean) => void
}

const initialState: Omit<ChatState, 'setHasHydrated' | 'deleteChat'> = {
  _hasHydrated: false,
  homePage: undefined
}

export const useScrollStore = create<ChatState>()(
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
      name: 'tabby-scroll-storage',
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

const set = useScrollStore.setState

export const setHomeScrollPosition = (scrollTop: number) => {
  return set(state => ({
    ...state,
    homePage: scrollTop
  }))
}

export const clearHomeScrollPosition = () => {
  return set(state => ({
    ...state,
    homePage: undefined
  }))
}
