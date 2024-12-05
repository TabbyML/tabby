import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export interface UserPreferences {
  isSidebarExpanded: boolean
  _hasHydrated: boolean
  setHasHydrated: (v: boolean) => void
}

const initialState: Omit<UserPreferences, 'setHasHydrated'> = {
  isSidebarExpanded: true,
  _hasHydrated: false
}

export const useUserPreferencesStore = create<UserPreferences>()(
  persist(
    set => ({
      ...initialState,
      setHasHydrated: (v: boolean) => set(() => ({ _hasHydrated: v }))
    }),
    {
      name: 'user-preferences-storage',
      version: 0,
      onRehydrateStorage: state => {
        return () => {
          state.setHasHydrated(true)
        }
      }
    }
  )
)

const set = useUserPreferencesStore.setState

export const toggleSidebar = (expanded: boolean) => {
  return set(() => ({ isSidebarExpanded: expanded }))
}
