import { create } from 'zustand'
import { persist, createJSONStorage  } from 'zustand/middleware'

export const useExperimentalFlag = create<{
  quickActionBarInCode: boolean;
  toggleQuickActionBarInCode: () => void
}>()(
  persist(
    (set, get) => ({
      quickActionBarInCode: false,
      toggleQuickActionBarInCode: () => set({ quickActionBarInCode: !get().quickActionBarInCode }),
    }),
    {
      name: 'exp-flag',
      storage: createJSONStorage(() => localStorage)
    },
  ),
)
