import { createStoreWithHydrated } from './utils'

interface InitialState {
  quickActionBarInCode: boolean
}

export const useExperimentalFlagStore = createStoreWithHydrated<InitialState>({
  initialState: {
    quickActionBarInCode: false
  },
  storeName: 'exp-flags'
})
