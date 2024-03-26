import { createStoreWithHydrated } from './utils'

export const useExperimentalFlagStore = createStoreWithHydrated({
  initialState: {
    quickActionBarInCode: false
  },
  storeName: 'exp-flags'
})
