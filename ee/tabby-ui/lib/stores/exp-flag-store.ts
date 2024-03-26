import { EXPERIMENTAL_FALG } from '@/lib/constants'
import { createStoreWithHydrated } from './utils'

interface ExperimentalFlagInitialState {
  [EXPERIMENTAL_FALG.QUICK_ACTION_BAR]: boolean
}

export const useExperimentalFlagStore = createStoreWithHydrated<ExperimentalFlagInitialState>({
  initialState: {
    [EXPERIMENTAL_FALG.QUICK_ACTION_BAR]: false
  },
  storeName: 'exp-flags'
})
