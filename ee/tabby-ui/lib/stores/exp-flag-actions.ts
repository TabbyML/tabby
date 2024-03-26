import { EXPERIMENTAL_FALG } from '@/lib/constants'

import { useExperimentalFlagStore } from './exp-flag-store'

const set = useExperimentalFlagStore.setState

export const toggleFlag = (flag: EXPERIMENTAL_FALG) => {
  set(state => ({
    [flag]: !state[flag]
  }))
}
