import { useExperimentalFlagStore } from './exp-flag-store'

const set = useExperimentalFlagStore.setState

export const toggleQuickActionBar = () => {
  set(state => ({
    quickActionBarInCode: !state.quickActionBarInCode
  }))
}
