import useLocalStorage from 'use-local-storage'

type DefineExperimentalFlagResponse = [
  ExperimentalFlag,
  () => {
    flag: {
      value: boolean
      description: string
    }
    toggleFlag: () => void
  }
]

class ExperimentalFlag {
  constructor(
    private storageKey: string,
    readonly description: string,
    readonly defaultValue: boolean
  ) {}

  get value() {
    if (typeof window !== 'undefined' && typeof localStorage !== 'undefined') {
      const storageValue = localStorage.getItem(this.storageKey)
      if (storageValue) {
        return storageValue === 'true'
      }
    }

    return this.defaultValue
  }
}

const defineExperimentalFlagHook = (
  storageKey: string,
  flag: ExperimentalFlag
) => {
  return () => {
    const [storageValue, setStorageValue] = useLocalStorage(
      storageKey,
      flag.defaultValue
    )
    const toggleFlag = () => {
      setStorageValue(!storageValue)
    }
    return {
      flag: {
        value: storageValue,
        description: flag.description
      },
      toggleFlag
    }
  }
}

const defineExperimentalFlag = (
  storageKey: string,
  description: string,
  defaultValue?: boolean
): DefineExperimentalFlagResponse => {
  const flagDefaultValue = defaultValue ?? false
  const flag = new ExperimentalFlag(storageKey, description, flagDefaultValue)
  const useFlagHook = defineExperimentalFlagHook(storageKey, flag)
  return [flag, useFlagHook]
}

const [
  EXP_enable_code_browser_quick_action_bar,
  useEnableCodeBrowserQuickActionBar
] = defineExperimentalFlag(
  'enable_code_browser_quick_action_bar',
  'Show a quick action popup upon selecting code snippets',
  false
)

export {
  EXP_enable_code_browser_quick_action_bar,
  useEnableCodeBrowserQuickActionBar
}
