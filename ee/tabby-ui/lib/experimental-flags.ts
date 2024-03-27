import useLocalStorage from 'use-local-storage'

class FeatureFlag {
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

class RuntimeFlagFactory {
  private storageKey: string
  private description: string
  private defaultValue: boolean

  constructor(
    storageKey: string,
    description: string,
    defaultValue?: boolean
  ) {
    this.storageKey = `EXP_${storageKey}`
    this.description = description
    this.defaultValue = defaultValue ?? false
  }

  defineGlobalVarAccess () {
    return new FeatureFlag(this.storageKey, this.description, this.defaultValue)
  }

  defineHookAccess () {
    return (): [
      { value: boolean, description: string },
      () => void
    ] => {
      const [storageValue, setStorageValue] = useLocalStorage(
        this.storageKey,
        this.defaultValue
      )
      const toggleFlag = () => {
        setStorageValue(!storageValue)
      }
      return [
        {
          value: storageValue,
          description: this.description
        },
        toggleFlag
      ]
    }
  }
}

const quickActionBarFlag = new RuntimeFlagFactory(
  'enable_code_browser_quick_action_bar',
  'Show a quick action popup upon selecting code snippets',
  false
)

export const EXP_enable_code_browser_quick_action_bar = quickActionBarFlag.defineGlobalVarAccess()
export const useEnableCodeBrowserQuickActionBar = quickActionBarFlag.defineHookAccess()
