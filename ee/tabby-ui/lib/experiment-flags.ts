import useLocalStorage from 'use-local-storage'

class ExperimentFlag {
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

class ExperimentFlagFactory {
  private storageKey: string
  private description: string
  private defaultValue: boolean

  constructor(storageKey: string, description: string, defaultValue?: boolean) {
    this.storageKey = `EXP_${storageKey}`
    this.description = description
    this.defaultValue = defaultValue ?? false
  }

  defineGlobalVar() {
    return new ExperimentFlag(this.storageKey, this.description, this.defaultValue)
  }

  defineHook() {
    return (): [{ value: boolean; description: string }, () => void] => {
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

const enableCodeBrowserQuickActionBarFactory = new ExperimentFlagFactory(
  'enable_code_browser_quick_action_bar',
  'Show a quick action popup upon selecting code snippets',
  false
)

export const EXP_enable_code_browser_quick_action_bar = enableCodeBrowserQuickActionBarFactory.defineGlobalVar()
export const useEnableCodeBrowserQuickActionBar = enableCodeBrowserQuickActionBarFactory.defineHook()
