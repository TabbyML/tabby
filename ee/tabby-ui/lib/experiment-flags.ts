import { useEffect, useState } from 'react'

const useLocalStorageForExperimentFlag = (
  storageKey: string,
  defaultValue: boolean
): [boolean, (value: boolean) => void, boolean] => {
  const [storageValue, setStorageValue] = useState(defaultValue)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const value = localStorage.getItem(storageKey)
    if (value) {
      setStorageValue(JSON.parse(value))
    }
    setLoading(false)
  }, [])

  const upateStorageValue = (newValue: boolean) => {
    setStorageValue(newValue)
    localStorage.setItem(storageKey, JSON.stringify(newValue))
  }

  return [storageValue, upateStorageValue, loading]
}

class ExperimentFlag {
  constructor(
    private storageKey: string,
    readonly title: string,
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
  private title: string
  private description: string
  private defaultValue: boolean

  constructor(
    storageKey: string,
    title: string,
    description: string,
    defaultValue?: boolean
  ) {
    this.storageKey = `EXP_${storageKey}`
    this.title = title
    this.description = description
    this.defaultValue = defaultValue ?? false
  }

  defineGlobalVar() {
    return new ExperimentFlag(
      this.storageKey,
      this.title,
      this.description,
      this.defaultValue
    )
  }

  defineHook() {
    return (): [
      { value: boolean; title: string; description: string; loading: boolean },
      () => void
    ] => {
      const [storageValue, setStorageValue, loading] =
        useLocalStorageForExperimentFlag(this.storageKey, this.defaultValue)

      const toggleFlag = () => {
        setStorageValue(!storageValue)
      }

      return [
        {
          value: storageValue,
          title: this.title,
          description: this.description,
          loading
        },
        toggleFlag
      ]
    }
  }
}

const enableCodeBrowserQuickActionBarFactory = new ExperimentFlagFactory(
  'enable_code_browser_quick_action_bar',
  'Quick Action Bar',
  'Enable Quick Action Bar to display a convenient toolbar when you select code, offering options to explain the code, add unit tests, and more.',
  true
)

export const EXP_enable_code_browser_quick_action_bar =
  enableCodeBrowserQuickActionBarFactory.defineGlobalVar()
export const useEnableCodeBrowserQuickActionBar =
  enableCodeBrowserQuickActionBarFactory.defineHook()
