import { useEffect, useState } from 'react'

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
    return new ExperimentFlag(
      this.storageKey,
      this.description,
      this.defaultValue
    )
  }

  defineHook() {
    return (): [
      { value: boolean; description: string; loading: boolean },
      () => void
    ] => {
      const [storageValue, setStorageValue] = useState(this.defaultValue)
      const [loading, setLoading] = useState(true)

      useEffect(() => {
        const value = localStorage.getItem(this.storageKey)
        if (value) {
          setStorageValue(value === 'true')
        }
        setLoading(false)
      }, [])

      const toggleFlag = () => {
        const newStorageValue = !storageValue
        setStorageValue(newStorageValue)
        localStorage.setItem(this.storageKey, String(newStorageValue))
      }

      return [
        {
          value: storageValue,
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
  'Show a quick action popup upon selecting code snippets',
  false
)

export const EXP_enable_code_browser_quick_action_bar =
  enableCodeBrowserQuickActionBarFactory.defineGlobalVar()
export const useEnableCodeBrowserQuickActionBar =
  enableCodeBrowserQuickActionBarFactory.defineHook()
