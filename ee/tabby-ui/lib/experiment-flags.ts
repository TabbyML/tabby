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

const enableDeveloperModeFactory = new ExperimentFlagFactory(
  'enable_developer_mode',
  'Developer Mode',
  'Enable the developer mode. The features involved include the Answer Engine.',
  false
)

export const EXP_enable_developer_mode =
  enableDeveloperModeFactory.defineGlobalVar()
export const useEnableDeveloperMode = enableDeveloperModeFactory.defineHook()

const enablePageFactory = new ExperimentFlagFactory(
  'enable_page',
  'Page',
  'Enable Page. This feature allows you to convert threads to page.',
  false
)

export const EXP_enable_page = enablePageFactory.defineGlobalVar()
export const useEnablePage = enablePageFactory.defineHook()
