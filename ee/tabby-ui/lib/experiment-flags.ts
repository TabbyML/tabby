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

// enableDeveloperMode
const enableDeveloperModeFactory = new ExperimentFlagFactory(
  'enable_developer_mode',
  'Developer Mode',
  'Enable the developer mode. This feature involved include the Answer Engine and the Page.',
  false
)
export const EXP_enable_developer_mode =
  enableDeveloperModeFactory.defineGlobalVar()
export const useEnableDeveloperMode = enableDeveloperModeFactory.defineHook()

// enablePage
const enablePageFactory = new ExperimentFlagFactory(
  'enable_page',
  'Page',
  'Enable creating page from scratch.',
  false
)
export const EXP_enable_page = enablePageFactory.defineGlobalVar()
export const useEnablePage = enablePageFactory.defineHook()

// enableSearchPages
const enableSearchPagesFactory = new ExperimentFlagFactory(
  'enable_search_pages',
  'Search Pages',
  'Enable searching pages. This feature allows you to use pages as context in Answer Engine and chat side panel.',
  false
)
export const EXP_enable_search_pages =
  enableSearchPagesFactory.defineGlobalVar()
export const useEnableSearchPages = enableSearchPagesFactory.defineHook()
