'use client'

import React from 'react'
import TopBarProgress from 'react-topbar-progress-indicator'

import { useCurrentTheme } from '@/lib/hooks/use-current-theme'
import { useDebounceValue } from '@/lib/hooks/use-debounce'
import { useLatest } from '@/lib/hooks/use-latest'

interface TopbarProgressProviderProps {
  children: React.ReactNode
}

interface TopbarProgressContextValue {
  progress: boolean
  setProgress: (v: boolean) => void
}

const TopbarProgressContext = React.createContext<TopbarProgressContextValue>(
  {} as TopbarProgressContextValue
)

const TopbarProgressProvider: React.FC<TopbarProgressProviderProps> = ({
  children
}) => {
  const { theme } = useCurrentTheme()

  const [progress, setProgress] = React.useState(false)
  const [debouncedProgress] = useDebounceValue(progress, 200, { leading: true })
  const latestProgress = useLatest(progress)

  React.useEffect(() => {
    TopBarProgress.config({
      barColors: {
        0: theme === 'dark' ? '#DC981A' : '#B7942B'
        // 0: '#2563eb'
      }
    })
  }, [])

  const updateProgress = React.useCallback((v: boolean) => {
    if (!!v && v === latestProgress.current) return

    setProgress(v)
  }, [])

  return (
    <TopbarProgressContext.Provider
      value={{ progress: debouncedProgress, setProgress: updateProgress }}
    >
      {debouncedProgress && <TopBarProgress />}
      {children}
    </TopbarProgressContext.Provider>
  )
}

const useTopbarProgress = () => {
  return React.useContext(TopbarProgressContext)
}

export { TopbarProgressProvider, useTopbarProgress }
