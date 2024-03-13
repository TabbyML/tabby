'use client'

import React from 'react'
import { useTheme } from 'next-themes'
import TopBarProgress from 'react-topbar-progress-indicator'

interface TopbarProgressProviderProps {
  children: React.ReactNode
}

interface TopbarProgressContextValue {
  progress: boolean
  setProgress: React.Dispatch<React.SetStateAction<boolean>>
}

const TopbarProgressContext = React.createContext<TopbarProgressContextValue>(
  {} as TopbarProgressContextValue
)

const TopbarProgressProvider: React.FC<TopbarProgressProviderProps> = ({
  children
}) => {
  const [progress, setProgress] = React.useState(false)
  const { theme } = useTheme()
  React.useEffect(() => {
    TopBarProgress.config({
      barColors: {
        0: theme === 'dark' ? '#DC981A' : '#B7942B'
        // 0: '#2563eb'
      }
    })
  }, [])

  return (
    <TopbarProgressContext.Provider value={{ progress, setProgress }}>
      {progress && <TopBarProgress />}
      {children}
    </TopbarProgressContext.Provider>
  )
}

const useTopbarProgress = () => {
  return React.useContext(TopbarProgressContext)
}

export { TopbarProgressProvider, useTopbarProgress }
