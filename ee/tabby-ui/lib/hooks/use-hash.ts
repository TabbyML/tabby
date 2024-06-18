import React from 'react'
import { usePathname } from 'next/navigation'

import { useLatest } from './use-latest'

export function useHash(): [string, (hash: string) => void] {
  const [hash, setHash] = React.useState<string>('')
  const pathname = usePathname()
  const hashRef = useLatest(hash)

  const changeHash = React.useCallback((hash: string) => {
    window.location.hash = hash
  }, [])

  React.useEffect(() => {
    const handleHashChange = () => {
      const newHash = window.location.hash
      if (hashRef.current !== newHash) {
        setHash(newHash)
      }
    }

    handleHashChange()

    window.addEventListener('hashchange', handleHashChange)

    return () => {
      window.removeEventListener('hashchange', handleHashChange)
    }
  }, [pathname])

  return [hash, changeHash]
}
