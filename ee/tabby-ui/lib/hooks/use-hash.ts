import React from 'react'
import { useParams } from 'next/navigation'

import { useLatest } from './use-latest'

export function useHash(): [string, (hash: string) => void] {
  const param = useParams()
  const [hash, setHash] = React.useState<string>('')
  const hashRef = useLatest(hash)

  const changeHash = React.useCallback((hash: string) => {
    window.location.hash = hash
  }, [])

  const handleHashChange = () => {
    const newHash = window.location.hash
    if (hashRef.current !== newHash) {
      setHash(newHash)
    }
  }

  React.useEffect(() => {
    window.addEventListener('hashchange', handleHashChange)
    return () => {
      window.removeEventListener('hashchange', handleHashChange)
    }
  }, [])

  // Handle hash changes from next/navigation
  React.useEffect(handleHashChange, [param])

  return [hash, changeHash]
}
