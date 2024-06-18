import React from 'react'
import Router from 'next/router'

import { useLatest } from './use-latest'

export function useHash(): [string, (hash: string) => void] {
  const [hash, setHash] = React.useState<string>('')
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

    Router.events.on('hashChangeComplete', handleHashChange)

    return () => {
      Router.events.off('hashChangeComplete', handleHashChange)
    }
  }, [])

  return [hash, changeHash]
}
