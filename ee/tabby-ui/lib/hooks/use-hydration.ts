import React from 'react'

let hydrating = true

export function useHydrated() {
  let [hydrated, setHydrated] = React.useState(() => !hydrating)

  React.useEffect(function hydrate() {
    hydrating = false
    setHydrated(true)
  }, [])

  return hydrated
}
