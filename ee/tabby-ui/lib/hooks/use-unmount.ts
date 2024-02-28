import React from 'react'

import { useLatest } from './use-latest'

const useUnmount = (fn: () => void) => {
  const fnRef = useLatest(fn)

  React.useEffect(
    () => () => {
      fnRef.current()
    },
    []
  )
}

export { useUnmount }
