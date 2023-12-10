import React from 'react'

import { useHydrated } from './use-hydration'

export const useStore = <T extends { _hasHydrated?: boolean }, F>(
  store: (callback: (state: T) => unknown) => unknown,
  callback: (state: T) => F
) => {
  const hydrated = useHydrated()
  const _hasZustandHydrated = store((state: T) => state?._hasHydrated)
  const result = store(callback) as F
  const [data, setData] = React.useState<F>(
    hydrated && _hasZustandHydrated ? result : (undefined as F)
  )

  React.useEffect(() => {
    setData(result)
  }, [result])

  return data
}
