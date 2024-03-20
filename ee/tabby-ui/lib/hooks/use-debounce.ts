import React from 'react'
import { debounce, type DebounceSettings } from 'lodash-es'

import { useLatest } from './use-latest'
import { useUnmount } from './use-unmount'

type noop = (...args: any[]) => any

// interface UseDebounceOptions<T = any> extends DebounceSettings {
//   onFire?: (value: T) => void
// }

function useDebounceCallback<T extends noop>(
  fn: T,
  wait?: number,
  options?: DebounceSettings
) {
  const fnRef = useLatest(fn)
  const debounced = React.useMemo(
    () =>
      debounce(
        (...args: Parameters<T>): ReturnType<T> => {
          return fnRef.current(...args)
        },
        wait,
        options
      ),
    []
  )

  useUnmount(() => debounced.cancel())

  return {
    run: debounced,
    cancel: debounced.cancel,
    flush: debounced.flush
  }
}

function useDebounceValue<T>(
  value: T,
  wait?: number,
  options?: DebounceSettings
): [T, React.Dispatch<React.SetStateAction<T>>] {
  const [debouncedValue, setDebouncedValue] = React.useState(value)

  const { run } = useDebounceCallback(
    () => {
      setDebouncedValue(value)
    },
    wait,
    options
  )

  React.useEffect(() => {
    run()
  }, [value])

  return [debouncedValue, setDebouncedValue]
}

export { useDebounceCallback, useDebounceValue }
