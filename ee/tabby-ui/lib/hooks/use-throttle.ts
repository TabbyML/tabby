import React from 'react'
import { DebouncedFunc, throttle, type ThrottleSettings } from 'lodash-es'

import { useLatest } from './use-latest'
import { useUnmount } from './use-unmount'

type noop = (...args: any[]) => any

interface UseThrottleOptions<T extends noop> extends ThrottleSettings {
  onUnmount?: (
    throttled: DebouncedFunc<(...args: Parameters<T>) => ReturnType<T>>
  ) => void
}

function useThrottleCallback<T extends noop>(
  fn: T,
  wait?: number,
  options?: UseThrottleOptions<T>
) {
  const fnRef = useLatest(fn)
  const throttled = React.useMemo(
    () =>
      throttle(
        (...args: Parameters<T>): ReturnType<T> => {
          return fnRef.current(...args)
        },
        wait,
        options
      ),
    []
  )

  useUnmount(() => {
    options?.onUnmount?.(throttled)
    throttled.cancel()
  })

  return {
    run: throttled,
    cancel: throttled.cancel,
    flush: throttled.flush
  }
}

function useThrottleValue<T>(
  value: T,
  wait?: number,
  options?: ThrottleSettings
): [T, React.Dispatch<React.SetStateAction<T>>] {
  const [throttledValue, setThrottledValue] = React.useState(value)

  const { run } = useThrottleCallback(
    () => {
      setThrottledValue(value)
    },
    wait,
    options
  )

  React.useEffect(() => {
    run()
  }, [value])

  return [throttledValue, setThrottledValue]
}

export { useThrottleCallback, useThrottleValue }
