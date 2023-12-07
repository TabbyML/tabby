import * as React from 'react'

export default function useInterval(
  callback: () => void,
  delay: number | null
): React.MutableRefObject<number | null> {
  const savedCallback = React.useRef(callback)
  const intervalRef = React.useRef<number | null>(null)

  // Remember the latest callback if it changes.
  React.useEffect(() => {
    savedCallback.current = callback
  }, [callback])

  // Set up the interval.
  React.useEffect(() => {
    const tick = () => savedCallback.current()

    if (typeof delay === 'number') {
      intervalRef.current = window.setInterval(tick, delay * 60 * 1000)
    }

    return () => {
      if (intervalRef.current) {
        window.clearTimeout(intervalRef.current)
      }
    }
  }, [delay])

  return intervalRef
}
