import React, { useEffect, useState } from 'react'
import { throttle } from 'lodash-es'

function useScrollTop(
  elementRef: React.RefObject<HTMLElement>,
  delay?: number
) {
  const [scrollTop, setScrollTop] = useState(elementRef.current?.scrollTop ?? 0)

  useEffect(() => {
    if (elementRef.current) {
      const handleScroll = throttle(
        () => {
          setScrollTop(elementRef.current?.scrollTop ?? 0)
        },
        delay,
        { leading: true }
      )

      const element = elementRef.current
      element.addEventListener('scroll', handleScroll)

      return () => {
        element.removeEventListener('scroll', handleScroll)
      }
    }
  }, [elementRef, delay])

  return scrollTop
}

export { useScrollTop }
