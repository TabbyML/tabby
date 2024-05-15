import * as React from 'react'
import { throttle } from 'lodash-es'

export function useAtBottom(offset = 0, container?: HTMLDivElement) {
  const [isAtBottom, setIsAtBottom] = React.useState(false)

  React.useEffect(() => {
    if (container) return

    const handleScroll = throttle(
      () => {
        setIsAtBottom(
          window.innerHeight + window.scrollY >=
            document.body.offsetHeight - offset
        )
      },
      100,
      { leading: true }
    )

    window.addEventListener('scroll', handleScroll, { passive: true })
    window.addEventListener('resize', handleScroll, { passive: true })
    handleScroll()

    return () => {
      window.removeEventListener('scroll', handleScroll)
      window.removeEventListener('resize', handleScroll)
    }
  }, [offset, container])

  React.useEffect(() => {
    if (!container) return

    const handleScroll = () => {
      const { scrollTop, clientHeight, scrollHeight } = container
      console.log(
        'scrollTop',
        scrollTop,
        'clientHeight',
        clientHeight,
        'scrollHeight',
        scrollHeight
      )
      setIsAtBottom(scrollTop + clientHeight >= scrollHeight)
    }

    container.addEventListener('scroll', handleScroll, { passive: true })
    handleScroll()

    return () => {
      container.removeEventListener('scroll', handleScroll)
    }
  }, [offset, container])

  return isAtBottom
}
