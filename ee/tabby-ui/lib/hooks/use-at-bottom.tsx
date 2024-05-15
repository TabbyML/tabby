import * as React from 'react'
import { throttle } from 'lodash-es'

export function useAtBottom(offset = 0) {
  const [isAtBottom, setIsAtBottom] = React.useState(false)

  React.useEffect(() => {
    const handleScroll = throttle(
      () => {
        setIsAtBottom(
          window.innerHeight + window.scrollY >=
            document.body.offsetHeight - offset
        )
      },
      300,
      { leading: true }
    )

    window.addEventListener('scroll', handleScroll, { passive: true })
    window.addEventListener('resize', handleScroll, { passive: true })
    handleScroll()

    return () => {
      window.removeEventListener('scroll', handleScroll)
      window.removeEventListener('resize', handleScroll)
    }
  }, [offset])

  return isAtBottom
}
