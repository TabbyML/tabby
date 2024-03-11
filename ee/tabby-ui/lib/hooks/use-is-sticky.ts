import { useEffect, useState } from 'react'

const useIsSticky = (ref: React.RefObject<HTMLElement>) => {
  const [isSticky, setIsSticky] = useState(false)

  useEffect(() => {
    const cachedRef = ref.current
    const observer = new IntersectionObserver(
      ([e]) => {
        setIsSticky(e.intersectionRatio < 1)
      },
      { threshold: [1] }
    )

    if (cachedRef) {
      observer.observe(cachedRef)
    }

    return () => {
      if (cachedRef) {
        observer.unobserve(cachedRef)
      }
    }
  }, [ref.current])

  return isSticky
}

export { useIsSticky }
