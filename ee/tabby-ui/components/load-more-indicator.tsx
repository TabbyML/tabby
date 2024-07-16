'use client'

import React from 'react'
import { useInView } from 'react-intersection-observer'

import { cn } from '@/lib/utils'

interface Props {
  onLoading: () => void
  className?: string
}

const LoadMoreIndicatorRender: React.FC<React.PropsWithChildren<Props>> = ({
  onLoading,
  children,
  className
}) => {
  const { ref, inView } = useInView()
  const [isLoaded, setIsLoaded] = React.useState(false)
  React.useEffect(() => {
    if (inView && !isLoaded) {
      setIsLoaded(true)
      onLoading?.()
    }
  }, [inView])

  return (
    <div className={cn('w-full', className)} ref={ref}>
      {children ?? <div>loading...</div>}
    </div>
  )
}

export const LoadMoreIndicator: React.FC<
  React.PropsWithChildren<Props>
> = props => {
  // FIXME add errorboundary
  return <LoadMoreIndicatorRender {...props} />
}
