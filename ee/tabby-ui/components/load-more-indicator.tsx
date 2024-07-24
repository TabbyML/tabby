'use client'

import React from 'react'
import { useInView } from 'react-intersection-observer'

import { cn } from '@/lib/utils'

interface Props {
  onLoad: () => void
  isFetching: boolean | undefined
  className?: string
}

const LoadMoreIndicatorRender: React.FC<React.PropsWithChildren<Props>> = ({
  onLoad,
  isFetching,
  children,
  className
}) => {
  const { ref, inView } = useInView()
  React.useEffect(() => {
    if (inView && !isFetching) {
      onLoad?.()
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
