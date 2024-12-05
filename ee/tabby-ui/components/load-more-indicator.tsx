'use client'

import React from 'react'
import { IntersectionOptions, useInView } from 'react-intersection-observer'

import { cn } from '@/lib/utils'

interface Props {
  onLoad: () => void
  isFetching: boolean | undefined
  className?: string
  intersectionOptions?: IntersectionOptions
}

const LoadMoreIndicatorRender: React.FC<React.PropsWithChildren<Props>> = ({
  onLoad,
  isFetching,
  children,
  className,
  intersectionOptions
}) => {
  const { ref, inView } = useInView(intersectionOptions)
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
