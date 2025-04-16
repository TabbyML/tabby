'use client'

import React, { useEffect, useState } from 'react'
import { IntersectionOptions, useInView } from 'react-intersection-observer'

import { useDebounceCallback } from '@/lib/hooks/use-debounce'
import { cn } from '@/lib/utils'

interface Props {
  onLoad: () => void
  isFetching: boolean | undefined
  itemCount: number
  className?: string
  intersectionOptions?: IntersectionOptions
}

const LoadMoreIndicatorRender: React.FC<React.PropsWithChildren<Props>> = ({
  onLoad,
  isFetching,
  itemCount,
  children,
  className,
  intersectionOptions
}) => {
  const { ref, inView } = useInView(intersectionOptions)
  const [hasTriggered, setHasTriggered] = useState(false)
  const triggerLoad = useDebounceCallback((v: boolean) => {
    setHasTriggered(v)
  }, 150)

  useEffect(() => {
    if (inView && !hasTriggered) {
      setHasTriggered(true)
      onLoad()
    }
  }, [inView, hasTriggered, itemCount])

  useEffect(() => {
    if (!isFetching) {
      triggerLoad.run(false)
    }
  }, [isFetching])

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
