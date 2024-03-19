'use client'

import React from 'react'
import { ListSkeleton } from './skeleton'
import { useDebounceValue } from '@/lib/hooks/use-debounce'

interface LoadingWrapperProps {
  loading?: boolean
  children?: React.ReactNode
  fallback?: React.ReactNode
  delay?: number
}

export const LoadingWrapper: React.FC<LoadingWrapperProps> = ({
  loading,
  fallback,
  delay,
  children
}) => {
  const [loaded, setLoaded] = React.useState(!loading)
  const [debouncedLoaded] = useDebounceValue(loaded, delay)

  React.useEffect(() => {
    if (!loading && !loaded) {
      setLoaded(true)
    }
  }, [loading])

  if (!debouncedLoaded) {
    return fallback ? fallback : <ListSkeleton />
  } else {
    return children
  }
}

export default LoadingWrapper
