'use client'

import React from 'react'

import { useDebounceValue } from '@/lib/hooks/use-debounce'

import { ListSkeleton } from './skeleton'

interface LoadingWrapperProps {
  loading?: boolean
  children?: React.ReactNode
  fallback?: React.ReactNode
  delay?: number
  // todo remove
  showFallback?: boolean
}

export const LoadingWrapper: React.FC<LoadingWrapperProps> = ({
  loading,
  fallback,
  delay,
  children,
  showFallback
}) => {
  const [loaded, setLoaded] = React.useState(!loading)
  const [debouncedLoaded] = useDebounceValue(loaded, delay ?? 200)

  React.useEffect(() => {
    if (!loading && !loaded) {
      setLoaded(true)
    }
  }, [loading])

  if (!debouncedLoaded) {
    return fallback ? fallback : <ListSkeleton />
  } else {
    return showFallback ? fallback : children
  }
}

export default LoadingWrapper
