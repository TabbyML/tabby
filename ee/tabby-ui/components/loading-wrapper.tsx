'use client'

import React from 'react'

import { useDebounceValue } from '@/lib/hooks/use-debounce'

import { ListSkeleton } from './skeleton'

interface LoadingWrapperProps {
  loading?: boolean
  children?: React.ReactNode
  fallback?: React.ReactNode
  delay?: number
  triggerOnce?: boolean
}

export const LoadingWrapper: React.FC<LoadingWrapperProps> = ({
  triggerOnce = true,
  loading,
  fallback,
  delay,
  children
}) => {
  const [loaded, setLoaded] = React.useState(!loading)
  const [debouncedLoaded] = useDebounceValue(loaded, delay ?? 200)

  React.useEffect(() => {
    if (!triggerOnce) {
      setLoaded(!loading)
    } else if (!loading && !loaded) {
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
