'use client'

import React from 'react'
import { AnimationProps, motion } from 'framer-motion'

interface LoadingWrapperProps {
  loading?: boolean
  children?: React.ReactNode
  fallback?: React.ReactNode
  animate?: AnimationProps
}

export const LoadingWrapper: React.FC<LoadingWrapperProps> = ({
  loading,
  fallback,
  children,
  animate
}) => {
  const [loaded, setLoaded] = React.useState(!loading)

  React.useEffect(() => {
    if (!loading && !loaded) {
      setLoaded(true)
    }
  }, [loading])

  if (!loaded) {
    return fallback
  } else {
    return <motion.div {...animate}>{children}</motion.div>
  }
}

export default LoadingWrapper
