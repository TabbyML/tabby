'use client'

import React from 'react';

interface LoadingWrapperProps {
  loading?: boolean
  children?: React.ReactNode
  fallback?: React.ReactNode
}

export const LoadingWrapper: React.FC<LoadingWrapperProps> = ({ loading, fallback, children }) => {

  const [loaded, setLoaded] = React.useState(!loading)

  React.useEffect(() => {
    if (!loading && !loaded) {
      setLoaded(true)
    }
  }, [loading])

  if (!loaded) {
    return fallback; 
  } else {
    return children;
  }
};

export default LoadingWrapper;