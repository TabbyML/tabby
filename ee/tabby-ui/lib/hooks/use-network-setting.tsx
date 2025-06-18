import React from 'react'
import { useQuery } from 'urql'

import { graphql } from '../gql/generates'
import { isClientSide } from '../utils'

const networkSettingQuery = graphql(/* GraphQL */ `
  query NetworkSetting {
    networkSetting {
      externalUrl
    }
  }
`)

const useNetworkSetting = (options?: any) => {
  return useQuery({ query: networkSettingQuery, ...options })
}

const useExternalURL = (): string | null => {
  const [{ data }] = useNetworkSetting()
  const networkSetting = data?.networkSetting
  const [origin, setOrigin] = React.useState<string | null>(null)

  React.useEffect(() => {
    // Client-side only
    if (isClientSide()) {
      const updateOrigin = () => {
        setOrigin(new URL(window.location.href).origin)
      }
      
      // Initialize
      updateOrigin()
      
      // Set up listeners for potential origin changes
      window.addEventListener('popstate', updateOrigin)
      window.addEventListener('hashchange', updateOrigin)
      
      return () => {
        window.removeEventListener('popstate', updateOrigin)
        window.removeEventListener('hashchange', updateOrigin)
      }
    }
  }, [])

  // Return priority: current origin > network setting > null
  return origin || networkSetting?.externalUrl || null
}

export { useNetworkSetting, useExternalURL } 