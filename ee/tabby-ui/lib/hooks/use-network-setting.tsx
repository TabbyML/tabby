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

const useExternalURL = () => {
  const [{ data }] = useNetworkSetting()
  const networkSetting = data?.networkSetting
  
  return React.useMemo(() => {
    const currentOrigin = getOrigin()
    // Always prefer current origin if available
    if (currentOrigin) return currentOrigin
    // Fallback to network setting only when origin is unavailable
    return networkSetting?.externalUrl || ''
  }, [networkSetting])
}
function getOrigin() {
  if (isClientSide()) {
    return new URL(window.location.href).origin
  }
  return ''
}

export { useNetworkSetting, useExternalURL }
