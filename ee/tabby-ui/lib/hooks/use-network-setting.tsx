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

const useNetworkSetting = () => {
  return useQuery({ query: networkSettingQuery })
}

const useExternalURL = () => {
  const [{ data }] = useNetworkSetting()
  const networkSetting = data?.networkSetting
  const externalUrl = React.useMemo(() => {
    return networkSetting?.externalUrl || getOrigin()
  }, [networkSetting])

  return externalUrl
}

function getOrigin() {
  if (isClientSide()) {
    return new URL(window.location.href).origin
  }
  return ''
}

export { useNetworkSetting, useExternalURL }
