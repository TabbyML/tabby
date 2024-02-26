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

const useNetworkSettingQuery = () => useQuery({ query: networkSettingQuery })

const useNetworkSetting = () => {
  const [{ data }] = useNetworkSettingQuery()
  return data?.networkSetting
}

const useExternalURL = () => {
  const networkSetting = useNetworkSetting()
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

export { useNetworkSettingQuery, useNetworkSetting, useExternalURL }
