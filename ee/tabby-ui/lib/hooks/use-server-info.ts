import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'

export const getServerInfo = graphql(/* GraphQL */ `
  query GetServerInfo {
    serverInfo {
      isAdminInitialized
      isEmailConfigured
      isChatEnabled
    }
  }
`)

const useServerInfo = () => {
  const [{ data }] = useQuery({ query: getServerInfo })
  return data?.serverInfo
}

const useIsChatEnabled = () => {
  return useServerInfo()?.isChatEnabled
}

const useIsAdminInitialized = () => {
  return useServerInfo()?.isAdminInitialized
}

const useIsEmailConfigured = () => {
  return useServerInfo()?.isEmailConfigured
}

export { useIsChatEnabled, useIsAdminInitialized, useIsEmailConfigured }
