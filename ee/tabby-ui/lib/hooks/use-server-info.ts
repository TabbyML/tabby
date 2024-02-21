import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'

export const getServerInfo = graphql(/* GraphQL */ `
  query GetServerInfo {
    serverInfo {
      isAdminInitialized
      isEmailConfigured
      isChatEnabled
      allowSelfSignup
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

const useAllowSelfSignup = () => {
  return useServerInfo()?.allowSelfSignup
}

export {
  useIsChatEnabled,
  useIsAdminInitialized,
  useIsEmailConfigured,
  useAllowSelfSignup
}
