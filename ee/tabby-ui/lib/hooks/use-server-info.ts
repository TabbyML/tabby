import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'

export const getServerInfo = graphql(/* GraphQL */ `
  query GetServerInfo {
    serverInfo {
      isAdminInitialized
      isEmailConfigured
      isChatEnabled
      allowSelfSignup
      isDemoMode
      disablePasswordLogin
    }
  }
`)

const useServerInfo = () => {
  const [{ data }] = useQuery({ query: getServerInfo })
  return data?.serverInfo
}

const useIsFetchingServerInfo = () => {
  const [{ fetching }] = useQuery({ query: getServerInfo })
  return fetching
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

const useIsDemoMode = () => {
  return useServerInfo()?.isDemoMode
}

const useIsDisablePasswordLogin = () => {
  return useServerInfo()?.disablePasswordLogin
}

export {
  useServerInfo,
  useIsChatEnabled,
  useIsAdminInitialized,
  useIsEmailConfigured,
  useAllowSelfSignup,
  useIsDemoMode,
  useIsFetchingServerInfo,
  useIsDisablePasswordLogin
}
