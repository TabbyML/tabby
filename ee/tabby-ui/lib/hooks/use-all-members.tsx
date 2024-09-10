import { useMemo } from 'react'
import { CombinedError, useQuery } from 'urql'

import { listSecuredUsers } from '@/lib/tabby/query'

import { ListUsersQuery } from '../gql/generates/graphql'

export type Member = ListUsersQuery['users']['edges'][0]['node']

export function useAllMembers(): [
  Member[],
  boolean,
  CombinedError | undefined
] {
  const [{ data, fetching, error }] = useQuery({
    query: listSecuredUsers
  })

  const allUsers = useMemo(() => {
    return data?.users.edges.map(edge => edge.node) ?? []
  }, [data?.users])

  return [allUsers, fetching, error]
}
