import { useEffect, useState } from 'react'
import { useQuery } from 'urql'

import { DEFAULT_PAGE_SIZE } from '@/lib/constants'
import { QueryVariables } from '@/lib/tabby/gql'
import { listUsers } from '@/lib/tabby/query'

type Member = {
  id: string
  email: string
}

export function useAllMembers() {
  const [queryVariables, setQueryVariables] = useState<
    QueryVariables<typeof listUsers>
  >({ first: DEFAULT_PAGE_SIZE })

  const [list, setList] = useState<Member[]>([])
  const [isAllLoaded, setIsAllLoaded] = useState(false)

  const [{ data, fetching }] = useQuery({
    query: listUsers,
    variables: queryVariables
  })

  useEffect(() => {
    if (isAllLoaded) return
    if (!fetching && data) {
      const members: Member[] = data?.users.edges.map(edge => ({
        id: edge.node.id,
        email: edge.node.email
      }))
      const cursor = data?.users.pageInfo.endCursor || ''
      const hasMore = data?.users.pageInfo.hasNextPage
      const currentList = [...list]

      setList(currentList.concat(members))
      if (hasMore) {
        setQueryVariables({
          first: DEFAULT_PAGE_SIZE,
          after: cursor
        })
      } else {
        setIsAllLoaded(true)
      }
    }
  }, [queryVariables, fetching])

  return [list]
}
