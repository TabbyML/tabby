import { createContext, useContext, useMemo } from 'react'
import Link from 'next/link'
import moment from 'moment'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import { ContextSource, ListThreadsQuery } from '@/lib/gql/generates/graphql'
import { Member, useAllMembers } from '@/lib/hooks/use-all-members'
import { contextInfoQuery, listThreadMessages } from '@/lib/tabby/query'
import { getTitleFromMessages } from '@/lib/utils'
import { Card, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import LoadingWrapper from '@/components/loading-wrapper'
import { UserAvatar } from '@/components/user-avatar'

interface ThreadFeedsProps {
  className?: string
}

type ThreadFeedsContextValue = {
  allUsers: Member[] | undefined
  fetchingUsers: boolean
  sources: ContextSource[] | undefined
  fetchingSources: boolean
}

export const ThreadFeedsContext = createContext<ThreadFeedsContextValue>(
  {} as ThreadFeedsContextValue
)

const listThreads = graphql(/* GraphQL */ `
  query ListThreads(
    $ids: [ID!]
    $after: String
    $before: String
    $first: Int
    $last: Int
  ) {
    threads(
      ids: $ids
      after: $after
      before: $before
      first: $first
      last: $last
    ) {
      edges {
        node {
          id
          userId
          createdAt
          updatedAt
        }
        cursor
      }
      pageInfo {
        hasNextPage
        hasPreviousPage
        startCursor
        endCursor
      }
    }
  }
`)

export function ThreadFeeds(props: ThreadFeedsProps) {
  const [{ data, fetching }] = useQuery({
    query: listThreads,
    variables: {
      last: 4
    }
  })

  const [{ data: contextInfoData, fetching: fetchingSources }] = useQuery({
    query: contextInfoQuery
  })

  const threads = useMemo(() => {
    const _threads = data?.threads?.edges
    if (!_threads?.length) return []

    return [..._threads].reverse()
  }, [data?.threads?.edges])

  const [allUsers, fetchingUsers] = useAllMembers()

  return (
    <ThreadFeedsContext.Provider
      value={{
        allUsers,
        fetchingUsers,
        sources: contextInfoData?.contextInfo.sources,
        fetchingSources
      }}
    >
      <Card className="border bg-transparent p-4">
        <CardHeader className="flex flex-row items-center justify-between p-0 pb-2">
          <CardTitle className="text-sm font-normal leading-none tracking-tight">
            Threads
          </CardTitle>
        </CardHeader>
        <LoadingWrapper
          loading={fetching || fetchingUsers}
          fallback={<TheadsSkeleton />}
        >
          <div className="space-y-1 text-sm">
            {threads?.length ? (
              <>
                {threads.map(thread => {
                  return <ThreadItem data={thread} key={thread.node.id} />
                })}
              </>
            ) : (
              <div>No data</div>
            )}
          </div>
        </LoadingWrapper>
      </Card>
    </ThreadFeedsContext.Provider>
  )
}

interface ThreadItemProps {
  data: ListThreadsQuery['threads']['edges'][0]
}
function ThreadItem({ data }: ThreadItemProps) {
  const userId = data.node.userId
  const threadId = data.node.id
  const { sources, allUsers } = useContext(ThreadFeedsContext)

  const [{ data: threadMessagesData, fetching }] = useQuery({
    query: listThreadMessages,
    variables: {
      first: 1,
      threadId: data.node.id
    }
  })

  const threadMessages = threadMessagesData?.threadMessages?.edges

  const title = useMemo(() => {
    if (!threadMessages?.length) return ''
    return getTitleFromMessages(
      sources || [],
      threadMessages[0]['node']['content']
    )
  }, [threadMessages, sources])

  const user = useMemo(() => {
    return allUsers?.find(u => u.id === data.node.userId)
  }, [allUsers, userId])

  return (
    <LoadingWrapper loading={fetching} fallback={<TheadsSkeleton />}>
      <div className="flex items-center gap-1">
        <UserAvatar user={user} className="h-7 w-7 shrink-0 border" />
        <Link
          href={`/search/${title}-${threadId}`}
          className="flex-1 truncate hover:underline"
        >
          {title}
        </Link>
        <div className="w-[130px]">
          {moment(data.node.updatedAt).isBefore(moment().subtract(1, 'days'))
            ? moment(data.node.updatedAt).format('YYYY-MM-DD HH:mm')
            : moment(data.node.updatedAt).fromNow()}
        </div>
      </div>
    </LoadingWrapper>
  )
}

const skeletonList = new Array(4).fill('')

function TheadsSkeleton() {
  return (
    <div className="space-y-1">
      {skeletonList.map((_, idx) => {
        return <ThreadItemSkeleton key={idx} />
      })}
    </div>
  )
}

function ThreadItemSkeleton() {
  return (
    <div className="flex items-center gap-2">
      <Skeleton className="h-6 w-6 rounded-full" />
      <Skeleton className="w-full" />
    </div>
  )
}
