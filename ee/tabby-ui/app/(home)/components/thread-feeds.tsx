import { createContext, useContext, useMemo, useState } from 'react'
import Link from 'next/link'
import slugify from '@sindresorhus/slugify'
import moment from 'moment'
import { useQuery } from 'urql'

import { SLUG_TITLE_MAX_LENGTH } from '@/lib/constants'
import { graphql } from '@/lib/gql/generates'
import { ContextSource, ListThreadsQuery } from '@/lib/gql/generates/graphql'
import { Member, useAllMembers } from '@/lib/hooks/use-all-members'
import { contextInfoQuery, listThreadMessages } from '@/lib/tabby/query'
import { cn, getTitleFromMessages } from '@/lib/utils'
import { CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { Skeleton } from '@/components/ui/skeleton'
import { LoadMoreIndicator } from '@/components/load-more-indicator'
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

export function ThreadFeeds({ className }: ThreadFeedsProps) {
  const [allUsers, fetchingUsers] = useAllMembers()
  const [beforeCursor, setBeforeCursor] = useState<string | undefined>()
  const [{ data, fetching }] = useQuery({
    query: listThreads,
    variables: {
      last: 10,
      before: beforeCursor
    }
  })

  const [{ data: contextInfoData, fetching: fetchingSources }] = useQuery({
    query: contextInfoQuery
  })

  const threads = useMemo(() => {
    const _threads = data?.threads?.edges
    if (!_threads?.length) return []

    return _threads.slice().reverse()
  }, [data?.threads?.edges])

  const pageInfo = data?.threads.pageInfo

  const loadMore = () => {
    if (pageInfo?.startCursor) {
      setBeforeCursor(pageInfo.startCursor)
    }
  }

  return (
    <ThreadFeedsContext.Provider
      value={{
        allUsers,
        fetchingUsers,
        sources: contextInfoData?.contextInfo.sources,
        fetchingSources
      }}
    >
      <div className={cn('mb-4 mt-6', className)}>
        <CardHeader className="flex flex-row items-center justify-between p-0 pb-2">
          <CardTitle className="text-base font-semibold leading-none tracking-tight">
            Threads
          </CardTitle>
        </CardHeader>
        <Separator className="mb-3" />
        <LoadingWrapper
          loading={fetching || fetchingUsers}
          fallback={<TheadsSkeleton className="my-6" />}
        >
          <div className="text-sm">
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
          {!!pageInfo?.hasPreviousPage && (
            <LoadMoreIndicator onLoad={loadMore} isFetching={fetching}>
              <TheadsSkeleton className="mt-6" />
            </LoadMoreIndicator>
          )}
        </LoadingWrapper>
      </div>
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

  const titleSlug = useMemo(() => {
    const titleInSlug = title.slice(0, SLUG_TITLE_MAX_LENGTH)
    return slugify(titleInSlug)
  }, [title])

  const user = useMemo(() => {
    return allUsers?.find(u => u.id === data.node.userId)
  }, [allUsers, userId])

  return (
    <>
      <LoadingWrapper
        loading={fetching}
        fallback={<ThreadItemSkeleton className="py-3" />}
      >
        <Link href={`/search/${titleSlug}-${threadId}`} className="group">
          <div className="py-3">
            <div className="break-anywhere mb-4 line-clamp-2 scroll-m-20 text-base font-semibold tracking-tight group-hover:underline">
              {title}
            </div>
            <div className="flex items-center gap-1">
              <UserAvatar user={user} className="h-10 w-10 shrink-0 border" />
              <div className="space-y-1.5">
                <div className="text-sm">{user?.name || user?.email}</div>
                <div className="text-xs leading-none text-muted-foreground">
                  {moment(data.node.updatedAt).isBefore(
                    moment().subtract(1, 'month')
                  )
                    ? moment(data.node.updatedAt).format('YYYY-MM-DD HH:mm')
                    : moment(data.node.updatedAt).fromNow()}
                </div>
              </div>
            </div>
          </div>
        </Link>
      </LoadingWrapper>
      <Separator className="my-3" />
    </>
  )
}

function TheadsSkeleton({ className }: { className?: string }) {
  return (
    <div className={cn('space-y-6', className)}>
      <ThreadItemSkeleton />
      <Separator className="my-3" />
      <ThreadItemSkeleton />
    </div>
  )
}

function ThreadItemSkeleton({ className }: { className?: string }) {
  return (
    <div className={cn('flex flex-col gap-2', className)}>
      <Skeleton className="w-full" />
      <Skeleton className="w-[60%]" />
      <Skeleton className="mt-4 h-10 w-10 rounded-full" />
    </div>
  )
}
