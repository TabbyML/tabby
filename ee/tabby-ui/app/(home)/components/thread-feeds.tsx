import { createContext, forwardRef, useContext, useMemo, useState } from 'react'
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
import { IconMessagesSquare, IconSpinner } from '@/components/ui/icons'
import { Separator } from '@/components/ui/separator'
import { Skeleton } from '@/components/ui/skeleton'
import { LoadMoreIndicator } from '@/components/load-more-indicator'
import LoadingWrapper from '@/components/loading-wrapper'
import { UserAvatar } from '@/components/user-avatar'

interface ThreadFeedsProps {
  className?: string
  onNavigateToThread: () => void
}

type ThreadFeedsContextValue = {
  allUsers: Member[] | undefined
  fetchingUsers: boolean
  sources: ContextSource[] | undefined
  fetchingSources: boolean
  onNavigateToThread: () => void
}

export const ThreadFeedsContext = createContext<ThreadFeedsContextValue>(
  {} as ThreadFeedsContextValue
)

const PAGE_SIZE = 10

const listThreads = graphql(/* GraphQL */ `
  query ListThreads(
    $ids: [ID!]
    $isEphemeral: Boolean
    $after: String
    $before: String
    $first: Int
    $last: Int
  ) {
    threads(
      ids: $ids
      isEphemeral: $isEphemeral
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

export const ThreadFeeds = forwardRef<HTMLDivElement, ThreadFeedsProps>(
  ({ onNavigateToThread, className }, ref) => {
    const [allUsers, fetchingUsers] = useAllMembers()
    const [beforeCursor, setBeforeCursor] = useState<string | undefined>()
    const [{ data, fetching }] = useQuery({
      query: listThreads,
      variables: {
        last: PAGE_SIZE,
        before: beforeCursor,
        isEphemeral: false
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
          fetchingSources,
          onNavigateToThread
        }}
      >
        <div className={cn('w-full', className)} ref={ref}>
          <div className="mb-2.5 w-full text-lg font-semibold">
            Recent Activities
          </div>
          <Separator className="mb-4 w-full" />
          <div className="w-full pb-4">
            <LoadingWrapper
              loading={fetching || fetchingUsers}
              fallback={
                <div className="flex justify-center">
                  <IconSpinner className="h-8 w-8" />
                </div>
              }
            >
              <div className="flex flex-col gap-3 text-sm">
                {threads?.length ? (
                  <>
                    {threads.map((t, idx) => {
                      return <ThreadItem data={t} key={t.node.id} />
                    })}
                  </>
                ) : (
                  <div className="text-center text-base">No shared threads</div>
                )}
              </div>
              {!!pageInfo?.hasPreviousPage && (
                <LoadMoreIndicator
                  onLoad={loadMore}
                  isFetching={fetching}
                  intersectionOptions={{ rootMargin: '0px 0px 200px 0px' }}
                >
                  <div className="mt-8 flex justify-center">
                    <IconSpinner className="h-8 w-8" />
                  </div>
                </LoadMoreIndicator>
              )}
            </LoadingWrapper>
          </div>
        </div>
      </ThreadFeedsContext.Provider>
    )
  }
)
ThreadFeeds.displayName = 'ThreadFeeds'

interface ThreadItemProps {
  data: ListThreadsQuery['threads']['edges'][0]
}
function ThreadItem({ data }: ThreadItemProps) {
  const userId = data.node.userId
  const threadId = data.node.id
  const { sources, allUsers, onNavigateToThread } =
    useContext(ThreadFeedsContext)

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
    <Link
      href={title ? `/search/${titleSlug}-${threadId}` : 'javascript:void'}
      onClick={onNavigateToThread}
    >
      <div className="transform-bg group flex-1 overflow-hidden rounded-lg px-3 py-2 hover:bg-accent">
        <div className="mb-1.5 flex items-center gap-2">
          <IconMessagesSquare className="shrink-0" />
          <LoadingWrapper
            loading={fetching}
            fallback={
              <div className="w-full py-1.5">
                <Skeleton className="w-[60%]" />
              </div>
            }
          >
            <div className="break-anywhere truncate text-lg font-medium">
              {title}
            </div>
          </LoadingWrapper>
        </div>
        <div className="flex items-center gap-2">
          <UserAvatar user={user} className="mr-0.5 h-4 w-4 shrink-0" />
          <div className="flex items-baseline gap-0.5">
            <div className="text-sm">{user?.name || user?.email}</div>
            <span className="text-muted-foreground">{'·'}</span>
            <div className="text-xs text-muted-foreground">
              {formatCreatedAt(data.node.createdAt, 'Asked')}
            </div>
          </div>
        </div>
      </div>
    </Link>
  )
}

function formatCreatedAt(time: string, prefix: string) {
  const targetTime = moment(time)

  if (targetTime.isBefore(moment().subtract(1, 'year'))) {
    const timeText = targetTime.format('MMM D, YYYY')
    return `${prefix} on ${timeText}`
  }

  if (targetTime.isBefore(moment().subtract(1, 'month'))) {
    const timeText = targetTime.format('MMM D')
    return `${prefix} on ${timeText}`
  }

  return `${prefix} ${targetTime.fromNow()}`
}
