import { createContext, useContext, useMemo, useState } from 'react'
import Link from 'next/link'
import slugify from '@sindresorhus/slugify'
import { motion, Variants } from 'framer-motion'
import moment from 'moment'
import { useQuery } from 'urql'

import { SLUG_TITLE_MAX_LENGTH } from '@/lib/constants'
import { graphql } from '@/lib/gql/generates'
import { ContextSource, ListThreadsQuery } from '@/lib/gql/generates/graphql'
import { Member, useAllMembers } from '@/lib/hooks/use-all-members'
import { contextInfoQuery, listThreadMessages } from '@/lib/tabby/query'
import { cn, getTitleFromMessages } from '@/lib/utils'
import { IconMessageCircle, IconSpinner } from '@/components/ui/icons'
import { Separator } from '@/components/ui/separator'
import { Skeleton } from '@/components/ui/skeleton'
import { LoadMoreIndicator } from '@/components/load-more-indicator'
import LoadingWrapper from '@/components/loading-wrapper'
import { UserAvatar } from '@/components/user-avatar'

const threadItemVariants: Variants = {
  initial: {
    opacity: 0,
    y: 60
  },
  onscreen: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.5,
      ease: 'easeOut'
    }
  }
}

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

export function ThreadFeeds({ className }: ThreadFeedsProps) {
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

  // const paginations = useMemo(() => {
  //   if (!threads?.length) return []

  //   const paginatedArray: Array<ListThreadsQuery['threads']['edges']> = []
  //   for (let i = 0; i < threads.length; i += PAGE_SIZE) {
  //     paginatedArray.push(threads.slice(i, i + PAGE_SIZE));
  //   }
  //   return paginatedArray
  // }, [threads])

  const threadLen = threads?.length ?? 0

  return (
    <ThreadFeedsContext.Provider
      value={{
        allUsers,
        fetchingUsers,
        sources: contextInfoData?.contextInfo.sources,
        fetchingSources
      }}
    >
      <motion.div
        initial="initial"
        whileInView="onscreen"
        viewport={{
          margin: '0px 0px -140px 0px',
          once: true
        }}
        transition={{
          delay: 1,
          delayChildren: 0.3,
          staggerChildren: 0.05,
          when: 'beforeChildren'
        }}
        style={{ width: '100%', paddingBottom: '1rem' }}
      >
        <div className="mb-3 text-lg font-semibold">Threads</div>
        <Separator className="mb-4" />
        <LoadingWrapper
          loading={fetching || fetchingUsers}
          // showFallback
          fallback={
            <div className="flex justify-center">
              <IconSpinner className="h-8 w-8" />
            </div>
          }
        >
          <div className="space-y-3 text-sm">
            {threads?.length ? (
              <>
                {threads.map((t, idx) => {
                  return (
                    <ThreadItem
                      data={t}
                      key={t.node.id}
                      isLast={idx === threadLen - 1}
                    />
                  )
                })}
              </>
            ) : (
              <div className="text-center text-base">No shared threads</div>
            )}
          </div>
          {!!pageInfo?.hasPreviousPage && (
            <LoadMoreIndicator onLoad={loadMore} isFetching={fetching}>
              <div className="flex justify-center">
                <IconSpinner className="h-8 w-8" />
              </div>
            </LoadMoreIndicator>
          )}
        </LoadingWrapper>
      </motion.div>
    </ThreadFeedsContext.Provider>
  )
}

// function ThreadPagination({ pagination }: { pagination: ListThreadsQuery['threads']['edges'] }) {
//   const [scope, animate] = useAnimate()
//   const isInView = useInView(scope)
//   useEffect(() => {
//     if (isInView) {
//       animate(scope.current, { opacity: 1 })
//     }
//   }, [isInView])

//   return (
//     <motion.div
//       transition={{
//         staggerChildren: 0.1
//       }}
//       ref={scope}
//     >
//       {
//         pagination.map(thread => {
//           return <ThreadItem data={thread} key={thread.node.id} />
//         })
//       }
//     </motion.div >
//   )
// }

interface ThreadItemProps {
  data: ListThreadsQuery['threads']['edges'][0]
  isLast?: boolean
}
function ThreadItem({ data, isLast }: ThreadItemProps) {
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
    <motion.div
      variants={threadItemVariants}
      initial="initial"
      whileInView="onscreen"
      viewport={{
        once: true
      }}
    >
      <div className="flex items-start gap-2">
        <div className="relative mt-2 h-8 w-8 rounded-full bg-[#AAA192] p-2 text-white dark:bg-[#E7E1D3]">
          <IconMessageCircle />
          {!isLast && (
            <div className="absolute left-4 top-10 h-10 w-0.5 bg-border"></div>
          )}
        </div>
        <Link
          href={title ? `/search/${titleSlug}-${threadId}` : 'javascript:void'}
          className="transform-bg group flex-1 overflow-hidden rounded-lg p-2 hover:bg-accent"
        >
          <div className="mb-1.5 flex items-center gap-2">
            <LoadingWrapper
              loading={fetching}
              fallback={
                <div className="w-full py-1.5">
                  <Skeleton className="w-[60%]" />
                </div>
              }
            >
              <div className="break-anywhere truncate text-lg font-semibold">
                {title}
              </div>
            </LoadingWrapper>
          </div>
          <div className="flex items-center gap-1">
            <UserAvatar user={user} className="h-6 w-6 shrink-0 border" />
            <div className="flex items-baseline gap-2.5">
              <div className="text-sm">{user?.name || user?.email}</div>
              <div className="text-xs text-muted-foreground">
                Asked{' '}
                {moment(data.node.createdAt).isBefore(
                  moment().subtract(1, 'month')
                )
                  ? moment(data.node.createdAt).format('YYYY-MM-DD HH:mm')
                  : moment(data.node.createdAt).fromNow()}
              </div>
            </div>
          </div>
        </Link>
      </div>
    </motion.div>
  )
}

function TheadsSkeleton({ className }: { className?: string }) {
  return (
    <div className={cn('space-y-6', className)}>
      <ThreadItemSkeleton />
      <ThreadItemSkeleton />
    </div>
  )
}

function ThreadItemSkeleton() {
  return (
    <div className="p-2">
      <Skeleton className="w-full" />
      <Skeleton className="mt-2.5 h-6 w-6 rounded-full" />
    </div>
  )
}
