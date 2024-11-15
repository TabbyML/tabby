import { createContext, useContext, useEffect, useMemo, useState } from 'react'
import Link from 'next/link'
import slugify from '@sindresorhus/slugify'
import { motion } from 'framer-motion'
import moment from 'moment'
import { useQuery } from 'urql'

import { SLUG_TITLE_MAX_LENGTH } from '@/lib/constants'
import { MARKDOWN_SOURCE_REGEX } from '@/lib/constants/regex'
import { graphql } from '@/lib/gql/generates'
import { ContextSource, ListThreadsQuery } from '@/lib/gql/generates/graphql'
import { Member, useAllMembers } from '@/lib/hooks/use-all-members'
import {
  resetThreadsPageNo,
  useAnswerEngineStore
} from '@/lib/stores/answer-engine-store'
import { contextInfoQuery, listThreadMessages } from '@/lib/tabby/query'
import { cn, getTitleFromMessages } from '@/lib/utils'
import { IconFiles, IconSpinner } from '@/components/ui/icons'
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationNext,
  PaginationPrevious
} from '@/components/ui/pagination'
import { Separator } from '@/components/ui/separator'
import { Skeleton } from '@/components/ui/skeleton'
import LoadingWrapper from '@/components/loading-wrapper'
import { Mention } from '@/components/mention-tag'
import { UserAvatar } from '@/components/user-avatar'

import { AnimationWrapper } from './animation-wrapper'

interface ThreadFeedsProps {
  className?: string
  onNavigateToThread: (params: { pageNo: number }) => void
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

const PAGE_SIZE = 8

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

export function ThreadFeeds({
  className,
  onNavigateToThread
}: ThreadFeedsProps) {
  const storedPageNo = useAnswerEngineStore(state => state.threadsPageNo)
  const [allUsers, fetchingUsers] = useAllMembers()
  const [beforeCursor, setBeforeCursor] = useState<string | undefined>()
  const [page, setPage] = useState(storedPageNo)
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
  const pageInfo = data?.threads.pageInfo
  const threadCount = data?.threads?.edges?.length
  const pageCount = Math.ceil((threadCount || 0) / PAGE_SIZE)
  const showPagination =
    pageCount > 1 || (pageCount === 1 && pageInfo?.hasPreviousPage)

  // threads for current page
  const threads = useMemo(() => {
    const _threads = data?.threads?.edges
    if (!_threads?.length) return []

    return _threads
      .slice()
      .reverse()
      .slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)
  }, [data?.threads?.edges, page])

  const loadMore = () => {
    if (pageInfo?.startCursor) {
      setBeforeCursor(pageInfo.startCursor)
    }
  }

  const handleNavigateToThread = () => {
    onNavigateToThread({ pageNo: page })
  }

  useEffect(() => {
    // reset pageNo store after it has been used
    resetThreadsPageNo()
  }, [])

  return (
    <ThreadFeedsContext.Provider
      value={{
        allUsers,
        fetchingUsers,
        sources: contextInfoData?.contextInfo.sources,
        fetchingSources,
        onNavigateToThread: handleNavigateToThread
      }}
    >
      <div className={cn('w-full', className)}>
        <AnimationWrapper style={{ width: '100%' }} viewport={{ once: true }}>
          <LoadingWrapper
            loading={fetching || fetchingUsers}
            fallback={
              <div className="flex justify-center">
                <IconSpinner className="h-8 w-8" />
              </div>
            }
          >
            <div className="mb-2.5 w-full text-lg font-semibold">
              Recent Activities
            </div>
            <Separator className="mb-4 w-full" />
            <motion.div
              initial="initial"
              whileInView="onscreen"
              viewport={{
                amount: 'some',
                // margin: '0px 0px -140px 0px',
                once: true
              }}
              style={{ width: '100%' }}
            >
              <div className="relative flex min-h-[40.25rem] flex-col gap-3 text-sm">
                {threads.map(t => {
                  return <ThreadItem data={t} key={t.node.id} />
                })}
                {!threads.length && fetching && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <IconSpinner className="h-8 w-8" />
                  </div>
                )}
              </div>
              {showPagination && (
                <Pagination className={cn('flex items-center justify-end')}>
                  <PaginationContent>
                    <span className="flex w-[64px] items-center justify-start text-sm font-medium">
                      Page {page}
                    </span>
                    <PaginationItem>
                      <PaginationPrevious
                        disabled={page === 1}
                        onClick={() => {
                          if (page === 1) return

                          const _page = page - 1
                          setPage(_page)
                        }}
                      />
                    </PaginationItem>
                    <PaginationItem>
                      <PaginationNext
                        disabled={
                          page === pageCount && !pageInfo?.hasPreviousPage
                        }
                        onClick={() => {
                          if (
                            page === pageCount &&
                            !pageInfo?.hasPreviousPage
                          ) {
                            return
                          }

                          const _page = page + 1
                          setPage(_page)
                          loadMore()
                        }}
                      />
                    </PaginationItem>
                  </PaginationContent>
                </Pagination>
              )}
            </motion.div>
          </LoadingWrapper>
        </AnimationWrapper>
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
  const { sources, allUsers, onNavigateToThread, fetchingSources } =
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
      href={title ? `/search/${titleSlug}-${threadId}` : ''}
      onClick={onNavigateToThread}
    >
      <div className="transform-bg group flex-1 overflow-hidden rounded-lg px-3 py-2 hover:bg-accent">
        <div className="mb-1.5 flex items-center gap-2">
          <IconFiles className="shrink-0" />
          <LoadingWrapper
            loading={fetching || fetchingSources}
            fallback={
              <div className="w-full py-1.5">
                <Skeleton className="w-[60%]" />
              </div>
            }
          >
            <ThreadTitleWithMentions
              className="break-anywhere truncate text-lg font-medium"
              sources={sources}
              message={threadMessages?.[0]['node']['content']}
            />
          </LoadingWrapper>
        </div>
        <div className="flex items-center gap-2">
          <UserAvatar user={user} className="mr-0.5 h-4 w-4 shrink-0" />
          <div className="flex items-baseline gap-0.5">
            <div className="text-sm">{user?.name || user?.email}</div>
            <span className="text-muted-foreground">{'·'}</span>
            <div className="whitespace-nowrap text-xs text-muted-foreground">
              {formatCreatedAt(data.node.createdAt, 'Asked')}
            </div>
          </div>
        </div>
      </div>
    </Link>
  )
}

function ThreadTitleWithMentions({
  message,
  sources,
  className
}: {
  sources: ContextSource[] | undefined
  message: string | undefined
  className?: string
}) {
  const contentWithTags = useMemo(() => {
    if (!message) return null

    const firstLine = message.split('\n')[0] ?? ''
    return firstLine.split(MARKDOWN_SOURCE_REGEX).map((part, index) => {
      if (index % 2 === 1) {
        const sourceId = part
        const source = sources?.find(s => s.sourceId === sourceId)
        if (source) {
          return (
            <Mention
              key={index}
              id={source.sourceId}
              kind={source.sourceKind}
              label={source.sourceName}
              className="rounded-md border border-[#b3ada0] border-opacity-30 bg-[#e8e1d3] py-[1px] text-sm dark:bg-[#333333]"
            />
          )
        } else {
          return null
        }
      }
      return part
    })
  }, [sources, message])

  return <div className={cn(className)}>{contentWithTags}</div>
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
