import { useContext, useEffect, useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import {
  resetThreadsPageNo,
  setThreadsPageNo,
  useAnswerEngineStore
} from '@/lib/stores/answer-engine-store'
import { cn } from '@/lib/utils'
import { getPaginationItem } from '@/lib/utils/pagination'
import { IconSpinner } from '@/components/ui/icons'
import {
  Pagination,
  PaginationContent,
  PaginationEllipsis,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious
} from '@/components/ui/pagination'
import LoadingWrapper from '@/components/loading-wrapper'

import { ThreadItem } from './thread-item'
import { ThreadFeedsContext } from './threads-context'

const PAGE_SIZE = 25

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

export function AllThreadFeeds() {
  const { fetchingUsers, onNavigateToThread } = useContext(ThreadFeedsContext)
  const storedPageNo = useAnswerEngineStore(state => state.threadsPageNo)
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

  const pageInfo = data?.threads.pageInfo
  const threadCount = data?.threads?.edges?.length
  const pageCount = Math.ceil((threadCount || 0) / PAGE_SIZE)
  const showPagination =
    pageCount > 1 || (pageCount === 1 && pageInfo?.hasPreviousPage)
  const paginationPages = getPaginationItem(threadCount || 0, page, PAGE_SIZE)

  // threads for current page
  const threads = useMemo(() => {
    const _threads = data?.threads?.edges
    if (!_threads?.length) return []

    if (fetching && page >= 2) {
      // if fetching next page, keep previous page
      const previousPage = _threads
        .slice()
        .reverse()
        .slice((page - 2) * PAGE_SIZE, (page - 1) * PAGE_SIZE)
      return previousPage || []
    }

    return _threads
      .slice()
      .reverse()
      .slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)
  }, [data?.threads?.edges, page, fetching])

  const loadMore = () => {
    const startCursor = pageInfo?.startCursor
    if (
      startCursor &&
      data?.threads.edges.length &&
      data.threads.edges.findIndex(o => o.cursor === startCursor) > -1
    ) {
      setBeforeCursor(startCursor)
    } else {
      setBeforeCursor(data?.threads.edges[0]?.cursor)
    }
  }

  const hasNextPage = !!pageInfo?.hasPreviousPage
  const isNextPageDisabled =
    fetching ||
    !threads.length ||
    (page >= pageCount && !pageInfo?.hasPreviousPage)

  const handleNavigateToThread = () => {
    setThreadsPageNo(page)
    onNavigateToThread()
  }

  useEffect(() => {
    // reset pageNo store after it has been used
    resetThreadsPageNo()
  }, [])

  const hasThreads = !!data?.threads?.edges?.length

  return (
    <div className={cn('w-full')}>
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{
          once: true
        }}
        transition={{
          ease: 'easeOut',
          delay: 0.3
        }}
      >
        <LoadingWrapper
          loading={fetching || fetchingUsers}
          fallback={
            <div className="flex justify-center">
              <IconSpinner className="h-8 w-8" />
            </div>
          }
          delay={600}
        >
          {hasThreads && (
            <>
              <div className="relative flex flex-col gap-3 text-sm">
                {threads.map(t => {
                  return (
                    <ThreadItem
                      data={t}
                      key={t.node.id}
                      onNavigateToThread={handleNavigateToThread}
                    />
                  )
                })}
                {fetching && (
                  <div className="absolute inset-0 bottom-10 z-10 flex items-center justify-center backdrop-blur-sm" />
                )}
                {showPagination && (
                  // FIXME abstract Pagination
                  <Pagination className={cn('flex items-center justify-end')}>
                    <PaginationContent>
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
                      {paginationPages.map((item, index) => {
                        return (
                          <PaginationItem
                            key={`${item}-${index}`}
                            onClick={() => {
                              if (typeof item === 'number') {
                                setPage(item)
                              }
                            }}
                          >
                            {typeof item === 'number' ? (
                              <PaginationLink
                                className="cursor-pointer"
                                isActive={item === page}
                              >
                                {item}
                              </PaginationLink>
                            ) : (
                              <PaginationEllipsis />
                            )}
                          </PaginationItem>
                        )
                      })}
                      {hasNextPage && (
                        <PaginationItem>
                          <PaginationEllipsis />
                        </PaginationItem>
                      )}
                      <PaginationItem>
                        <PaginationNext
                          disabled={isNextPageDisabled}
                          onClick={() => {
                            if (isNextPageDisabled) {
                              return
                            }

                            const _page = page + 1
                            setPage(_page)
                            if (_page > pageCount) {
                              loadMore()
                            }
                          }}
                        />
                      </PaginationItem>
                    </PaginationContent>
                  </Pagination>
                )}
              </div>
            </>
          )}
        </LoadingWrapper>
      </motion.div>
    </div>
  )
}
