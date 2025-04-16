import { useMemo, useState } from 'react'
import { useQuery } from 'urql'

import { useMutation } from '@/lib/tabby/gql'
import {
  contextInfoQuery,
  deleteThreadMutation,
  listMyThreads
} from '@/lib/tabby/query'
import { Button } from '@/components/ui/button'
import { CardContent } from '@/components/ui/card'
import { IconFileSearch, IconSpinner } from '@/components/ui/icons'
import { ThreadItem } from '@/components/chat/thread-item'
import { LoadMoreIndicator } from '@/components/load-more-indicator'
import LoadingWrapper from '@/components/loading-wrapper'

interface HistoryViewProps {
  onClose: () => void
  onNavigate: (threadId: string) => void
  onDeleted: (threadId: string) => void
}

export function HistoryView({
  onClose,
  onNavigate,
  onDeleted
}: HistoryViewProps) {
  const [{ data: contextInfoData, fetching: fetchingContextInfo }] = useQuery({
    query: contextInfoQuery
  })
  const [beforeCursor, setBeforeCursor] = useState<string | undefined>()
  const [{ data, fetching }] = useQuery({
    query: listMyThreads,
    variables: {
      last: 25,
      before: beforeCursor
    }
  })

  const pageInfo = data?.myThreads?.pageInfo
  const threads = useMemo(() => {
    return data?.myThreads?.edges.slice().reverse()
  }, [data?.myThreads?.edges])

  const loadMore = () => {
    const startCursor = pageInfo?.startCursor
    if (
      startCursor &&
      data?.myThreads.edges.length &&
      data.myThreads.edges.findIndex(o => o.cursor === startCursor) > -1
    ) {
      setBeforeCursor(startCursor)
    } else {
      setBeforeCursor(data?.myThreads.edges[0]?.cursor)
    }
  }

  const onNavigateToThread = (threadId: string) => {
    onNavigate(threadId)
    setTimeout(() => {
      onClose()
    }, 100)
  }

  const deleteThread = useMutation(deleteThreadMutation)

  const onDeleteThread = (threadId: string) => {
    return deleteThread({ id: threadId }).then(data => {
      if (data?.data?.deleteThread) {
        onDeleted(threadId)
      }
    })
  }

  return (
    <div className="editor-bg fixed inset-0 z-50 overflow-hidden px-[16px] pt-4 md:pt-10">
      <div className="mx-auto h-full max-w-5xl overflow-y-auto pb-8">
        <div className="editor-bg sticky top-0 flex items-center justify-between pb-3">
          <span className="text-lg font-semibold">History</span>
          <Button size="sm" onClick={onClose}>
            Done
          </Button>
        </div>
        <div className="mt-4 space-y-4">
          <LoadingWrapper
            loading={fetching}
            fallback={
              <div className="flex justify-center">
                <IconSpinner className="h-6 w-6" />
              </div>
            }
          >
            {!threads?.length ? (
              <>
                <CardContent className="mt-6 flex items-center justify-center gap-1 rounded-lg border py-12">
                  <IconFileSearch className="h-6 w-6" />
                  <p className="font-semibold">No data</p>
                </CardContent>
              </>
            ) : (
              <>
                {threads?.map(thread => {
                  return (
                    <ThreadItem
                      key={thread.node.id}
                      data={thread}
                      sources={contextInfoData?.contextInfo?.sources}
                      fetchingSources={fetchingContextInfo}
                      onNavigate={onNavigateToThread}
                      onDeleteThread={onDeleteThread}
                    />
                  )
                })}
                {!!pageInfo?.hasPreviousPage && (
                  <LoadMoreIndicator
                    onLoad={loadMore}
                    isFetching={fetching}
                    itemCount={data?.myThreads?.edges?.length ?? 0}
                  >
                    <div className="flex justify-center">
                      <IconSpinner className="h-6 w-6" />
                    </div>
                  </LoadMoreIndicator>
                )}
              </>
            )}
          </LoadingWrapper>
        </div>
      </div>
    </div>
  )
}
