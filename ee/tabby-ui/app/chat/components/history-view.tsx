import { useMemo, useState } from 'react'
import { useQuery } from 'urql'

import { useMutation } from '@/lib/tabby/gql'
import { deleteThreadMutation, listMyThreads } from '@/lib/tabby/query'
import { Button } from '@/components/ui/button'
import { IconSpinner } from '@/components/ui/icons'
import { ThreadItem } from '@/components/chat/thread-item'
import { LoadMoreIndicator } from '@/components/load-more-indicator'

interface HistoryViewProps {
  onClose: () => void
  onNavigate: (threadId: string) => void
}

export function HistoryView({ onClose, onNavigate }: HistoryViewProps) {
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
    return deleteThread({ id: threadId })
  }

  return (
    <div className="fixed inset-0 z-10 overflow-hidden px-[16px] pt-4 md:pt-10">
      <div className="mx-auto h-full max-w-5xl overflow-y-auto pb-8">
        <div className="editor-bg sticky top-0 flex items-center justify-between">
          <span className="text-lg font-semibold">History</span>
          <Button size="sm" onClick={onClose}>
            Done
          </Button>
        </div>
        <div className="mt-4 space-y-4">
          {threads?.map(thread => {
            return (
              <ThreadItem
                key={thread.node.id}
                data={thread}
                sources={undefined}
                onNavigate={onNavigateToThread}
                onDeleteThread={onDeleteThread}
              />
            )
          })}
          {!!pageInfo?.hasPreviousPage && (
            <LoadMoreIndicator onLoad={loadMore} isFetching={fetching}>
              <div className="flex justify-center">
                <IconSpinner className="h-6 w-6" />
              </div>
            </LoadMoreIndicator>
          )}
        </div>
      </div>
    </div>
  )
}
