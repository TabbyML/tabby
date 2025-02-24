import { useQuery } from 'urql'

import { useMutation } from '@/lib/tabby/gql'
import { deleteThreadMutation, listMyThreads } from '@/lib/tabby/query'
import { Button } from '@/components/ui/button'
import { ThreadItem } from '@/components/chat/thread-item'

interface HistoryViewProps {
  onClose: () => void
  onNavigate: (threadId: string) => void
}

export function HistoryView({ onClose, onNavigate }: HistoryViewProps) {
  const [{ data, fetching }] = useQuery({
    query: listMyThreads,
    variables: {
      last: 25
    }
  })

  const onNavigateToThread = (threadId: string) => {
    onNavigate(threadId)
    onClose()
  }

  const deleteThread = useMutation(deleteThreadMutation)

  const onDeleteThread = (threadId: string) => {
    return deleteThread({ id: threadId })
  }

  const threads = data?.myThreads?.edges

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
        </div>
      </div>
    </div>
  )
}
