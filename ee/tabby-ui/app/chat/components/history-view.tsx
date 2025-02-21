import { ThreadItem } from '@/components/chat/thread-item'
import { Button } from '@/components/ui/button'
import { listThreads } from '@/lib/tabby/query'
import { useQuery } from 'urql'

interface HistoryViewProps {
  onClose: () => void
  onNavigate: (threadId: string) => void
}

export function HistoryView({ onClose, onNavigate }: HistoryViewProps) {

  const [{ data, fetching }] = useQuery({
    // todo  -> myThreads
    query: listThreads,
    variables: {
      last: 25
    }
  })

  const onNavigateToThread = (threadId: string) => {
    onNavigate(threadId)
    onClose()
  }

  const threads = data?.threads?.edges
  const pageInfo = data?.threads?.pageInfo

  return (
    <div className="fixed inset-0 z-10 px-[16px] pt-4 md:pt-10 overflow-hidden">
      <div className="mx-auto max-w-5xl overflow-y-auto">
        <div className="sticky top-0 flex items-center justify-between">
          <span className="text-lg font-semibold">History</span>
          <Button size="sm" onClick={onClose}>
            Done
          </Button>
        </div>
        <div className='space-y-4 mt-4'>
          {threads?.map(thread => {
            return (
              <ThreadItem
                key={thread.node.id}
                data={thread}
                sources={undefined}
                onNavigate={onNavigateToThread}
              />
            )
          })}
        </div>
      </div>
    </div>
  )
}
