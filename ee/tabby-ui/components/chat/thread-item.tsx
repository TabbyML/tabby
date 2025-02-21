import { useQuery } from 'urql'

import { ContextSource, ListThreadsQuery } from '@/lib/gql/generates/graphql'
import { listThreadMessages } from '@/lib/tabby/query'
import { formatThreadTime } from '@/lib/utils'

import LoadingWrapper from '../loading-wrapper'
import { ThreadTitleWithMentions } from '../mention-tag'
import { Skeleton } from '../ui/skeleton'
import { replaceAtMentionPlaceHolderWithAt } from './form-editor/utils'

interface ThreadItemProps {
  data: ListThreadsQuery['threads']['edges'][0]
  sources: ContextSource[] | undefined
  fetchingSources?: boolean
  onNavigate: (threadId: string) => void
}

export function ThreadItem({
  data,
  sources,
  fetchingSources,
  onNavigate
}: ThreadItemProps) {
  const threadId = data.node.id

  const [{ data: threadMessagesData, fetching }] = useQuery({
    query: listThreadMessages,
    variables: {
      first: 1,
      threadId
    }
  })

  const threadMessages = threadMessagesData?.threadMessages?.edges

  return (
    <div
      onClick={e => onNavigate(data.node.id)}
      className="cursor-pointer transform-bg group flex-1 overflow-hidden rounded-md bg-background/60 px-4 py-3 hover:bg-accent/60"
    >
      <div className="mb-1.5 flex items-center gap-2">
        <LoadingWrapper
          loading={fetching || fetchingSources}
          fallback={
            <div className="w-full py-1.5">
              <Skeleton className="w-[60%]" />
            </div>
          }
        >
          <ThreadTitleWithMentions
            className="break-anywhere truncate text-base font-medium text-foreground/90"
            sources={sources}
            message={replaceAtMentionPlaceHolderWithAt(
              threadMessages?.[0]?.['node']['content'] ?? ''
            )}
          />
        </LoadingWrapper>
      </div>
      <div className="flex items-center gap-2">
        <div className="flex items-baseline gap-0.5">
          <div className="whitespace-nowrap text-xs text-muted-foreground">
            {formatThreadTime(data.node.createdAt, 'Asked')}
          </div>
        </div>
      </div>
    </div>
  )
}
