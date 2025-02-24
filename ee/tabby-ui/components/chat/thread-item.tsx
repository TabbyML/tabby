import { useQuery } from 'urql'

import { ContextSource, ListThreadsQuery } from '@/lib/gql/generates/graphql'
import { listThreadMessages } from '@/lib/tabby/query'
import { formatThreadTime } from '@/lib/utils'

import LoadingWrapper from '../loading-wrapper'
import { ThreadTitleWithMentions } from '../mention-tag'
import { Button } from '../ui/button'
import { IconTrash } from '../ui/icons'
import { Skeleton } from '../ui/skeleton'
import { replaceAtMentionPlaceHolderWithAt } from './form-editor/utils'

interface ThreadItemProps {
  data: ListThreadsQuery['threads']['edges'][0]
  sources: ContextSource[] | undefined
  fetchingSources?: boolean
  onNavigate: (threadId: string) => void
  onDeleteThread?: (threadId: string) => Promise<any>
}

export function ThreadItem({
  data,
  sources,
  fetchingSources,
  onNavigate,
  onDeleteThread
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
      className="transform-bg group flex-1 cursor-pointer overflow-hidden rounded-md bg-background/70 px-4 py-3 hover:bg-accent/60"
    >
      <div className="flex flex-none justify-between gap-2">
        <div className="flex-1">
          <LoadingWrapper
            loading={fetching || fetchingSources}
            fallback={
              <div className="w-full py-1.5">
                <Skeleton className="w-[60%]" />
              </div>
            }
          >
            <ThreadTitleWithMentions
              className="break-anywhere flex-1 truncate text-base font-medium text-foreground/90"
              sources={sources}
              message={replaceAtMentionPlaceHolderWithAt(
                threadMessages?.[0]?.['node']['content'] ?? ''
              )}
            />
          </LoadingWrapper>
          <div className="flex items-center gap-2">
            <div className="flex items-baseline gap-0.5">
              <div className="whitespace-nowrap text-xs text-muted-foreground">
                {formatThreadTime(data.node.createdAt, 'Asked')}
              </div>
            </div>
          </div>
        </div>
        {!!onDeleteThread && (
          <Button
            size="icon"
            variant="hover-destructive"
            className="hidden shrink-0 p-0 group-hover:flex"
            onClick={e => {
              e.stopPropagation()
              onDeleteThread(data.node.id)
            }}
          >
            <IconTrash />
          </Button>
        )}
      </div>
    </div>
  )
}
