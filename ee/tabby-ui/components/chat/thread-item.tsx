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

  // If we don't have any messages, hide the thread item
  if (!fetching && !threadMessages?.length) {
    return null
  }

  return (
    <div
      onClick={e => onNavigate(data.node.id)}
      className="group cursor-pointer overflow-hidden rounded-md bg-background/70 px-4 py-3 transition-colors hover:bg-accent/60"
    >
      <div className="flex flex-nowrap justify-between gap-2 overflow-hidden">
        <div className="flex-1 overflow-hidden">
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
          <div className="mt-1 truncate whitespace-nowrap text-xs text-muted-foreground">
            {formatThreadTime(data.node.createdAt, 'Asked')}
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
