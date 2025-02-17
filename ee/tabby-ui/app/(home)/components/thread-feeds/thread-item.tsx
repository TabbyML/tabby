import { useContext, useMemo } from 'react'
import Link from 'next/link'
import slugify from '@sindresorhus/slugify'
import moment from 'moment'
import { useQuery } from 'urql'

import { SLUG_TITLE_MAX_LENGTH } from '@/lib/constants'
import { MARKDOWN_SOURCE_REGEX } from '@/lib/constants/regex'
import { ContextSource, ListThreadsQuery } from '@/lib/gql/generates/graphql'
import { listThreadMessages } from '@/lib/tabby/query'
import { cn, getTitleFromMessages } from '@/lib/utils'
import { IconFiles } from '@/components/ui/icons'
import { Skeleton } from '@/components/ui/skeleton'
import { replaceAtMentionPlaceHolderWithAt } from '@/components/chat/form-editor/utils'
import LoadingWrapper from '@/components/loading-wrapper'
import { Mention } from '@/components/mention-tag'
import { UserAvatar } from '@/components/user-avatar'

import { ThreadFeedsContext } from './threads-context'

interface ThreadItemProps {
  data: ListThreadsQuery['threads']['edges'][0]
  onNavigateToThread: () => void
}

export function ThreadItem({ data, onNavigateToThread }: ThreadItemProps) {
  const userId = data.node.userId
  const threadId = data.node.id
  const { sources, allUsers, fetchingSources } = useContext(ThreadFeedsContext)

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
      href={title ? `/search/${titleSlug}-${threadId}` : `/search/${threadId}`}
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
              message={replaceAtMentionPlaceHolderWithAt(
                threadMessages?.[0]?.['node']['content'] ?? ''
              )}
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
