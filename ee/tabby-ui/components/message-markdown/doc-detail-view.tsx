import DOMPurify from 'dompurify'
import he from 'he'
import { isNil } from 'lodash-es'
import { marked } from 'marked'

import { Maybe } from '@/lib/gql/generates/graphql'
import type { AttachmentDocItem } from '@/lib/types'
import { getContent } from '@/lib/utils'

import { SiteFavicon } from '../site-favicon'
import { Badge } from '../ui/badge'
import {
  IconCheckCircled,
  IconCircleDot,
  IconGitMerge,
  IconGitPullRequest
} from '../ui/icons'
import { UserAvatar } from '../user-avatar'

export function DocDetailView({
  relevantDocument,
  enableDeveloperMode
}: {
  relevantDocument: AttachmentDocItem
  enableDeveloperMode?: boolean
}) {
  const sourceUrl = relevantDocument ? new URL(relevantDocument.link) : null
  const isIssue = relevantDocument?.__typename === 'MessageAttachmentIssueDoc'
  const isPR = relevantDocument?.__typename === 'MessageAttachmentPullDoc'
  const author =
    relevantDocument.__typename === 'MessageAttachmentWebDoc'
      ? undefined
      : relevantDocument.author
  const score = relevantDocument?.extra?.score

  return (
    <div className="prose max-w-none break-words dark:prose-invert prose-p:leading-relaxed prose-pre:mt-1 prose-pre:p-0">
      <div className="flex w-full flex-col gap-y-1 text-sm">
        <div className="m-0 flex items-center space-x-1 text-xs leading-none text-muted-foreground">
          <SiteFavicon
            hostname={sourceUrl!.hostname}
            className="m-0 mr-1 leading-none"
          />
          <p className="m-0 leading-none">{sourceUrl!.hostname}</p>
        </div>
        <p
          className="m-0 cursor-pointer font-bold leading-none transition-opacity hover:opacity-70"
          onClick={() => window.open(relevantDocument.link)}
        >
          {relevantDocument.title}
        </p>
        <div className="mb-2 w-auto">
          {isIssue && (
            <IssueDocInfoView closed={relevantDocument.closed} user={author} />
          )}
          {isPR && (
            <PullDocInfoView merged={relevantDocument.merged} user={author} />
          )}
        </div>
        <p className="m-0 line-clamp-4 leading-none">
          {normalizedText(getContent(relevantDocument))}
        </p>
        {!!enableDeveloperMode && !isNil(score) && (
          <p className="mt-4">Score: {score}</p>
        )}
      </div>
    </div>
  )
}

function PullDocInfoView({
  merged,
  user
}: {
  merged: boolean
  user: Maybe<{ id: string; email: string; name: string }> | undefined
}) {
  return (
    <div className="flex items-center gap-3">
      <PRStateBadge merged={merged} />
      <div className="flex flex-1 items-center gap-1.5">
        {!!user && (
          <>
            <UserAvatar user={user} className="not-prose h-5 w-5 shrink-0" />
            <span className="font-semibold text-muted-foreground">
              {user.name || user.email}
            </span>
          </>
        )}
      </div>
    </div>
  )
}

function IssueDocInfoView({
  closed,
  user
}: {
  closed: boolean
  user: Maybe<{ id: string; email: string; name: string }> | undefined
}) {
  return (
    <div className="flex items-center gap-3">
      <IssueStateBadge closed={closed} />
      <div className="flex flex-1 items-center gap-1.5">
        {!!user && (
          <>
            <UserAvatar user={user} className="not-prose h-5 w-5 shrink-0" />
            <span className="font-semibold text-muted-foreground">
              {user.name || user.email}
            </span>
          </>
        )}
      </div>
    </div>
  )
}

function IssueStateBadge({ closed }: { closed: boolean }) {
  return (
    <Badge
      variant={closed ? 'default' : 'secondary'}
      className="shrink-0 gap-1 py-1 text-xs"
    >
      {closed ? (
        <IconCheckCircled className="h-3.5 w-3.5" />
      ) : (
        <IconCircleDot className="h-3.5 w-3.5" />
      )}
      {closed ? 'Closed' : 'Open'}
    </Badge>
  )
}

function PRStateBadge({ merged }: { merged: boolean }) {
  return (
    <Badge
      variant={merged ? 'default' : 'secondary'}
      className="shrink-0 gap-1 py-1 text-xs"
    >
      {merged ? (
        <IconGitMerge className="h-3.5 w-3.5" />
      ) : (
        <IconGitPullRequest className="h-3.5 w-3.5" />
      )}
      {merged ? 'Merged' : 'Open'}
    </Badge>
  )
}

const normalizedText = (input: string) => {
  const sanitizedHtml = DOMPurify.sanitize(input, {
    ALLOWED_TAGS: [],
    ALLOWED_ATTR: []
  })
  const parsed = marked.parse(sanitizedHtml) as string
  const decoded = he.decode(parsed)
  const plainText = decoded.replace(/<\/?[^>]+(>|$)/g, '')
  return plainText
}
