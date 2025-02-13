'use client'

import { useContext } from 'react'
import DOMPurify from 'dompurify'
import he from 'he'
import { marked } from 'marked'

import { MoveSectionDirection } from '@/lib/gql/generates/graphql'
import { AttachmentCodeItem, AttachmentDocItem } from '@/lib/types'
import { cn, getContent } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import {
  IconArrowDown,
  IconCheckCircled,
  IconCircleDot,
  IconGitMerge,
  IconGitPullRequest,
  IconTrash
} from '@/components/ui/icons'
import {
  Sheet,
  SheetClose,
  SheetContent,
  SheetFooter,
  SheetHeader,
  SheetTitle,
  SheetTrigger
} from '@/components/ui/sheet'
import { Skeleton } from '@/components/ui/skeleton'
import LoadingWrapper from '@/components/loading-wrapper'
import { MessageMarkdown } from '@/components/message-markdown'
import { SiteFavicon } from '@/components/site-favicon'
import { UserAvatar } from '@/components/user-avatar'

import { SectionItem } from '../types'
import { PageContext } from './page-context'
import { SectionContentSkeleton } from './skeleton'

export function SectionContent({
  className,
  section,
  isGenerating,
  enableMoveUp,
  enableMoveDown
}: {
  className?: string
  section: SectionItem
  isGenerating?: boolean
  enableMoveUp?: boolean
  enableMoveDown?: boolean
}) {
  const {
    mode,
    isPageOwner,
    isLoading,
    pendingSectionIds,
    onDeleteSection,
    onMoveSectionPosition
  } = useContext(PageContext)
  const isPending = pendingSectionIds.has(section.id) && !section.content
  // FIXME
  const sources: any[] = []
  const sourceLen = 0

  const onMoveUp = () => {
    onMoveSectionPosition(section.id, MoveSectionDirection.Up)
  }

  const onMoveDown = () => {
    onMoveSectionPosition(section.id, MoveSectionDirection.Down)
  }

  return (
    <div className={cn('flex flex-col gap-y-5', className)}>
      <LoadingWrapper loading={isPending} fallback={<SectionContentSkeleton />}>
        <div>
          {isGenerating && !section.content && (
            <Skeleton className="mt-1 h-40 w-full" />
          )}
          <MessageMarkdown
            message={section.content}
            canWrapLongLines={!isGenerating}
            supportsOnApplyInEditorV2={false}
            className="prose-p:my-0.5 prose-ol:my-1 prose-ul:my-1"
          />
          {!isGenerating && (
            <div className="mt-3 flex items-center gap-3 text-sm">
              {sourceLen > 0 && (
                <Sheet>
                  <SheetTrigger asChild>
                    <div className="cursor-pointer rounded-full border px-2 py-1">
                      {sourceLen} sources
                    </div>
                  </SheetTrigger>
                  <SheetContent className="flex w-[50vw] min-w-[300px] flex-col">
                    <SheetHeader className="border-b">
                      <SheetTitle>Sources</SheetTitle>
                      <SheetClose />
                    </SheetHeader>
                    <div className="flex-1 space-y-3 overflow-y-auto">
                      {sources.map((x, index) => {
                        return <SourceCard source={x} key={index} />
                      })}
                    </div>
                    <SheetFooter>
                      <Button>Remove sources</Button>
                    </SheetFooter>
                  </SheetContent>
                </Sheet>
              )}
              <div className="flex items-center gap-x-3">
                {isPageOwner && mode === 'edit' && !isLoading && (
                  <>
                    {enableMoveUp && (
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-auto gap-0.5 px-2 py-1 font-normal"
                        onClick={e => onMoveUp()}
                        disabled={isLoading}
                      >
                        <IconArrowDown className="rotate-180" />
                        Move Up
                      </Button>
                    )}
                    {enableMoveDown && (
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-auto gap-0.5 px-2 py-1 font-normal"
                        onClick={e => onMoveDown()}
                        disabled={isLoading}
                      >
                        <IconArrowDown />
                        Move Down
                      </Button>
                    )}
                    <Button
                      size="sm"
                      variant="hover-destructive"
                      className="h-auto gap-0.5 px-2 py-1 font-normal"
                      disabled={isLoading}
                      onClick={() => {
                        onDeleteSection(section.id)
                      }}
                    >
                      <IconTrash />
                      Delete Section
                    </Button>
                  </>
                )}
              </div>
            </div>
          )}
        </div>
      </LoadingWrapper>
    </div>
  )
}

function SourceCard({
  source
}: {
  source: AttachmentDocItem | AttachmentCodeItem
}) {
  const { mode } = useContext(PageContext)
  const isEditMode = mode === 'edit'

  const isDoc =
    source.__typename === 'MessageAttachmentIssueDoc' ||
    source.__typename === 'MessageAttachmentPullDoc' ||
    source.__typename === 'MessageAttachmentWebDoc'

  if (isDoc) {
    return (
      <div className="flex items-start gap-2">
        {isEditMode && <Checkbox className="mt-2" />}
        <div
          className="relative flex cursor-pointer flex-col justify-between rounded-lg border bg-card p-3 text-card-foreground hover:bg-card/60"
          onClick={() => window.open(source.link)}
        >
          <DocSourceCard source={source} />
        </div>
      </div>
    )
  }

  return (
    <div className="flex w-full items-start gap-2">
      {isEditMode && <Checkbox className="mt-2" />}
      <div className="relative flex flex-1 cursor-pointer flex-col justify-between rounded-lg border bg-card p-3 text-card-foreground hover:bg-card/60">
        <div className="flex flex-1 flex-col justify-between gap-y-1">
          <div className="flex flex-col gap-y-0.5">
            <p className="line-clamp-1 w-full overflow-hidden text-ellipsis break-all text-xs font-semibold">
              {source.filepath}
            </p>
          </div>
          <div className="flex items-center text-xs text-muted-foreground">
            <div className="flex w-full flex-1 items-center justify-between gap-1">
              <div className="flex items-center">
                <SiteFavicon hostname={source.gitUrl} />
                <p className="ml-1 truncate">{source.gitUrl}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function DocSourceCard({ source }: { source: AttachmentDocItem }) {
  const { hostname } = new URL(source.link)
  const isIssue = source.__typename === 'MessageAttachmentIssueDoc'
  const isPR = source.__typename === 'MessageAttachmentPullDoc'
  const author =
    source.__typename === 'MessageAttachmentWebDoc' ? undefined : source.author

  const showAvatar = (isIssue || isPR) && !!author

  return (
    <div className="flex flex-1 flex-col justify-between gap-y-1">
      <div className="flex flex-col gap-y-0.5">
        <p className="line-clamp-1 w-full overflow-hidden text-ellipsis break-all text-xs font-semibold">
          {source.title}
        </p>

        {showAvatar && (
          <div className="flex items-center gap-1 overflow-x-hidden">
            <UserAvatar user={author} className="h-3.5 w-3.5 shrink-0" />
            <p className="truncate text-xs font-medium text-muted-foreground">
              {author?.name}
            </p>
          </div>
        )}
        {!showAvatar && (
          <p
            className={cn(
              ' w-full overflow-hidden text-ellipsis break-all text-xs text-muted-foreground',
              !showAvatar ? 'line-clamp-2' : 'line-clamp-1'
            )}
          >
            {normalizedText(getContent(source))}
          </p>
        )}
      </div>
      <div className="flex items-center text-xs text-muted-foreground">
        <div className="flex w-full flex-1 items-center justify-between gap-1">
          <div className="flex items-center">
            <SiteFavicon hostname={hostname} />
            <p className="ml-1 truncate">
              {hostname.replace('www.', '').split('/')[0]}
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-1">
            {isIssue && (
              <>
                {source.closed ? (
                  <IconCheckCircled className="h-3.5 w-3.5" />
                ) : (
                  <IconCircleDot className="h-3.5 w-3.5" />
                )}
                <span>{source.closed ? 'Closed' : 'Open'}</span>
              </>
            )}
            {isPR && (
              <>
                {source.merged ? (
                  <IconGitMerge className="h-3.5 w-3.5" />
                ) : (
                  <IconGitPullRequest className="h-3.5 w-3.5" />
                )}
                {source.merged ? 'Merged' : 'Open'}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

// Remove HTML and Markdown format
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
