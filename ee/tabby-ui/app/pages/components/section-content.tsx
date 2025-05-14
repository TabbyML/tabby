'use client'

import { ReactNode, useContext, useMemo, useState } from 'react'
import DOMPurify from 'dompurify'
import he from 'he'
import { compact, uniq } from 'lodash-es'
import { marked } from 'marked'

import { graphql } from '@/lib/gql/generates'
import {
  AttachmentCodeFileList,
  AttachmentIssueDoc,
  AttachmentPullDoc,
  MoveSectionDirection
} from '@/lib/gql/generates/graphql'
import { useMutation } from '@/lib/tabby/gql'
import { AttachmentCodeItem, AttachmentDocItem } from '@/lib/types'
import {
  buildCodeBrowserUrlForContext,
  cn,
  getAttachmentDocContent,
  getRangeFromAttachmentCode,
  isAttachmentCommitDoc,
  isAttachmentIngestedDoc,
  isAttachmentIssueDoc,
  isAttachmentPageDoc,
  isAttachmentPullDoc,
  isAttachmentWebDoc,
  resolveDirectoryPath,
  resolveFileNameForDisplay
} from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  IconArrowDown,
  IconBookOpen,
  IconCheckCircled,
  IconCircleDot,
  IconCode,
  IconEdit,
  IconFileUp,
  IconGitCommit,
  IconGitMerge,
  IconGitPullRequest,
  IconListTree,
  IconTrash
} from '@/components/ui/icons'
import {
  Sheet,
  SheetClose,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger
} from '@/components/ui/sheet'
import { Skeleton } from '@/components/ui/skeleton'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'
import { CodeRangeLabel } from '@/components/code-range-label'
import LoadingWrapper from '@/components/loading-wrapper'
import { MessageMarkdown } from '@/components/message-markdown'
import { SiteFavicon } from '@/components/site-favicon'
import { UserAvatar } from '@/components/user-avatar'

import { SectionItem } from '../types'
import { MessageContentForm } from './message-content-form'
import { PageContext } from './page-context'
import { SectionContentSkeleton } from './skeleton'

const updatePageSectionContentMutation = graphql(/* GraphQL */ `
  mutation updatePageSectionContent($input: UpdatePageSectionContentInput!) {
    updatePageSectionContent(input: $input)
  }
`)

export function SectionContent({
  className,
  section,
  isGenerating,
  enableMoveUp,
  enableMoveDown,
  onUpdate,
  enableDeveloperMode
}: {
  className?: string
  section: SectionItem
  isGenerating?: boolean
  enableMoveUp?: boolean
  enableMoveDown?: boolean
  onUpdate: (content: string) => void
  enableDeveloperMode: boolean
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
  const [showForm, setShowForm] = useState(false)
  const updatePageSectionContent = useMutation(updatePageSectionContentMutation)

  const attachmentCode = section.attachments.code
  const attachmentCodeFileList = section.attachments.codeFileList
  const attachmentDoc = section.attachments.doc
  const sources = useMemo(() => {
    return compact([
      ...(attachmentDoc || []),
      attachmentCodeFileList,
      ...attachmentCode
    ])
  }, [attachmentCodeFileList, attachmentCode, attachmentDoc])
  const sourceLen = sources?.length

  const sourceHostnames = useMemo(() => {
    let result: string[] = []
    for (let item of sources) {
      if (!item.__typename) {
        continue
      }

      switch (item.__typename) {
        case 'AttachmentCode':
          result.push('code')
          break
        case 'AttachmentCodeFileList':
          result.push('codeFileList')
          break
        case 'AttachmentIngestedDoc':
        case 'MessageAttachmentIngestedDoc':
          result.push('ingestedDoc')
          break
        case 'MessageAttachmentCommitDoc':
        case 'AttachmentCommitDoc':
          result.push('commit')
          break
        case 'AttachmentPageDoc':
        case 'MessageAttachmentPageDoc':
          result.push('page')
          break
        case 'AttachmentWebDoc':
        case 'MessageAttachmentWebDoc':
        case 'AttachmentIssueDoc':
        case 'MessageAttachmentIssueDoc':
        case 'MessageAttachmentPullDoc':
        case 'AttachmentPullDoc': {
          result.push(new URL(item.link).hostname)
          break
        }
      }
    }
    return uniq(compact(result)).slice(0, 3)
  }, [sources])

  const onMoveUp = () => {
    onMoveSectionPosition(section.id, MoveSectionDirection.Up)
  }

  const onMoveDown = () => {
    onMoveSectionPosition(section.id, MoveSectionDirection.Down)
  }

  const handleSubmitContentChange = async (content: string) => {
    const result = await updatePageSectionContent({
      input: {
        id: section.id,
        content
      }
    })

    if (result?.data?.updatePageSectionContent) {
      onUpdate(content)
      setShowForm(false)
    } else {
      let error = result?.error
      return error
    }
  }

  return (
    <div className={cn('flex flex-col gap-y-5', className)}>
      <LoadingWrapper loading={isPending} fallback={<SectionContentSkeleton />}>
        <div>
          {isGenerating && !section.content && (
            <Skeleton className="mt-1 h-40 w-full" />
          )}
          {showForm ? (
            <MessageContentForm
              message={section.content}
              onCancel={() => setShowForm(false)}
              onSubmit={handleSubmitContentChange}
            />
          ) : (
            <MessageMarkdown
              message={section.content}
              isStreaming={isGenerating}
              supportsOnApplyInEditorV2={false}
              className="prose-p:my-0.5 prose-ol:my-1 prose-ul:my-1"
              attachmentCode={attachmentCode}
              attachmentDocs={attachmentDoc}
            />
          )}
          {!isGenerating && (
            <div className="mt-3 flex items-center gap-3 text-sm">
              {sourceLen > 0 && (
                <Sheet>
                  <SheetTrigger asChild>
                    <div className="group relative flex w-32 cursor-pointer items-center overflow-hidden rounded-full border py-1 hover:bg-muted">
                      <div className="ml-1.5 flex items-center -space-x-2 transition-all duration-300 ease-in-out group-hover:space-x-0">
                        <SourceIconSummary hostnames={sourceHostnames} />
                      </div>
                      <span className="ml-2 whitespace-nowrap text-xs text-muted-foreground">
                        {sourceLen} sources
                      </span>
                      <div className="pointer-events-none absolute inset-y-0 right-0 w-2 bg-gradient-to-l from-background to-transparent"></div>
                    </div>
                  </SheetTrigger>
                  <SheetContent className="flex w-[50vw] min-w-[300px] flex-col">
                    <SheetHeader className="border-b">
                      <SheetTitle>{sourceLen} Sources</SheetTitle>
                      <SheetClose />
                    </SheetHeader>
                    <div className="flex-1 space-y-3 overflow-y-auto">
                      {sources.map((x, index) => {
                        return (
                          <SourcePreviewCard
                            enableDeveloperMode={enableDeveloperMode}
                            source={x}
                            key={index}
                          />
                        )
                      })}
                    </div>
                  </SheetContent>
                </Sheet>
              )}
              <div className="flex items-center gap-x-3">
                {isPageOwner && mode === 'edit' && !isLoading && !showForm && (
                  <>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-auto gap-0.5 px-2 py-1 font-medium text-foreground/60"
                      disabled={isLoading}
                      onClick={() => {
                        setShowForm(true)
                      }}
                    >
                      <IconEdit />
                      Edit
                    </Button>
                    {enableMoveUp && (
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-auto gap-0.5 px-2 py-1 font-medium text-foreground/60"
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
                        className="h-auto gap-0.5 px-2 py-1 font-medium text-foreground/60"
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
                      className="h-auto gap-0.5 px-2 py-1 font-medium text-foreground/60"
                      disabled={isLoading}
                      onClick={() => {
                        onDeleteSection(section.id)
                      }}
                    >
                      <IconTrash />
                      Delete
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

function SourcePreviewCard({
  source,
  enableDeveloperMode
}: {
  source: AttachmentDocItem | AttachmentCodeItem | AttachmentCodeFileList
  enableDeveloperMode: boolean
}) {
  const isCodeFileList = source.__typename === 'AttachmentCodeFileList'
  const isCode =
    source.__typename === 'MessageAttachmentCode' ||
    source.__typename === 'AttachmentCode'
  const isDoc =
    source.__typename === 'AttachmentIssueDoc' ||
    source.__typename === 'AttachmentPullDoc' ||
    source.__typename === 'AttachmentWebDoc' ||
    source.__typename === 'AttachmentPageDoc'
  const isCommit = source.__typename === 'AttachmentCommitDoc'
  const isIngestedDoc = source.__typename === 'AttachmentIngestedDoc'

  if (isCodeFileList) {
    return (
      <div className="flex gap-2 overflow-hidden rounded-lg border bg-accent p-3 text-accent-foreground hover:bg-accent/70">
        <div className="flex h-5 w-5 items-center justify-center">
          <IconListTree />
        </div>
        <div className="flex flex-1 flex-col overflow-hidden">
          <div className="shrink-0 text-xs">
            <span className="font-semibold leading-5">File list</span>
          </div>
          <div className="max-h-[90px] flex-1 overflow-auto">
            <pre className="text-xs">{source.fileList.join('\n')}</pre>
          </div>
          {!!source.truncated && (
            <div className="mt-2 shrink-0 border-t pt-2 text-xs text-muted-foreground">
              File list truncated. (Maximum number of items has been reached)
            </div>
          )}
        </div>
      </div>
    )
  }

  if (isCode) {
    const path = resolveDirectoryPath(source.filepath)
    const scores = source?.extra?.scores
    const showScores = enableDeveloperMode && !!scores
    return (
      <Tooltip delayDuration={0}>
        <TooltipTrigger asChild>
          <div
            className="flex w-full items-start gap-2"
            onClick={() => {
              if (!source.filepath) return
              const url = buildCodeBrowserUrlForContext(
                window.location.origin,
                {
                  kind: 'file',
                  ...source,
                  commit: source.commit ?? undefined,
                  range: getRangeFromAttachmentCode(source)
                }
              )
              window.open(url, '_blank')
            }}
          >
            <div className="relative flex flex-1 cursor-pointer gap-2 rounded-lg bg-accent p-3 text-accent-foreground hover:bg-accent/70">
              <div className="flex h-5 w-5 items-center justify-center">
                <IconCode />
              </div>
              <div className="flex flex-1 flex-col justify-between gap-y-1">
                <p className="line-clamp-1 w-full overflow-hidden text-ellipsis break-all text-xs font-semibold leading-5">
                  {resolveFileNameForDisplay(source.filepath)}
                  <CodeRangeLabel range={getRangeFromAttachmentCode(source)} />
                </p>
                {!!path && (
                  <div className="break-all text-xs text-muted-foreground">
                    {path}
                  </div>
                )}
              </div>
            </div>
          </div>
        </TooltipTrigger>
        <TooltipContent hidden={!showScores} side="left">
          {!!scores && (
            <div className="pt-1">
              <div className="mb-1 font-semibold">Scores</div>
              <div className="space-y-1">
                <div className="flex">
                  <span className="w-20">rrf:</span>
                  {scores?.rrf ?? '-'}
                </div>
                <div className="flex">
                  <span className="w-20">bm25:</span>
                  {scores?.bm25 ?? '-'}
                </div>
                <div className="flex">
                  <span className="w-20">embedding:</span>
                  {scores?.embedding ?? '-'}
                </div>
              </div>
            </div>
          )}
        </TooltipContent>
      </Tooltip>
    )
  }

  if (isDoc) {
    return (
      <div className="flex items-start gap-2">
        <div className="relative flex w-full cursor-pointer gap-2 rounded-lg bg-accent p-3 text-accent-foreground hover:bg-accent/70">
          <div
            className="relative flex w-full flex-col justify-between"
            onClick={() => window.open(source.link)}
          >
            <DocPreviewCard source={source} />
          </div>
        </div>
      </div>
    )
  }

  if (isIngestedDoc) {
    return (
      <div className="flex items-start gap-2">
        <div className="relative flex w-full gap-2 rounded-lg bg-accent p-3 text-accent-foreground hover:bg-accent/70">
          <div className="relative flex w-full flex-col justify-between">
            <DocPreviewCard source={source} />
          </div>
        </div>
      </div>
    )
  }

  if (isCommit) {
    return (
      <div className="flex items-start gap-2">
        <div className="relative flex flex-1 cursor-pointer flex-col justify-between rounded-lg bg-accent p-3 text-accent-foreground hover:bg-accent/70">
          <CommitPreviewCard source={source} />
        </div>
      </div>
    )
  }

  return null
}

function DocPreviewCard({ source }: { source: AttachmentDocItem }) {
  const isCommit = isAttachmentCommitDoc(source)
  const isIssue = isAttachmentIssueDoc(source)
  const isPR = isAttachmentPullDoc(source)
  const isWeb = isAttachmentWebDoc(source)
  const isPage = isAttachmentPageDoc(source)
  const isIngestion = isAttachmentIngestedDoc(source)

  const hostname = useMemo(() => {
    if (isCommit) return null
    try {
      let link = isIngestion ? source.ingestedDocLink : source.link
      if (link) {
        return new URL(link).hostname
      }
      return null
    } catch {
      return null
    }
  }, [source])

  const author =
    isWeb || isPage
      ? undefined
      : (source as AttachmentPullDoc | AttachmentIssueDoc).author

  const showAvatar = (isIssue || isPR) && !!author

  if (isCommit) return null

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
            {normalizedText(getAttachmentDocContent(source))}
          </p>
        )}
      </div>
      <div className="flex items-center text-xs text-muted-foreground">
        <div className="flex w-full flex-1 items-center justify-between gap-1">
          {isPage ? (
            <div className="flex flex-1 items-center gap-1">
              <IconBookOpen className="h-3.5 w-3.5" />
              Pages
            </div>
          ) : isIngestion ? (
            <div className="flex flex-1 items-center gap-1">
              <IconFileUp className="h-3.5 w-3.5" />
              Ingestion
            </div>
          ) : (
            <>
              {!!hostname && (
                <div className="flex flex-1 items-center">
                  <SiteFavicon hostname={hostname} />
                  <p className="ml-1 truncate">
                    {hostname.replace('www.', '').split('/')[0]}
                  </p>
                </div>
              )}
            </>
          )}
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

function CommitPreviewCard({ source }: { source: AttachmentDocItem }) {
  if (!isAttachmentCommitDoc(source)) {
    return null
  }

  const author = source.author
  const showAvatar = !!author

  return (
    <div className="flex flex-1 flex-col justify-between gap-y-1">
      <div className="flex flex-col gap-y-0.5">
        <p className="line-clamp-1 w-full overflow-hidden text-ellipsis break-all text-xs font-semibold">
          {source.sha.slice(0, 7)}
          {source.message ? `: ${source.message}` : ''}
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
            {normalizedText(getAttachmentDocContent(source))}
          </p>
        )}
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

function SourceIcon({
  children,
  className
}: {
  children: ReactNode
  className?: string
}) {
  return (
    <div
      className={cn(
        'relative z-20 flex h-5 w-5 items-center justify-center rounded-full transition-all duration-300 ease-in-out',
        className
      )}
    >
      {children}
    </div>
  )
}

function SourceIconSummary({ hostnames }: { hostnames: string[] }) {
  return (
    <>
      {hostnames.map(hostname => {
        if (hostname === 'codeFileList') {
          return (
            <SourceIcon
              key={hostname}
              className="bg-background group-hover:bg-transparent"
            >
              <IconListTree className="h-3.5 w-3.5 rounded-full bg-primary p-0.5 text-primary-foreground" />
            </SourceIcon>
          )
        }
        if (hostname === 'code') {
          return (
            <SourceIcon
              key={hostname}
              className="bg-background group-hover:bg-transparent"
            >
              <IconCode className="h-3.5 w-3.5 rounded-full bg-primary p-0.5 text-primary-foreground" />
            </SourceIcon>
          )
        }
        if (hostname === 'commit') {
          return (
            <SourceIcon
              key={hostname}
              className="bg-background group-hover:bg-transparent"
            >
              <IconGitCommit className="h-3.5 w-3.5 rounded-full bg-primary p-0.5 text-primary-foreground" />
            </SourceIcon>
          )
        }
        if (hostname === 'page') {
          return (
            <SourceIcon
              key={hostname}
              className="bg-background group-hover:bg-transparent"
            >
              <IconBookOpen className="h-3.5 w-3.5 rounded-full bg-primary p-[0.15rem] text-primary-foreground" />
            </SourceIcon>
          )
        }

        if (hostname === 'ingestedDoc') {
          return (
            <SourceIcon
              key={hostname}
              className="bg-background group-hover:bg-transparent"
            >
              <IconFileUp className="h-3.5 w-3.5 rounded-full bg-primary p-[0.15rem] text-primary-foreground" />
            </SourceIcon>
          )
        }

        return (
          <SourceIcon
            className="flex h-5 w-5 items-center justify-center bg-background p-0 group-hover:bg-transparent"
            key={hostname}
          >
            <SiteFavicon hostname={hostname} />
          </SourceIcon>
        )
      })}
    </>
  )
}
