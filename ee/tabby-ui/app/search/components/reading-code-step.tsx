'use client'

import { ReactNode, useContext, useMemo } from 'react'
import { Maybe } from 'graphql/jsutils/Maybe'
import { isNil } from 'lodash-es'

import {
  ContextSource,
  MessageAttachmentCodeFileList,
  ThreadAssistantMessageReadingCode
} from '@/lib/gql/generates/graphql'
import { AttachmentDocItem, RelevantCodeContext } from '@/lib/types'
import { cn, isCodeSourceContext, resolveFileNameForDisplay } from '@/lib/utils'
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger
} from '@/components/ui/accordion'
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger
} from '@/components/ui/hover-card'
import {
  IconCheckCircled,
  IconCircleDot,
  IconCode,
  IconFileText,
  IconGitCommit,
  IconGitMerge,
  IconGitPullRequest,
  IconListTree
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
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'
import { DocDetailView } from '@/components/message-markdown/doc-detail-view'
import { SourceIcon } from '@/components/source-icon'

import { StepItem } from './intermediate-step'
import { SearchContext } from './search-context'

interface ReadingCodeStepperProps {
  isReadingCode: boolean | undefined
  isReadingFileList: boolean | undefined
  isReadingDocs: boolean | undefined
  readingCode: ThreadAssistantMessageReadingCode | undefined
  codeSourceId: Maybe<string>
  docQuery?: boolean
  className?: string
  serverCodeContexts: RelevantCodeContext[]
  clientCodeContexts: RelevantCodeContext[]
  webResources?: Maybe<AttachmentDocItem[]> | undefined
  commitResources?:
    | Maybe<
        Array<
          Extract<
            AttachmentDocItem,
            { __typename: 'MessageAttachmentCommitDoc' }
          >
        >
      >
    | undefined
  docQueryResources: Omit<ContextSource, 'id'>[] | undefined
  onContextClick?: (
    context: RelevantCodeContext,
    isInWorkspace?: boolean
  ) => void
  codeFileList?: Maybe<MessageAttachmentCodeFileList>
}

export function ReadingCodeStepper({
  docQuery,
  isReadingCode,
  isReadingFileList,
  isReadingDocs,
  readingCode,
  codeSourceId,
  serverCodeContexts,
  clientCodeContexts,
  webResources,
  commitResources,
  codeFileList,
  onContextClick
}: ReadingCodeStepperProps) {
  const { contextInfo, enableDeveloperMode } = useContext(SearchContext)
  const totalContextLength =
    (clientCodeContexts?.length || 0) +
    serverCodeContexts.length +
    (webResources?.length || 0) +
    (commitResources?.length || 0)
  const targetRepo = useMemo(() => {
    if (!codeSourceId) return undefined

    const target = contextInfo?.sources.find(
      x => isCodeSourceContext(x.sourceKind) && x.sourceId === codeSourceId
    )
    return target
  }, [codeSourceId, contextInfo])

  const steps = useMemo(() => {
    let result: Array<'fileList' | 'snippet' | 'docs' | 'commits'> = []
    if (readingCode?.fileList) {
      result.push('fileList')
    }
    if (readingCode?.snippet) {
      result.push('snippet')
    }
    if (commitResources?.length) {
      result.push('commits')
    }
    if (docQuery) {
      result.push('docs')
    }
    return result
  }, [readingCode?.fileList, readingCode?.snippet, commitResources, docQuery])

  const lastItem = useMemo(() => {
    return steps.slice().pop()
  }, [steps])

  return (
    <Accordion collapsible type="single" defaultValue="readingCode">
      <AccordionItem value="readingCode" className="border-0">
        <AccordionTrigger className="w-full py-2 pr-2">
          <div className="flex flex-1 items-center justify-between pr-2">
            <div className="flex flex-1 items-center gap-2">
              <IconCode className="mr-2 h-5 w-5 shrink-0" />
              <span>Look into</span>
              {!!targetRepo && (
                <div className="inline-flex cursor-pointer items-center gap-0.5 font-medium">
                  <SourceIcon
                    kind={targetRepo.sourceKind}
                    className="h-3.5 w-3.5 shrink-0"
                  />
                  <span className="truncate">{targetRepo.sourceName}</span>
                </div>
              )}
            </div>
            <div>
              {totalContextLength ? (
                <div className="text-sm text-muted-foreground">
                  {totalContextLength} sources
                </div>
              ) : null}
            </div>
          </div>
        </AccordionTrigger>
        <AccordionContent className="pb-0">
          <div className="space-y-2 text-sm text-muted-foreground">
            {readingCode?.fileList && (
              <StepItem
                key="fileList"
                title="Read codebase structure ..."
                isLoading={isReadingFileList}
                isLastItem={lastItem === 'fileList'}
              >
                {codeFileList?.fileList?.length ? (
                  <Sheet>
                    <SheetTrigger>
                      <div className="mb-3 mt-2 flex cursor-pointer flex-nowrap items-center gap-0.5 rounded-md bg-muted px-1.5 py-0.5 text-xs font-semibold hover:text-foreground">
                        <IconListTree className="h-3 w-3" />
                        <span>{codeFileList.fileList.length} items</span>
                      </div>
                    </SheetTrigger>
                    <SheetContent className="flex w-[50vw] min-w-[300px] flex-col gap-0 px-4 pb-0">
                      <SheetHeader className="border-b">
                        <SheetTitle>
                          {codeFileList.fileList.length} items
                        </SheetTitle>
                        <SheetClose />
                      </SheetHeader>
                      <pre className="flex-1 overflow-auto py-3">
                        {codeFileList.fileList.join('\n')}
                      </pre>
                      {codeFileList.truncated && (
                        <SheetFooter className="!justify-start border-t py-3 font-medium">
                          File list truncated. (Maximum number of items has been
                          reached)
                        </SheetFooter>
                      )}
                    </SheetContent>
                  </Sheet>
                ) : null}
              </StepItem>
            )}
            {readingCode?.snippet && (
              <StepItem
                key="snippet"
                title="Search for relevant code snippets ..."
                isLoading={isReadingCode}
                defaultOpen={!isReadingCode}
                isLastItem={lastItem === 'snippet'}
              >
                {!!totalContextLength && (
                  <div className="mb-3 mt-2">
                    <div className="flex flex-wrap gap-2 text-xs font-semibold">
                      {clientCodeContexts?.map((item, index) => {
                        return (
                          <CodeContextItem
                            key={`client-${index}`}
                            context={item}
                            clickable={false}
                            onContextClick={ctx => onContextClick?.(ctx, true)}
                            enableDeveloperMode={enableDeveloperMode}
                          />
                        )
                      })}
                      {serverCodeContexts?.map((item, index) => {
                        return (
                          <CodeContextItem
                            key={`server-${index}`}
                            context={item}
                            onContextClick={ctx => onContextClick?.(ctx, true)}
                            enableDeveloperMode={enableDeveloperMode}
                          />
                        )
                      })}
                    </div>
                  </div>
                )}
              </StepItem>
            )}
            {!!commitResources?.length && (
              <StepItem
                key="commits"
                title="Search for relevant Commits ..."
                isLastItem={lastItem === 'commits'}
                isLoading={isReadingDocs}
              >
                {!!commitResources?.length && (
                  <div className="mb-3 mt-2 space-y-1">
                    <div className="flex flex-wrap items-center gap-2 text-xs">
                      {commitResources?.map((x, index) => {
                        return (
                          <div key={`${x.sha}_${index}`}>
                            <HoverCard openDelay={100} closeDelay={100}>
                              <HoverCardTrigger>
                                <CodebaseDocView doc={x} />
                              </HoverCardTrigger>
                              <HoverCardContent className="w-96 bg-background text-sm text-foreground dark:border-muted-foreground/60">
                                <DocDetailView
                                  enableDeveloperMode={enableDeveloperMode}
                                  relevantDocument={x}
                                />
                              </HoverCardContent>
                            </HoverCard>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}
              </StepItem>
            )}
            {docQuery && (
              <StepItem
                key="docs"
                title="Search for relevant Issues/PRs ..."
                isLastItem={lastItem === 'docs'}
                isLoading={isReadingDocs}
              >
                {!!webResources?.length && (
                  <div className="mb-3 mt-2 space-y-1">
                    <div className="flex flex-wrap items-center gap-2 text-xs">
                      {webResources?.map((x, index) => {
                        const link =
                          x.__typename === 'MessageAttachmentCommitDoc'
                            ? `${x.gitUrl}/blob/${x.sha}/${x.changedFile}`
                            : x.link
                        return (
                          <div key={`${link}_${index}`}>
                            <HoverCard openDelay={100} closeDelay={100}>
                              <HoverCardTrigger>
                                <CodebaseDocView doc={x} />
                              </HoverCardTrigger>
                              <HoverCardContent className="w-96 bg-background text-sm text-foreground dark:border-muted-foreground/60">
                                <DocDetailView
                                  enableDeveloperMode={enableDeveloperMode}
                                  relevantDocument={x}
                                />
                              </HoverCardContent>
                            </HoverCard>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}
              </StepItem>
            )}
          </div>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  )
}

interface CodeContextItemProps {
  context: RelevantCodeContext
  onContextClick?: (context: RelevantCodeContext) => void
  clickable?: boolean
  enableDeveloperMode?: boolean
}

function CodeContextItem({
  context,
  clickable = true,
  onContextClick,
  enableDeveloperMode
}: CodeContextItemProps) {
  const isMultiLine =
    context.range &&
    !isNil(context.range?.start) &&
    !isNil(context.range?.end) &&
    context.range.start < context.range.end
  const pathSegments = context.filepath.split('/')
  const path = pathSegments.slice(0, pathSegments.length - 1).join('/')

  const fileName = useMemo(() => {
    return resolveFileNameForDisplay(context.filepath)
  }, [context.filepath])

  const rangeText = useMemo(() => {
    if (!context.range) return undefined

    let text = ''
    if (context.range.start) {
      text = String(context.range.start)
    }
    if (isMultiLine) {
      text += `-${context.range.end}`
    }
    return text
  }, [context.range])

  const scores = context?.extra?.scores

  return (
    <Tooltip delayDuration={100}>
      <TooltipTrigger asChild>
        <div
          className={cn(
            'group flex flex-nowrap items-center gap-0.5 rounded-md bg-muted px-1.5 py-0.5',
            {
              'cursor-pointer hover:text-foreground': clickable
            }
          )}
          onClick={e => {
            if (clickable) {
              onContextClick?.(context)
            }
          }}
        >
          <IconFileText className="h-3 w-3" />
          <span>
            <span>{fileName}</span>
            {rangeText ? (
              <span
                className={cn('font-normal text-muted-foreground', {
                  'group-hover:text-foreground': clickable
                })}
              >
                :{rangeText}
              </span>
            ) : null}
          </span>
        </div>
      </TooltipTrigger>
      <TooltipContent align="start" sideOffset={8} className="max-w-[24rem]">
        <div className="space-y-2">
          <div className="whitespace-nowrap font-medium">
            <span>{fileName}</span>
            {rangeText ? (
              <span className="text-muted-foreground">:{rangeText}</span>
            ) : null}
          </div>
          {!!path && (
            <div className="break-all text-xs text-muted-foreground">
              {path}
            </div>
          )}
          {enableDeveloperMode && context?.extra?.scores && (
            <div className="mt-4">
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
        </div>
      </TooltipContent>
    </Tooltip>
  )
}

// Issue, PR, Commit
function CodebaseDocView({ doc }: { doc: AttachmentDocItem }) {
  const isIssue = doc.__typename === 'MessageAttachmentIssueDoc'
  const isPR = doc.__typename === 'MessageAttachmentPullDoc'
  const isCommit = doc.__typename === 'MessageAttachmentCommitDoc'

  const docName = isCommit
    ? `${doc.sha.slice(0, 7)}`
    : `#${doc.link.split('/').pop()}`
  const link = isCommit
    ? `${doc.gitUrl}/blob/${doc.sha}/${doc.changedFile}`
    : doc.link

  let icon: ReactNode = null
  if (isIssue) {
    icon = doc.closed ? (
      <IconCheckCircled className="h-3 w-3" />
    ) : (
      <IconCircleDot className="h-3 w-3" />
    )
  }
  if (isPR) {
    icon = doc.merged ? (
      <IconGitMerge className="h-3 w-3" />
    ) : (
      <IconGitPullRequest className="h-3 w-3" />
    )
  }
  if (isCommit) {
    icon = <IconGitCommit className="h-3 w-3" />
  }

  return (
    <div
      className="flex cursor-pointer flex-nowrap items-center gap-0.5 rounded-md bg-muted px-1.5 py-0.5 font-semibold hover:text-foreground"
      onClick={() => window.open(link)}
    >
      {icon}
      <span>{docName}</span>
    </div>
  )
}
