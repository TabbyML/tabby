'use client'

import { ReactNode, useContext, useMemo } from 'react'
import { Maybe } from 'graphql/jsutils/Maybe'
import { TerminalContext } from 'tabby-chat-panel/index'

import {
  ContextSource,
  MessageAttachmentCodeFileList,
  ThreadAssistantMessageReadingCode,
  ThreadAssistantMessageReadingDoc
} from '@/lib/gql/generates/graphql'
import {
  AttachmentDocItem,
  RelevantCodeContext,
  ServerFileContext
} from '@/lib/types'
import {
  cn,
  isAttachmentCommitDoc,
  isAttachmentIngestedDoc,
  isAttachmentIssueDoc,
  isAttachmentPullDoc,
  isCodeSourceContext,
  resolveDirectoryPath,
  resolveFileNameForDisplay
} from '@/lib/utils'
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
  IconListTree,
  IconTerminalSquare
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
import { CodeRangeLabel } from '@/components/code-range-label'
import { DocDetailView } from '@/components/message-markdown/doc-detail-view'
import { SourceIcon } from '@/components/source-icon'

import { StepItem } from './intermediate-step'
import { SearchContext } from './search-context'

interface ReadingCodeStepperProps {
  readingCode: ThreadAssistantMessageReadingCode | undefined
  // determine whether to display the 'collect documents' step
  readingDoc: ThreadAssistantMessageReadingDoc | undefined

  // loading state start
  isReadingCode: boolean | undefined
  isReadingFileList: boolean | undefined
  isReadingDocs: boolean | undefined
  // loading state end

  codeSourceId: Maybe<string>
  docQuery?: boolean
  className?: string
  serverCodeContexts: RelevantCodeContext[]
  clientCodeContexts: RelevantCodeContext[]
  docs?: Maybe<AttachmentDocItem[]> | undefined
  docQueryResources: Omit<ContextSource, 'id'>[] | undefined
  onContextClick?: (
    context: RelevantCodeContext,
    isInWorkspace?: boolean
  ) => void
  codeFileList?: Maybe<MessageAttachmentCodeFileList>
}

export function ReadingCodeStepper({
  isReadingCode,
  isReadingFileList,
  isReadingDocs,
  readingCode,
  readingDoc,
  codeSourceId,
  serverCodeContexts,
  clientCodeContexts,
  docs,
  codeFileList,
  onContextClick
}: ReadingCodeStepperProps) {
  const { contextInfo, enableDeveloperMode } = useContext(SearchContext)
  const totalContextLength =
    (clientCodeContexts?.length || 0) +
    serverCodeContexts.length +
    (docs?.length || 0)

  const targetRepo = useMemo(() => {
    if (!codeSourceId) return undefined

    const target = contextInfo?.sources.find(
      x => isCodeSourceContext(x.sourceKind) && x.sourceId === codeSourceId
    )
    return target
  }, [codeSourceId, contextInfo])

  const showFileListSubStep =
    !!readingCode?.fileList || !!codeFileList?.fileList?.length
  const showCodeSnippetsSubStep =
    readingCode?.snippet ||
    !!clientCodeContexts?.length ||
    !!serverCodeContexts?.length
  const showDocSubStep =
    !!codeSourceId &&
    (readingDoc?.sourceIds.includes(codeSourceId) || !!docs?.length)

  const steps = useMemo(() => {
    let result: Array<'fileList' | 'snippet' | 'docs' | 'commits'> = []
    if (showFileListSubStep) {
      result.push('fileList')
    }
    if (showCodeSnippetsSubStep) {
      result.push('snippet')
    }
    if (showDocSubStep) {
      result.push('docs')
    }
    return result
  }, [showFileListSubStep, showCodeSnippetsSubStep, showDocSubStep])

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
            {showFileListSubStep && (
              <StepItem
                key="fileList"
                title="Read codebase structure ..."
                isLoading={isReadingFileList}
                isLastItem={lastItem === 'fileList'}
                defaultOpen={!!codeFileList?.fileList?.length}
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
            {showCodeSnippetsSubStep && (
              <StepItem
                key="snippet"
                title="Search for relevant code snippets ..."
                isLoading={isReadingCode}
                defaultOpen={
                  !!clientCodeContexts?.length || !!serverCodeContexts?.length
                }
                isLastItem={lastItem === 'snippet'}
              >
                {(!!clientCodeContexts?.length ||
                  !!serverCodeContexts?.length) && (
                  <div className="mb-3 mt-2">
                    <div className="flex flex-wrap gap-2 text-xs font-semibold">
                      {clientCodeContexts?.map((item, index) => {
                        if (item.kind === 'terminal') {
                          return (
                            <TerminalContextItem
                              key={`client-termianl-${index}`}
                              context={item}
                            />
                          )
                        }
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
                        if (item.kind === 'terminal') {
                          return (
                            <TerminalContextItem
                              key={`server-terminal-${index}`}
                              context={item}
                            />
                          )
                        }
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
            {showDocSubStep && (
              <StepItem
                key="docs"
                title="Collect documents ..."
                isLastItem={lastItem === 'docs'}
                isLoading={isReadingDocs}
                defaultOpen={!!docs?.length}
              >
                {!!docs?.length && (
                  <div className="mb-3 mt-2 space-y-1">
                    <div className="flex flex-wrap items-center gap-2 text-xs">
                      {docs?.map((x, index) => {
                        const _key = isAttachmentIngestedDoc(x)
                          ? x.id
                          : isAttachmentCommitDoc(x)
                          ? x.sha
                          : x.link
                        return (
                          <div key={`${_key}_${index}`}>
                            <HoverCard openDelay={100} closeDelay={100}>
                              <HoverCardTrigger>
                                <CodebaseDocSummaryView doc={x} />
                              </HoverCardTrigger>
                              <HoverCardContent className="w-96 bg-background text-sm text-foreground dark:border-muted-foreground/60">
                                <DocDetailView
                                  enableDeveloperMode={enableDeveloperMode}
                                  relevantDocument={x}
                                  onLinkClick={url => {
                                    window.open(url)
                                  }}
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
  context: ServerFileContext
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
  const path = resolveDirectoryPath(context.filepath)

  const fileName = useMemo(() => {
    return resolveFileNameForDisplay(context.filepath)
  }, [context.filepath])

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
            <CodeRangeLabel
              className={cn('font-normal text-muted-foreground', {
                'group-hover:text-foreground': clickable
              })}
              range={context.range}
            ></CodeRangeLabel>
          </span>
        </div>
      </TooltipTrigger>
      <TooltipContent align="start" sideOffset={8} className="max-w-[24rem]">
        <div className="space-y-2">
          <div className="whitespace-nowrap font-medium">
            <span>{fileName}</span>
            <CodeRangeLabel range={context.range} />
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
function CodebaseDocSummaryView({ doc }: { doc: AttachmentDocItem }) {
  const isIssue = isAttachmentIssueDoc(doc)
  const isPR = isAttachmentPullDoc(doc)
  const isCommit = isAttachmentCommitDoc(doc)

  if (!isIssue && !isPR && !isCommit) {
    return null
  }

  const docName = isCommit
    ? `${doc.sha.slice(0, 7)}`
    : `#${doc.link.split('/').pop()}`
  const link = isCommit ? undefined : doc.link

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
      className={cn(
        'flex flex-nowrap items-center gap-0.5 rounded-md bg-muted px-1.5 py-0.5 font-semibold hover:text-foreground',
        {
          'cursor-pointer': !!link
        }
      )}
      onClick={() => {
        if (link) {
          window.open(link)
        }
      }}
    >
      {icon}
      <span>{docName}</span>
    </div>
  )
}

const TerminalContextItem: React.FC<{ context: TerminalContext }> = ({
  context
}) => {
  return (
    <div
      className={cn(
        'group flex flex-nowrap items-center gap-0.5 rounded-md bg-muted px-1.5 py-0.5',
        'hover:bg-muted/90'
      )}
      title={context.selection}
    >
      <IconTerminalSquare className="h-3 w-3" />
      <span>{context.name}</span>
    </div>
  )
}
