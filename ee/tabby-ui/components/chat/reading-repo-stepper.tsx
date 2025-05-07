import { ReactNode, useContext, useMemo, useState } from 'react'
import { isNil } from 'lodash-es'
import { TerminalContext } from 'tabby-chat-panel/index'

import {
  Maybe,
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
  resolveDirectoryPath,
  resolveFileNameForDisplay
} from '@/lib/utils'
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger
} from '@/components/ui/hover-card'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'

import { DocDetailView } from '../message-markdown/doc-detail-view'
import { SourceIcon } from '../source-icon'
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger
} from '../ui/accordion'
import {
  IconCheckCircled,
  IconCircleDot,
  IconCode,
  IconExternalLink,
  IconFile,
  IconFileSearch2,
  IconGitCommit,
  IconGitMerge,
  IconGitPullRequest,
  IconListTree
} from '../ui/icons'
import { ChatContext } from './chat-context'
import { StepItem } from './imtermediate-step'

interface ReadingRepoStepperProps {
  supportsOpenInEditor?: boolean
  isReadingCode: boolean | undefined
  isReadingFileList: boolean | undefined
  isReadingDocs: boolean | undefined
  readingCode: ThreadAssistantMessageReadingCode | undefined
  readingDoc: ThreadAssistantMessageReadingDoc | undefined
  codeSourceId: Maybe<string>
  serverCodeContexts: RelevantCodeContext[]
  clientCodeContexts: RelevantCodeContext[]
  codeFileList?: Maybe<MessageAttachmentCodeFileList>
  docs?: Maybe<Array<AttachmentDocItem>> | undefined
  onContextClick?: (
    context: RelevantCodeContext,
    isInWorkspace?: boolean
  ) => void
  openExternal: (url: string) => Promise<void>
}

export function ReadingRepoStepper({
  codeSourceId,
  codeFileList,
  clientCodeContexts,
  serverCodeContexts,
  docs,
  readingCode,
  readingDoc,
  isReadingCode,
  isReadingDocs,
  isReadingFileList,
  onContextClick,
  openExternal
}: ReadingRepoStepperProps) {
  const { repos } = useContext(ChatContext)
  const totalContextLength =
    (clientCodeContexts?.length || 0) +
    serverCodeContexts.length +
    (docs?.length || 0)

  const targetRepo = useMemo(() => {
    return repos?.find(x => x.sourceId === codeSourceId)
  }, [repos, codeSourceId])

  const showFileListSubStep =
    !!readingCode?.fileList || !!codeFileList?.fileList?.length
  const showCodeSnippetsSubStep =
    readingCode?.snippet || !!serverCodeContexts?.length
  const showDocSubStep =
    !!codeSourceId &&
    (readingDoc?.sourceIds.includes(codeSourceId) || !!docs?.length)

  const steps = useMemo(() => {
    let result: Array<'clientCode' | 'fileList' | 'snippet' | 'docs'> = []
    if (!!clientCodeContexts?.length) {
      result.push('clientCode')
    }
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
  }, [
    clientCodeContexts?.length,
    showFileListSubStep,
    showCodeSnippetsSubStep,
    showDocSubStep
  ])

  const lastItem = useMemo(() => {
    return steps.slice().pop()
  }, [steps])

  return (
    <Accordion defaultValue="readingRepo" collapsible type="single">
      <AccordionItem value="readingRepo" className="border-none">
        <AccordionTrigger className="w-full py-2 pr-2">
          <div className="flex flex-1 flex-nowrap items-center justify-between gap-0.5 truncate whitespace-nowrap pr-2">
            <div className="flex flex-1 items-center gap-2 overflow-x-hidden">
              <IconCode className="mr-2 h-5 w-5 shrink-0" />
              <span>Look into</span>
              {!!targetRepo && (
                <div className="inline-flex cursor-pointer items-center gap-0.5 truncate font-medium">
                  <SourceIcon
                    kind={targetRepo.sourceKind}
                    className="h-3.5 w-3.5 shrink-0"
                  />
                  <span className="truncate">{targetRepo.sourceName}</span>
                </div>
              )}
            </div>
            <div className="shrink-0">
              {totalContextLength ? (
                <div className="text-sm text-muted-foreground">
                  {totalContextLength} sources
                </div>
              ) : null}
            </div>
          </div>
        </AccordionTrigger>
        <AccordionContent className="pb-0">
          <div className="space-y-2 text-xs text-muted-foreground">
            {!!clientCodeContexts?.length && (
              <StepItem
                key="clientCode"
                title="Read code ..."
                isLoading={false}
                triggerClassname="text-sm"
                defaultOpen
                isLastItem={lastItem === 'clientCode'}
              >
                <div className="mb-3 mt-2">
                  <CodeContextList>
                    {clientCodeContexts.map((ctx, index) => {
                      if (ctx.kind === 'terminal') {
                        return (
                          <TerminalSummaryView
                            key={`terminal-${index}`}
                            context={ctx}
                          />
                        )
                      }
                      return (
                        <CodeSummaryView
                          key={`clientCode-${index}`}
                          context={ctx}
                          onContextClick={ctx => onContextClick?.(ctx, true)}
                        />
                      )
                    })}
                  </CodeContextList>
                </div>
              </StepItem>
            )}
            {showFileListSubStep && (
              <StepItem
                key="fileList"
                title="Read codebase structure ..."
                isLoading={isReadingFileList}
                triggerClassname="text-sm"
                defaultOpen={!!codeFileList?.fileList?.length}
                isLastItem={lastItem === 'fileList'}
              >
                {codeFileList?.fileList?.length ? (
                  <div className="mb-3 ml-2 mt-2 flex flex-nowrap items-center gap-2 whitespace-nowrap rounded-md px-1 py-0.5 text-sm font-semibold text-foreground hover:bg-accent">
                    <IconListTree className="shrink-0" />
                    <span>{codeFileList.fileList.length} items</span>
                    {!!codeFileList.truncated && (
                      <span className="ml-2 truncate text-muted-foreground">
                        File list truncated. (Maximum number of items has been
                        reached)
                      </span>
                    )}
                  </div>
                ) : null}
              </StepItem>
            )}
            {showCodeSnippetsSubStep && (
              <StepItem
                key="snippet"
                title="Search for relevant code snippets ..."
                isLoading={isReadingCode}
                defaultOpen={!!serverCodeContexts?.length}
                triggerClassname="text-sm"
                isLastItem={lastItem === 'snippet'}
              >
                {!!serverCodeContexts?.length && (
                  <div className="mb-3 mt-2">
                    <CodeContextList>
                      {serverCodeContexts?.map((item, index) => {
                        if (item.kind === 'terminal') {
                          return (
                            <TerminalSummaryView
                              key={`terminal-${index}`}
                              context={item}
                            />
                          )
                        }
                        return (
                          <CodeSummaryView
                            key={`serverCode-${index}`}
                            context={item}
                            onContextClick={ctx => onContextClick?.(ctx, false)}
                          />
                        )
                      })}
                    </CodeContextList>
                  </div>
                )}
              </StepItem>
            )}
            {showDocSubStep && (
              <StepItem
                key="docs"
                title="Collect documents ..."
                isLoading={isReadingDocs}
                triggerClassname="text-sm"
                defaultOpen={!!docs?.length}
                isLastItem={lastItem === 'docs'}
              >
                {!!docs?.length && (
                  <div className="mb-3 mt-2">
                    <div className="space-y-2 pl-2 text-sm">
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
                                <CodebaseDocSummaryView
                                  doc={x}
                                  openExternal={openExternal}
                                />
                              </HoverCardTrigger>
                              <HoverCardContent className="w-[50vw] bg-background text-sm text-foreground dark:border-muted-foreground/60 sm:w-96">
                                <DocDetailView
                                  relevantDocument={x}
                                  onLinkClick={openExternal}
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

function CodeContextList({ children }: { children: React.ReactNode }) {
  if (!children) return null

  return <div className="space-y-2 overflow-y-auto pl-2">{children}</div>
}

function CodeSummaryView({
  context,
  clickable = true,
  onContextClick,
  enableTooltip,
  onTooltipClick,
  showExternalLinkIcon,
  showClientCodeIcon
}: {
  context: ServerFileContext
  clickable?: boolean
  onContextClick?: (context: ServerFileContext) => void
  enableTooltip?: boolean
  onTooltipClick?: () => void
  showExternalLinkIcon?: boolean
  showClientCodeIcon?: boolean
}) {
  const [tooltipOpen, setTooltipOpen] = useState(false)
  const isMultiLine =
    context.range &&
    !isNil(context.range?.start) &&
    !isNil(context.range?.end) &&
    context.range.start < context.range.end
  const path = resolveDirectoryPath(context.filepath)
  const scores = context?.extra?.scores
  const onTooltipOpenChange = (v: boolean) => {
    if (!enableTooltip || !scores) return

    setTooltipOpen(v)
  }

  return (
    <Tooltip
      open={tooltipOpen}
      onOpenChange={onTooltipOpenChange}
      delayDuration={0}
    >
      <TooltipTrigger asChild>
        <div
          className={cn('rounded-md px-1 py-0.5 text-foreground', {
            'cursor-pointer hover:bg-accent': clickable,
            'cursor-default pointer-events-auto': !clickable
          })}
          onClick={e => clickable && onContextClick?.(context)}
        >
          <div className="flex items-center gap-2 overflow-hidden text-foreground">
            <IconFile className="shrink-0" />
            <div
              className="flex-1 truncate font-semibold"
              title={context.filepath}
            >
              <span>{resolveFileNameForDisplay(context.filepath)}</span>
              {context.range ? (
                <>
                  {context.range?.start && (
                    <span className="text-muted-foreground">
                      :{context.range.start}
                    </span>
                  )}
                  {isMultiLine && (
                    <span className="text-muted-foreground">
                      -{context.range.end}
                    </span>
                  )}
                </>
              ) : null}
              <span className="ml-2 text-muted-foreground">{path}</span>
            </div>
            {showClientCodeIcon && (
              <IconFileSearch2 className="shrink-0 text-muted-foreground" />
            )}
            {showExternalLinkIcon && (
              <IconExternalLink className="shrink-0 text-muted-foreground" />
            )}
          </div>
        </div>
      </TooltipTrigger>
      <TooltipContent
        align="start"
        onClick={onTooltipClick}
        className="cursor-pointer p-2"
      >
        <div>
          <div className="mb-2 font-semibold">Scores</div>
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
      </TooltipContent>
    </Tooltip>
  )
}

// Issue, PR, Commit
function CodebaseDocSummaryView({
  doc,
  openExternal
}: {
  doc: AttachmentDocItem
  openExternal: (url: string) => Promise<void>
}) {
  const isIssue = isAttachmentIssueDoc(doc)
  const isPR = isAttachmentPullDoc(doc)
  const isCommit = isAttachmentCommitDoc(doc)

  if (!isIssue && !isPR && !isCommit) {
    return null
  }

  const docName = isCommit ? `Commit-${doc.sha.slice(0, 7)}` : doc.title
  const link = isCommit ? undefined : doc.link

  let icon: ReactNode = null
  if (isIssue) {
    icon = doc.closed ? <IconCheckCircled /> : <IconCircleDot />
  }
  if (isPR) {
    icon = doc.merged ? <IconGitMerge /> : <IconGitPullRequest />
  }
  if (isCommit) {
    icon = <IconGitCommit />
  }

  return (
    <div
      className={cn(
        'flex flex-nowrap items-center gap-2 rounded-md px-1.5 py-0.5 font-semibold text-foreground hover:bg-accent hover:text-accent-foreground',
        {
          'cursor-pointer': !!link
        }
      )}
      onClick={() => {
        if (link) {
          openExternal(link)
        }
      }}
    >
      <span className="shrink-0">{icon}</span>
      <span className="truncate whitespace-nowrap">{docName}</span>
    </div>
  )
}

const TerminalSummaryView: React.FC<{ context: TerminalContext }> = ({
  context
}) => {
  return (
    <div className="flex flex-nowrap items-center gap-2 rounded-md px-1.5 py-0.5 font-semibold text-foreground hover:bg-accent hover:text-accent-foreground">
      <IconCode className="shrink-0" />
      <span className="truncate whitespace-nowrap">
        {context.name ?? `Terminal Selection`}
      </span>
    </div>
  )
}
