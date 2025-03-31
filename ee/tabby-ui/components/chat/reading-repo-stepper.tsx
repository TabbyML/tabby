import { ReactNode, useContext, useMemo, useState } from 'react'

import {
  Maybe,
  MessageAttachmentCodeFileList,
  ThreadAssistantMessageReadingCode
} from '@/lib/gql/generates/graphql'
import { AttachmentDocItem, RelevantCodeContext } from '@/lib/types'

import { SourceIcon } from '../source-icon'
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '../ui/accordion'
import { IconCheckCircled, IconCircleDot, IconCode, IconExternalLink, IconFile, IconFileSearch2, IconGitCommit, IconGitMerge, IconGitPullRequest, IconListTree } from '../ui/icons'
import { ChatContext } from './chat-context'
import { StepItem } from './imtermediate-step'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger
} from '@/components/ui/hover-card'
import { cn, isAttachmentCommitDoc, isAttachmentIssueDoc, isAttachmentPullDoc, resolveFileNameForDisplay } from '@/lib/utils'
import { isNil } from 'lodash-es'
import { DocDetailView } from '../message-markdown/doc-detail-view'


interface ReadingRepoStepperProps {
  supportsOpenInEditor?: boolean
  isReadingCode: boolean | undefined
  isReadingFileList: boolean | undefined
  isReadingDocs: boolean | undefined
  readingCode: ThreadAssistantMessageReadingCode | undefined
  codeSourceId: Maybe<string>
  serverCodeContexts: RelevantCodeContext[]
  clientCodeContexts: RelevantCodeContext[]
  codeFileList?: Maybe<MessageAttachmentCodeFileList>
  docs?: Maybe<AttachmentDocItem[]> | undefined
  onContextClick?: (
    context: RelevantCodeContext,
    isInWorkspace?: boolean
  ) => void
}

export function ReadingRepoStepper({
  codeSourceId,
  codeFileList,
  clientCodeContexts,
  serverCodeContexts,
  docs,
  readingCode,
  isReadingCode,
  isReadingDocs,
  isReadingFileList,
  onContextClick
}: ReadingRepoStepperProps) {
  const { repos } = useContext(ChatContext)
  const totalContextLength =
    (clientCodeContexts?.length || 0) +
    serverCodeContexts.length +
    (docs?.length || 0)

  const targetRepo = useMemo(() => {
    return repos?.find(x => x.sourceId === codeSourceId)
  }, [repos, codeSourceId])

  return (
    <Accordion collapsible type="single">
      <AccordionItem value="readingRepo" className='border-none'>
        <AccordionTrigger className="w-full py-2 pr-2">
          <div className="flex whitespace-nowrap flex-nowrap flex-1 items-center justify-between pr-2">
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
        <AccordionContent className='pb-0'>
          <div className='space-y-2 text-xs text-muted-foreground'>
            {!!clientCodeContexts?.length && (
              <StepItem
                key='clientCode'
                title='Read references'
                isLoading={false}
                triggerClassname='text-sm'
              >
                <div className='mb-3 mt-2'>
                  <CodeContextList>
                    {clientCodeContexts.map((ctx, index) => {
                      return (
                        <ContextItem
                          key={`clientCode-${index}`}
                          context={ctx}
                        />
                      )
                    })}
                  </CodeContextList>
                </div>
              </StepItem>
            )}
            {readingCode?.fileList && (
              <StepItem
                key="fileList"
                title="Read codebase structure ..."
                isLoading={isReadingFileList}
                triggerClassname='text-sm'
              >
                {codeFileList?.fileList?.length ? (
                  // todo scrollarea
                  <div className="mb-3 mt-2 flex cursor-pointer flex-nowrap items-center gap-0.5 rounded-md bg-muted px-1.5 py-0.5 text-xs font-semibold hover:text-foreground">
                    <IconListTree className="h-3 w-3" />
                    <span>{codeFileList.fileList.length} items</span>
                  </div>
                ) : null}
              </StepItem>
            )}
            {readingCode?.snippet && (
              <StepItem
                key="snippet"
                title="Search for relevant code snippets ..."
                isLoading={isReadingCode}
                defaultOpen={!isReadingCode}
                triggerClassname='text-sm'
              >
                {(!!clientCodeContexts?.length ||
                  !!serverCodeContexts?.length) && (
                    <div className="mb-3 mt-2">
                      <CodeContextList>
                        {serverCodeContexts?.map((item, index) => {
                          return (
                            <ContextItem
                              key={`serverCode-${index}`}
                              context={item}
                              onContextClick={ctx => onContextClick?.(ctx, true)}
                            />
                          )
                        })}
                      </CodeContextList>
                    </div>
                  )}
              </StepItem>
            )}
            <StepItem
              key="docs"
              title="Collect documents ..."
              isLastItem
              isLoading={isReadingDocs}
              triggerClassname='text-sm'
            >
              {!!docs?.length && (
                <div className="mb-3 mt-2 space-y-1">
                  <div className="text-sm border p-2 rounded">
                    {docs?.map((x, index) => {
                      const _key =
                        isAttachmentCommitDoc(x)
                          ? x.sha
                          : x.link
                      return (
                        <div key={`${_key}_${index}`}>
                          <HoverCard openDelay={100} closeDelay={100}>
                            <HoverCardTrigger>
                              <CodebaseDocView doc={x} />
                            </HoverCardTrigger>
                            <HoverCardContent className="w-96 bg-background text-sm text-foreground dark:border-muted-foreground/60">
                              <DocDetailView
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
          </div>
        </AccordionContent>
      </AccordionItem>
    </Accordion >
  )
}

function CodeContextList({ children }: { children: React.ReactNode }) {
  if (!children) return null

  return (
    <div className='overflow-y-auto border rounded-lg p-2 space-y-2'>
      {children}
    </div>
  )
}

function ContextItem({
  context,
  clickable = true,
  onContextClick,
  enableTooltip,
  onTooltipClick,
  showExternalLinkIcon,
  showClientCodeIcon,
  isHighlighted
}: {
  context: RelevantCodeContext
  clickable?: boolean
  onContextClick?: (context: RelevantCodeContext) => void
  enableTooltip?: boolean
  onTooltipClick?: () => void
  showExternalLinkIcon?: boolean
  showClientCodeIcon?: boolean
  isHighlighted?: boolean
}) {
  const [tooltipOpen, setTooltipOpen] = useState(false)
  const isMultiLine =
    context.range &&
    !isNil(context.range?.start) &&
    !isNil(context.range?.end) &&
    context.range.start < context.range.end
  const pathSegments = context.filepath.split('/')
  const path = pathSegments.slice(0, pathSegments.length - 1).join('/')
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
          className={cn('rounded-md px-1 py-0.5', {
            'cursor-pointer hover:bg-accent': clickable,
            'cursor-default pointer-events-auto': !clickable,
            'bg-accent transition-all': isHighlighted
          })}
          onClick={e => clickable && onContextClick?.(context)}
        >
          <div className="flex items-center gap-1 overflow-hidden">
            <IconFile className="shrink-0" />
            <div className="flex-1 truncate" title={context.filepath}>
              <span className='text-foreground'>{resolveFileNameForDisplay(context.filepath)}</span>
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
              <span className="ml-2 text-xs text-muted-foreground">{path}</span>
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

// todo public component
// Issue, PR, Commit
function CodebaseDocView({ doc }: { doc: AttachmentDocItem }) {
  const isIssue = isAttachmentIssueDoc(doc)
  const isPR = isAttachmentPullDoc(doc)
  const isCommit = isAttachmentCommitDoc(doc)

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