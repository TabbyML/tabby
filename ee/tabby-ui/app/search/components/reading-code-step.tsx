'use client'

import { ReactNode, useContext, useEffect, useMemo, useState } from 'react'
import { Maybe } from 'graphql/jsutils/Maybe'
import { isNil } from 'lodash-es'

import {
  ContextSource,
  ThreadAssistantMessageReadingCode
} from '@/lib/gql/generates/graphql'
import { AttachmentDocItem, RelevantCodeContext } from '@/lib/types'
import { isCodeSourceContext, resolveFileNameForDisplay } from '@/lib/utils'
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
import { IconBlocks, IconCheckFull, IconSpinner } from '@/components/ui/icons'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'
import { DocDetailView } from '@/components/message-markdown/doc-detail-view'
import { SourceIcon } from '@/components/source-icon'

import { SearchContext } from './search-context'

interface ReadingCodeStepperProps {
  isReadingCode: boolean | undefined
  isReadingFileList: boolean | undefined
  isReadingDocs: boolean | undefined
  readingCode: ThreadAssistantMessageReadingCode | undefined
  codeSourceId: Maybe<string>
  // docs & repo
  sourceIds?: string[]
  // whether publicSearch is enabled or not
  publicSearch?: boolean
  docQuery?: boolean
  className?: string
  serverCodeContexts: RelevantCodeContext[]
  clientCodeContexts: RelevantCodeContext[]
  webResources?: Maybe<AttachmentDocItem[]> | undefined
  docQueryResources: Omit<ContextSource, 'id'>[] | undefined
  onContextClick?: (
    context: RelevantCodeContext,
    isInWorkspace?: boolean
  ) => void
}

export function ReadingCodeStepper({
  isReadingCode,
  isReadingFileList,
  isReadingDocs,
  readingCode,
  codeSourceId,
  serverCodeContexts,
  clientCodeContexts,
  webResources,
  docQuery,
  docQueryResources,
  onContextClick
}: ReadingCodeStepperProps) {
  const { contextInfo } = useContext(SearchContext)
  const totalContextLength =
    (clientCodeContexts?.length || 0) + serverCodeContexts.length
  const targetRepo = useMemo(() => {
    if (!codeSourceId) return undefined

    const target = contextInfo?.sources.find(
      x => isCodeSourceContext(x.sourceKind) && x.sourceId === codeSourceId
    )
    return target
  }, [codeSourceId, contextInfo])

  const steps = useMemo(() => {
    let result: Array<'fileList' | 'snippet' | 'docs'> = []
    if (readingCode?.fileList) {
      result.push('fileList')
    }
    if (readingCode?.snippet) {
      result.push('snippet')
    }
    if (docQuery) {
      result.push('docs')
    }
    return result
  }, [readingCode?.fileList, readingCode?.snippet])

  const lastItem = useMemo(() => {
    return steps.slice().pop()
  }, [steps])

  return (
    <Accordion collapsible type="single" defaultValue="readingCode">
      <AccordionItem value="readingCode" className="mb-6 border-0">
        <AccordionTrigger className="w-full py-2 pr-2">
          <div className="flex flex-1 items-center justify-between pr-2">
            <div className="flex flex-1 items-center gap-2">
              <IconBlocks className="mr-2 h-5 w-5 shrink-0" />
              <span>Thinking</span>
              {/* {!!targetRepo && (
                <div className="inline-flex cursor-pointer items-center gap-0.5 font-medium">
                  <SourceIcon
                    kind={targetRepo.sourceKind}
                    className="h-3.5 w-3.5 shrink-0"
                  />
                  <span className="truncate">{targetRepo.sourceName}</span>
                </div>
              )} */}
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
              />
            )}
            {readingCode?.snippet && (
              <StepItem
                key="snippet"
                title="Search for relevant code snippets ..."
                isLoading={isReadingCode}
                defaultOpen={!isReadingCode}
                isLastItem={lastItem === 'snippet'}
              >
                {!!targetRepo && (
                  <div className="mb-3 mt-2 space-y-2">
                    <div className="text-xs">Searching</div>
                    <div className="inline-flex cursor-pointer items-center gap-0.5 rounded-md text-xs font-medium">
                      <SourceIcon
                        kind={targetRepo.sourceKind}
                        className="h-3 w-3 shrink-0"
                      />
                      <span className="truncate">{targetRepo.sourceName}</span>
                    </div>
                  </div>
                )}
                {!!totalContextLength && (
                  <div className="mb-3 mt-2 space-y-2">
                    <div className="text-xs">Reading</div>
                    <div className="flex flex-wrap gap-2 text-xs font-semibold">
                      {clientCodeContexts?.map((item, index) => {
                        return (
                          <CodeContextItem
                            key={`client-${index}`}
                            context={item}
                            onContextClick={ctx => onContextClick?.(ctx, true)}
                          />
                        )
                      })}
                      {serverCodeContexts?.map((item, index) => {
                        return (
                          <CodeContextItem
                            key={`server-${index}`}
                            context={item}
                            onContextClick={ctx => onContextClick?.(ctx, true)}
                          />
                        )
                      })}
                    </div>
                  </div>
                )}
              </StepItem>
            )}
            {docQuery && (
              <StepItem
                title="Search for relevant web docs ..."
                isLastItem={lastItem === 'docs'}
                isLoading={isReadingDocs}
              >
                {(!!targetRepo || !!docQueryResources?.length) && (
                  <div className="mb-3 mt-2 space-y-2">
                    <div className="text-xs">Searching</div>
                    <div className="flex flex-wrap gap-2 text-xs font-semibold">
                      {!!targetRepo && (
                        <div className="inline-flex cursor-pointer items-center gap-0.5 rounded-md text-xs font-medium">
                          <SourceIcon
                            kind={targetRepo.sourceKind}
                            className="h-3 w-3 shrink-0"
                          />
                          <span className="truncate">
                            {targetRepo.sourceName}
                          </span>
                        </div>
                      )}
                      {docQueryResources?.map(x => {
                        return (
                          <div className="whitespace-nowrap rounded-md px-1.5 py-0.5 font-semibold">
                            {x.sourceName}
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}
                {!!webResources?.length && (
                  <div className="mb-3 mt-2 space-y-1">
                    <div className="text-xs">Reading</div>
                    <div className="flex flex-wrap items-center gap-2 text-xs">
                      {webResources?.map((x, index) => {
                        return (
                          <div key={`${x.link}_${index}`}>
                            <HoverCard openDelay={100} closeDelay={100}>
                              <HoverCardTrigger>
                                <div
                                  className="cursor-pointer whitespace-nowrap rounded-md bg-muted px-1.5 py-0.5 font-semibold"
                                  onClick={() => window.open(x.link)}
                                >
                                  {x.link}
                                </div>
                              </HoverCardTrigger>
                              <HoverCardContent className="w-96 bg-background text-sm text-foreground dark:border-muted-foreground/60">
                                <DocDetailView relevantDocument={x} />
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

function StepItem({
  isLoading,
  children,
  title,
  defaultOpen,
  isLastItem
}: {
  isLoading: boolean | undefined
  children?: ReactNode
  title: string
  defaultOpen?: boolean
  isLastItem?: boolean
}) {
  const itemName = 'item'
  const [open, setOpen] = useState(!!defaultOpen)
  const hasChildren = !!children

  useEffect(() => {
    if (hasChildren && !open) {
      setTimeout(() => {
        setOpen(true)
      }, 0)
    }
  }, [hasChildren])

  return (
    <div>
      <Accordion
        type="single"
        value={open ? itemName : ''}
        collapsible={hasChildren}
        className="z-10"
        onValueChange={v => {
          setOpen(v === itemName)
        }}
      >
        <AccordionItem value={itemName} className="relative border-0">
          {/* vertical separator */}
          {(!isLastItem || (open && hasChildren)) && (
            <div className="absolute left-2 top-5 block h-full w-0.5 shrink-0 translate-x-px rounded-full bg-muted"></div>
          )}
          <AccordionTrigger
            className="group w-full gap-2 rounded-lg py-1 pl-0.5 pr-2 !no-underline hover:bg-muted/70"
            showChevron={!!children}
          >
            <div className="flex flex-1 items-center gap-4">
              <div className="relative z-10 shrink-0 bg-background group-hover:bg-muted/70">
                {isLoading ? (
                  <IconSpinner className="h-4 w-4" />
                ) : (
                  <IconCheckFull className="h-4 w-4" />
                )}
              </div>
              <span>{title}</span>
            </div>
          </AccordionTrigger>
          {!!children && (
            <AccordionContent className="pb-0 pl-9">
              {children}
            </AccordionContent>
          )}
        </AccordionItem>
      </Accordion>
    </div>
  )
}

interface CodeContextItemProps {
  context: RelevantCodeContext
  onContextClick?: (context: RelevantCodeContext) => void
}

// todo clickable highlighted, dev mode tooltip
function CodeContextItem({ context, onContextClick }: CodeContextItemProps) {
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

  return (
    <Tooltip delayDuration={100}>
      <TooltipTrigger asChild>
        <div
          className="cursor-pointer whitespace-nowrap rounded-md bg-muted px-1.5 py-0.5"
          onClick={e => onContextClick?.(context)}
        >
          <span>{fileName}</span>
          {rangeText ? (
            <span className="font-normal text-muted-foreground">
              :{rangeText}
            </span>
          ) : null}
        </div>
      </TooltipTrigger>
      <TooltipContent align="start">
        <div className="whitespace-nowrap">
          <span>{fileName}</span>
          {rangeText ? (
            <span className="text-muted-foreground">:{rangeText}</span>
          ) : null}
          <span className="ml-2 text-xs text-muted-foreground">{path}</span>
        </div>
      </TooltipContent>
    </Tooltip>
  )
}

function DocItem() {
  return <div></div>
}
