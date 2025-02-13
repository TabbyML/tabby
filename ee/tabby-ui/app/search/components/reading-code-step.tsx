'use client'

import { ReactNode, useContext, useEffect, useMemo, useState } from 'react'
import { Maybe } from 'graphql/jsutils/Maybe'
import { isNil } from 'lodash-es'

import { ThreadAssistantMessageReadingCode } from '@/lib/gql/generates/graphql'
import { RelevantCodeContext } from '@/lib/types'
import { isCodeSourceContext, resolveFileNameForDisplay } from '@/lib/utils'
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger
} from '@/components/ui/accordion'
import { IconCheckFull, IconCode, IconSpinner } from '@/components/ui/icons'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'
import { SourceIcon } from '@/components/source-icon'

import { SearchContext } from './search-context'

interface ReadingCodeStepperProps {
  isReadingCode: boolean | undefined
  isReadingFileList: boolean | undefined
  readingCode: ThreadAssistantMessageReadingCode | undefined
  codeSourceId: Maybe<string>
  className?: string
  serverCodeContexts: RelevantCodeContext[]
  clientCodeContexts: RelevantCodeContext[]
  onContextClick?: (
    context: RelevantCodeContext,
    isInWorkspace?: boolean
  ) => void
}

export function ReadingCodeStepper({
  isReadingCode,
  isReadingFileList,
  readingCode,
  codeSourceId,
  serverCodeContexts,
  clientCodeContexts,
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
    let result: Array<'fileList' | 'snippet'> = []
    if (readingCode?.fileList) {
      result.push('fileList')
    }
    if (readingCode?.snippet) {
      result.push('snippet')
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
              <IconCode className="mx-1 h-5 w-5 shrink-0" />
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
                {!!totalContextLength && (
                  <>
                    <div className="mt-1 flex flex-wrap gap-2 text-xs font-semibold">
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
                  </>
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
            <div className="absolute left-3 top-5 block h-full w-0.5 shrink-0 translate-x-px rounded-full bg-muted"></div>
          )}
          <AccordionTrigger
            className="group w-full gap-2 rounded-lg py-1 pl-1.5 pr-2 !no-underline hover:bg-muted/70"
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
            <AccordionContent className="pb-0 pl-10">
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
