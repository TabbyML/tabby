'use client'

import { ReactNode, useContext, useMemo, useState } from 'react'
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
import { IconCheck2, IconCode, IconSpinner } from '@/components/ui/icons'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'
import { SourceIcon } from '@/components/source-icon'

import { SearchContext } from './search-context'

// FIXME for readingCode, should be transform after persisted
interface ReadingCodeStepperProps {
  isReadingCode: boolean | undefined
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
  readingCode,
  codeSourceId,
  serverCodeContexts,
  clientCodeContexts,
  onContextClick,
  className
}: ReadingCodeStepperProps) {
  const { contextInfo } = useContext(SearchContext)
  const [codeSnippetExpand, setCodeSnippetExpand] = useState(true)
  const totalContextLength =
    (clientCodeContexts?.length || 0) + serverCodeContexts.length
  const targetRepo = useMemo(() => {
    if (!codeSourceId) return undefined

    const target = contextInfo?.sources.find(
      x => isCodeSourceContext(x.sourceKind) && x.sourceId === codeSourceId
    )
    return target
  }, [codeSourceId, contextInfo])

  return (
    <Accordion collapsible type="single" defaultValue="readingCode">
      <AccordionItem value="readingCode" className="border-0 mb-4">
        <AccordionTrigger className="w-full py-2">
          <div className="flex flex-1 items-center justify-between pr-2">
            <div className="flex gap-2 items-center flex-1">
              <IconCode className="h-5 w-5 mr-1 ml-1 shrink-0" />
              <span>Reading</span>
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
                title="Reading directory structure"
                isLoading={isReadingCode}
              />
            )}
            {readingCode?.snippet && (
              <StepItem
                title="Search code snippets..."
                isLoading={isReadingCode}
                open={codeSnippetExpand}
                onOpenChange={setCodeSnippetExpand}
              >
                <div className="mb-2 mt-2">Reading</div>
                <div className="flex flex-wrap gap-4 text-xs font-semibold">
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
              </StepItem>
            )}
          </div>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  )
}

// with icon & line
function StepItem({
  isLoading,
  children,
  title,
  open,
  onOpenChange
}: {
  isLoading: boolean | undefined
  children?: ReactNode
  title: string
  open?: boolean
  onOpenChange?: (v: boolean) => void
}) {
  return (
    <div className="relative">
      <Accordion
        type="single"
        defaultValue={open ? 'default' : undefined}
        collapsible={!!children}
        className="z-10"
        onValueChange={v => onOpenChange?.(!!v)}
      >
        <AccordionItem value="default" className="border-0">
          {/* vertical separator */}
          <div className="group-data-[disabled]:bg-muted group-data-[disabled]:opacity-50 absolute left-3 top-5 block h-full w-0.5 shrink-0 rounded-full bg-muted group-data-[state=completed]:bg-primary"></div>
          <AccordionTrigger
            className="group py-1 w-full gap-2 hover:bg-muted/70 !no-underline px-2 rounded-lg"
            showChevron={!!children}
          >
            <div className="flex flex-1 items-center gap-4">
              <div className="bg-background group-hover:bg-muted/70">
                {isLoading ? (
                  <IconSpinner className="w-3 h-3" />
                ) : (
                  <IconCheck2 className="w-3 h-3 bg-foreground/70 rounded-full text-background" />
                )}
              </div>
              <span>{title}</span>
            </div>
          </AccordionTrigger>
          {!!children && (
            <AccordionContent className="pl-9 pb-0">
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
  // console.log(context.range)
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
          className="whitespace-nowrap px-1.5 py-0.5 bg-muted rounded-md cursor-pointer"
          onClick={e => onContextClick?.(context)}
        >
          <span>{fileName}</span>
          {rangeText ? (
            <span className="text-muted-foreground font-normal">
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
