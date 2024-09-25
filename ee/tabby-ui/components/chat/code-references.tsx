import React, { forwardRef, useEffect, useState } from 'react'
import { isNil } from 'lodash-es'

import { RelevantCodeContext } from '@/lib/types'
import { cn } from '@/lib/utils'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger
} from '../ui/accordion'
import { IconExternalLink, IconFile } from '../ui/icons'

interface ContextReferencesProps {
  contexts: RelevantCodeContext[]
  userContexts?: RelevantCodeContext[]
  className?: string
  triggerClassname?: string
  onContextClick?: (
    context: RelevantCodeContext,
    isInWorkspace?: boolean
  ) => void
  enableTooltip?: boolean
  onTooltipClick?: () => void
  highlightIndex?: number | undefined
  showExternalLink: boolean
}

export const CodeReferences = forwardRef<
  HTMLDivElement,
  ContextReferencesProps
>(
  (
    {
      contexts,
      userContexts,
      className,
      triggerClassname,
      onContextClick,
      enableTooltip,
      onTooltipClick,
      highlightIndex,
      showExternalLink
    },
    ref
  ) => {
    const totalContextLength = (userContexts?.length || 0) + contexts.length
    const isMultipleReferences = totalContextLength > 1
    const serverCtxLen = contexts?.length ?? 0
    const clientCtxLen = userContexts?.length ?? 0
    const ctxLen = serverCtxLen + clientCtxLen
    const [accordionValue, setAccordionValue] = useState<string | undefined>(
      ctxLen <= 5 ? 'references' : undefined
    )
    useEffect(() => {
      if (ctxLen <= 5) {
        setAccordionValue('references')
      } else {
        setAccordionValue(undefined)
      }
    }, [ctxLen])

    if (totalContextLength === 0) return null

    return (
      <Accordion
        type="single"
        collapsible
        className={cn('bg-transparent text-foreground', className)}
        ref={ref}
        value={accordionValue}
        onValueChange={setAccordionValue}
      >
        <AccordionItem value="references" className="my-0 border-0">
          <AccordionTrigger
            className={cn('my-0 py-2 font-semibold', triggerClassname)}
          >
            <span className="mr-2">{`Read ${totalContextLength} file${
              isMultipleReferences ? 's' : ''
            }`}</span>
          </AccordionTrigger>
          <AccordionContent className="space-y-2">
            {userContexts?.map((item, index) => {
              return (
                <ContextItem
                  key={`user-${index}`}
                  context={item}
                  onContextClick={ctx => onContextClick?.(ctx, true)}
                  isHighlighted={highlightIndex === index}
                />
              )
            })}
            {contexts.map((item, index) => {
              return (
                <ContextItem
                  key={`assistant-${index}`}
                  context={item}
                  onContextClick={ctx => onContextClick?.(ctx, false)}
                  enableTooltip={enableTooltip}
                  onTooltipClick={onTooltipClick}
                  showExternalLinkIcon={showExternalLink}
                  isHighlighted={
                    highlightIndex === index + (userContexts?.length || 0)
                  }
                />
              )
            })}
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    )
  }
)
CodeReferences.displayName = 'CodeReferences'

function ContextItem({
  context,
  clickable = true,
  onContextClick,
  enableTooltip,
  onTooltipClick,
  showExternalLinkIcon,
  isHighlighted
}: {
  context: RelevantCodeContext
  clickable?: boolean
  onContextClick?: (context: RelevantCodeContext) => void
  enableTooltip?: boolean
  onTooltipClick?: () => void
  showExternalLinkIcon?: boolean
  isHighlighted?: boolean
}) {
  const [tooltipOpen, setTooltipOpen] = useState(false)
  const isMultiLine =
    !isNil(context.range?.start) &&
    !isNil(context.range?.end) &&
    context.range.start < context.range.end
  const pathSegments = context.filepath.split('/')
  const fileName = pathSegments[pathSegments.length - 1]
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
          className={cn('rounded-md border p-2', {
            'cursor-pointer hover:bg-accent': clickable,
            'cursor-default pointer-events-auto': !clickable,
            'bg-accent transition-all': isHighlighted
          })}
          onClick={e => clickable && onContextClick?.(context)}
        >
          <div className="flex items-center gap-1 overflow-hidden">
            <IconFile className="shrink-0" />
            <div className="flex-1 truncate" title={context.filepath}>
              <span>{fileName}</span>
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
              <span className="ml-2 text-xs text-muted-foreground">{path}</span>
            </div>
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
