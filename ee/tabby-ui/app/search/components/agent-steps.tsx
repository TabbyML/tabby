import { useContext, useMemo } from 'react'

import { ThreadAssistantMessageReadingCode } from '@/lib/gql/generates/graphql'
import { cn, isCodeSourceContext } from '@/lib/utils'
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger
} from '@/components/ui/accordion'
import { IconCode, IconListTree, IconSpinner } from '@/components/ui/icons'
import { SourceIcon } from '@/components/source-icon'

import { SearchContext } from './search-context'

interface AgentStepsProps {
  steps?: ThreadAssistantMessageReadingCode
  isReadingCode?: boolean
  codeSourceId?: string | null
  className?: string
}

export function AgentSteps({
  steps,
  className,
  codeSourceId,
  isReadingCode
}: AgentStepsProps) {
  const { contextInfo } = useContext(SearchContext)
  const targetRepo = useMemo(() => {
    const target = contextInfo?.sources.find(
      x => isCodeSourceContext(x.sourceKind) && x.sourceId === codeSourceId
    )
    return target
  }, [codeSourceId, contextInfo])

  return (
    <Accordion
      type="single"
      collapsible
      className={cn('w-full border rounded-lg px-4 my-1.5', className)}
    >
      <AccordionItem value="item" className="border-0">
        <AccordionTrigger className="w-full">
          <div className="font-semibold">Tabby Agent</div>
        </AccordionTrigger>
        <AccordionContent className="space-y-3">
          {steps?.fileList && (
            <div className="flex items-start gap-1">
              <span className="shrink-0 mt-0.5">
                {isReadingCode ? (
                  <IconSpinner className="h-5 w-5" />
                ) : (
                  <IconListTree className="h-5 w-5" />
                )}
              </span>
              <div className="space-y-1">
                <div className="font-medium text-base">Read file list</div>
                {targetRepo ? (
                  <>
                    <div className="inline-flex items-center gap-1 bg-muted rounded-md px-1 py-0.5 cursor-pointer text-secondary-foreground hover:bg-muted/70 font-medium">
                      <SourceIcon
                        kind={targetRepo.sourceKind}
                        className="h-3.5 w-3.5 shrink-0"
                      />
                      <span className="truncate text-sm">
                        {targetRepo.sourceName}
                      </span>
                    </div>
                  </>
                ) : null}
              </div>
            </div>
          )}
          {steps?.snippet && (
            <div className="flex items-start gap-1">
              <span className="shrink-0 mt-0.5">
                {isReadingCode ? (
                  <IconSpinner className="h-5 w-5" />
                ) : (
                  <IconCode className="h-5 w-5" />
                )}
              </span>
              <div className="space-y-1">
                <div className="font-medium text-base">Read code snippets</div>
                {targetRepo ? (
                  <>
                    <div className="inline-flex items-center gap-1 bg-muted rounded-md px-1 py-0.5 cursor-pointer text-secondary-foreground hover:bg-muted/70 font-medium">
                      <SourceIcon
                        kind={targetRepo.sourceKind}
                        className="h-3.5 w-3.5 shrink-0"
                      />
                      <span className="truncate text-sm">
                        {targetRepo.sourceName}
                      </span>
                    </div>
                  </>
                ) : null}
              </div>
            </div>
          )}
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  )
}
