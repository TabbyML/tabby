'use client'

import { useContext } from 'react'
import { Maybe } from 'graphql/jsutils/Maybe'

import { ContextSource, ContextSourceKind } from '@/lib/gql/generates/graphql'
import { AttachmentDocItem } from '@/lib/types'
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
  IconBlocks,
  IconEmojiBook,
  IconEmojiGlobe
} from '@/components/ui/icons'
import { DocDetailView } from '@/components/message-markdown/doc-detail-view'
import { SiteFavicon } from '@/components/site-favicon'

import { StepItem } from './intermediate-step'
import { SearchContext } from './search-context'

interface ReadingDocStepperProps {
  isReadingDocs: boolean | undefined
  sourceIds?: string[]
  className?: string
  webResources?: Maybe<AttachmentDocItem[]> | undefined
  docQueryResources: Omit<ContextSource, 'id'>[] | undefined
}

export function ReadingDocStepper({
  isReadingDocs,
  webResources,
  docQueryResources
}: ReadingDocStepperProps) {
  const resultLen = webResources?.length
  const { enableDeveloperMode } = useContext(SearchContext)

  return (
    <Accordion collapsible type="single" defaultValue="readingCode">
      <AccordionItem value="readingCode" className="border-0">
        <AccordionTrigger className="w-full py-2 pr-2">
          <div className="flex flex-1 items-center justify-between pr-2">
            <div className="flex flex-1 items-center gap-2">
              <IconBlocks className="mr-2 h-5 w-5 shrink-0" />
              <span className="shrink-0">Look into</span>
              <div className="flex flex-1 flex-nowrap gap-2 truncate">
                {docQueryResources?.map(x => {
                  return (
                    <div
                      className="flex items-center gap-1 rounded-lg border px-2 py-0 font-medium no-underline"
                      key={x.sourceId}
                    >
                      {x.sourceKind === ContextSourceKind.Web ? (
                        <IconEmojiGlobe
                          className="h-3 w-3 shrink-0"
                          emojiClassName="text-sm"
                        />
                      ) : (
                        <IconEmojiBook
                          className="h-3 w-3 shrink-0"
                          emojiClassName="text-sm"
                        />
                      )}
                      {x.sourceName}
                    </div>
                  )
                })}
              </div>
            </div>
            <div className="shrink-0">
              {resultLen ? (
                <div className="text-sm text-muted-foreground">
                  {resultLen} sources
                </div>
              ) : null}
            </div>
          </div>
        </AccordionTrigger>
        <AccordionContent className="pb-0">
          <div className="space-y-2 text-sm text-muted-foreground">
            <StepItem
              title="Search for relevant web docs ..."
              isLastItem
              isLoading={isReadingDocs}
            >
              {!!webResources?.length && (
                <div className="mb-3 mt-2">
                  <div className="flex flex-wrap items-center gap-2 text-xs">
                    {webResources.map((x, index) => {
                      return (
                        <div key={`${x.link}_${index}`}>
                          <HoverCard openDelay={100} closeDelay={100}>
                            <HoverCardTrigger>
                              <div
                                className="group cursor-pointer whitespace-nowrap rounded-md bg-muted px-1.5 py-0.5 font-semibold"
                                onClick={() => window.open(x.link)}
                              >
                                <DocItem doc={x} />
                              </div>
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
          </div>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  )
}

function DocItem({ doc }: { doc: AttachmentDocItem }) {
  if (doc.__typename !== 'MessageAttachmentWebDoc') return null

  const sourceUrl = doc ? new URL(doc.link) : null

  return (
    <div className="m-0 flex items-center space-x-1 text-xs leading-none text-muted-foreground group-hover:text-foreground">
      <SiteFavicon
        hostname={sourceUrl!.hostname}
        className="m-0 mr-1 leading-none"
      />
      <p className="m-0 leading-none">{sourceUrl!.hostname}</p>
    </div>
  )
}
