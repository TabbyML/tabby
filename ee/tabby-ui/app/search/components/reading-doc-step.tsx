'use client'

import { useContext, useMemo } from 'react'
import { Maybe } from 'graphql/jsutils/Maybe'

import { ContextSource, ContextSourceKind } from '@/lib/gql/generates/graphql'
import { AttachmentDocItem } from '@/lib/types'
import {
  isAttachmentCommitDoc,
  isAttachmentPageDoc,
  isAttachmentWebDoc,
  normalizedMarkdownText
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
  IconBlocks,
  IconBookOpen,
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
  docQuerySources: Omit<ContextSource, 'id'>[] | undefined
  webDocs?: Maybe<AttachmentDocItem[]> | undefined
  pages?: Maybe<AttachmentDocItem[]> | undefined
}

export function ReadingDocStepper({
  isReadingDocs,
  webDocs,
  docQuerySources,
  pages
}: ReadingDocStepperProps) {
  const { enableDeveloperMode } = useContext(SearchContext)
  const hasMentionDocs =
    !!docQuerySources &&
    docQuerySources.filter(x => x.sourceKind !== ContextSourceKind.Page)
      .length > 0
  const webDocLen = webDocs?.length ?? 0
  const pagesLen = pages?.length ?? 0
  const totalLen = webDocLen + pagesLen

  return (
    <Accordion collapsible type="single" defaultValue="readingCode">
      <AccordionItem value="readingCode" className="border-0">
        <AccordionTrigger className="w-full py-2 pr-2">
          <div className="flex flex-1 items-center justify-between pr-2">
            <div className="flex flex-1 items-center gap-2">
              <IconBlocks className="mr-2 h-5 w-5 shrink-0" />
              <span className="shrink-0">Look into</span>
              <div className="flex flex-1 flex-nowrap gap-2 truncate">
                {docQuerySources?.map(x => {
                  return (
                    <div
                      className="flex items-center gap-1 rounded-lg border px-2 py-0 text-sm font-medium no-underline"
                      key={x.sourceId}
                    >
                      {x.sourceKind === ContextSourceKind.Web ? (
                        <IconEmojiGlobe
                          className="h-3 w-3 shrink-0"
                          emojiClassName="text-sm"
                        />
                      ) : x.sourceKind === ContextSourceKind.Page ? (
                        <IconBookOpen className="h-3.5 w-3.5 shrink-0" />
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
              {totalLen ? (
                <div className="text-sm text-muted-foreground">
                  {totalLen} sources
                </div>
              ) : null}
            </div>
          </div>
        </AccordionTrigger>
        <AccordionContent className="pb-0">
          <div className="space-y-2 text-sm text-muted-foreground">
            <StepItem
              title="Search pages ..."
              isLoading={isReadingDocs}
              defaultOpen={!!pagesLen}
              isLastItem={!hasMentionDocs}
            >
              {!!pages?.length && (
                <div className="mb-3 mt-2">
                  <div className="flex flex-wrap items-center gap-2 text-xs">
                    {pages.map((x, index) => {
                      const _key = isAttachmentCommitDoc(x) ? x.sha : x.link
                      return (
                        <div key={`${_key}_${index}`}>
                          <HoverCard openDelay={100} closeDelay={100}>
                            <HoverCardTrigger>
                              <div
                                className="group cursor-pointer whitespace-nowrap rounded-md bg-muted px-1.5 py-0.5 font-semibold"
                                onClick={() => {
                                  if (_key) {
                                    window.open(_key)
                                  }
                                }}
                              >
                                <PageSummaryView doc={x} />
                              </div>
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
            {hasMentionDocs && (
              <StepItem
                title="Collect documents ..."
                isLastItem
                isLoading={isReadingDocs}
                defaultOpen={!!webDocLen}
              >
                {!!webDocs?.length && (
                  <div className="mb-3 mt-2">
                    <div className="flex flex-wrap items-center gap-2 text-xs">
                      {webDocs.map((x, index) => {
                        const _key = isAttachmentCommitDoc(x) ? x.sha : x.link
                        return (
                          <div key={`${_key}_${index}`}>
                            <HoverCard openDelay={100} closeDelay={100}>
                              <HoverCardTrigger>
                                <div
                                  className="group cursor-pointer whitespace-nowrap rounded-md bg-muted px-1.5 py-0.5 font-semibold"
                                  onClick={() => {
                                    if (_key) {
                                      window.open(_key)
                                    }
                                  }}
                                >
                                  <WebDocSummaryView doc={x} />
                                </div>
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

function WebDocSummaryView({ doc }: { doc: AttachmentDocItem }) {
  const isWebDoc = isAttachmentWebDoc(doc)
  const sourceUrl = useMemo(() => {
    if (!isWebDoc) return null

    try {
      return doc ? new URL(doc.link) : null
    } catch {
      return null
    }
  }, [doc])

  if (!isWebDoc || !sourceUrl) return null

  return (
    <div className="m-0 flex items-center space-x-1 text-xs text-muted-foreground group-hover:text-foreground">
      <SiteFavicon hostname={sourceUrl.hostname} className="m-0 mr-1" />
      <p className="m-0 ">{sourceUrl.hostname}</p>
    </div>
  )
}

function PageSummaryView({ doc }: { doc: AttachmentDocItem }) {
  if (!isAttachmentPageDoc(doc)) return null

  return (
    <div className="m-0 flex items-center space-x-1 text-xs text-muted-foreground group-hover:text-foreground">
      <IconBookOpen className="h-3 w-3" />
      <p>{normalizedMarkdownText(doc.title, 20)}</p>
    </div>
  )
}
