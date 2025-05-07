'use client'

import { useMemo } from 'react'
import { Maybe } from 'graphql/jsutils/Maybe'

import {
  ContextSource,
  ContextSourceKind,
  ThreadAssistantMessageReadingDoc
} from '@/lib/gql/generates/graphql'
import { AttachmentDocItem } from '@/lib/types'
import {
  isAttachmentIngestedDoc,
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
  IconEmojiGlobe,
  IconFileUp,
  IconFolderUp
} from '@/components/ui/icons'
import { DocDetailView } from '@/components/message-markdown/doc-detail-view'
import { SiteFavicon } from '@/components/site-favicon'

import { StepItem } from './imtermediate-step'

interface ReadingDocStepperProps {
  codeSourceId: Maybe<string>
  readingDoc: ThreadAssistantMessageReadingDoc | undefined
  isReadingDocs: boolean | undefined
  sourceIds?: string[]
  className?: string
  webDocs?:
    | Maybe<
        Array<
          Extract<
            AttachmentDocItem,
            {
              __typename:
                | 'MessageAttachmentWebDoc'
                | 'AttachmentWebDoc'
                | 'AttachmentIngestedDoc'
                | 'MessageAttachmentIngestedDoc'
            }
          >
        >
      >
    | undefined
  pages?:
    | Maybe<
        Array<
          Extract<
            AttachmentDocItem,
            { __typename: 'MessageAttachmentPageDoc' | 'AttachmentPageDoc' }
          >
        >
      >
    | undefined
  docQuerySources: Omit<ContextSource, 'id'>[] | undefined
  openExternal: (url: string) => Promise<void>
}

export function ReadingDocStepper({
  codeSourceId,
  isReadingDocs,
  readingDoc,
  docQuerySources,
  webDocs,
  pages,
  openExternal
}: ReadingDocStepperProps) {
  const webDocLen = webDocs?.length ?? 0
  const pagesLen = pages?.length ?? 0
  const totalLen = webDocLen + pagesLen

  const showPageSubStep =
    readingDoc?.sourceIds.includes('page') || !!pages?.length
  const showWebSubStep =
    (!!readingDoc &&
      readingDoc.sourceIds.filter(x => {
        if (codeSourceId) {
          return x !== 'page' && x !== codeSourceId
        }
        return x !== 'page'
      }).length > 0) ||
    !!webDocs?.length

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
                      className="flex items-center gap-1 rounded-lg border px-2 py-0 font-medium no-underline"
                      key={x.sourceId}
                    >
                      {x.sourceKind === ContextSourceKind.Web ? (
                        <IconEmojiGlobe
                          className="h-3 w-3 shrink-0"
                          emojiClassName="text-sm"
                        />
                      ) : x.sourceKind === ContextSourceKind.Page ? (
                        <IconBookOpen className="h-3.5 w-3.5 shrink-0" />
                      ) : x.sourceKind === ContextSourceKind.Ingested ? (
                        <IconFolderUp className="shrink-0" />
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
            {showPageSubStep && (
              <StepItem
                title="Search pages ..."
                isLoading={isReadingDocs}
                defaultOpen={!!pages?.length}
                isLastItem={!showWebSubStep}
              >
                {!!pages?.length && (
                  <div className="mb-3 mt-2 space-y-2">
                    {pages.map((x, index) => {
                      const _key = x.link
                      return (
                        <div key={`${_key}_${index}`}>
                          <HoverCard openDelay={100} closeDelay={100}>
                            <HoverCardTrigger>
                              <div
                                className="group cursor-pointer pl-2"
                                onClick={() => {
                                  openExternal(x.link)
                                }}
                              >
                                <PageSummaryView doc={x} />
                              </div>
                            </HoverCardTrigger>
                            <HoverCardContent className="w-[60vw] bg-background text-sm text-foreground dark:border-muted-foreground/60 sm:w-96">
                              <DocDetailView
                                relevantDocument={x}
                                onLinkClick={url => {
                                  openExternal(url)
                                }}
                              />
                            </HoverCardContent>
                          </HoverCard>
                        </div>
                      )
                    })}
                  </div>
                )}
              </StepItem>
            )}
            {showWebSubStep && (
              <StepItem
                title="Collect documents ..."
                isLastItem
                isLoading={isReadingDocs}
                defaultOpen={!!webDocs?.length}
              >
                {!!webDocs?.length && (
                  <div className="mb-3 mt-2 space-y-2">
                    {webDocs.map((x, index) => {
                      const link = isAttachmentWebDoc(x)
                        ? x.link
                        : x.ingestedDocLink
                      return (
                        <div key={`${link}_${index}`}>
                          <HoverCard openDelay={100} closeDelay={100}>
                            <HoverCardTrigger>
                              <div
                                className="group cursor-pointer pl-2"
                                onClick={() => {
                                  if (link) {
                                    openExternal(link)
                                  }
                                }}
                              >
                                <DocumentSummaryView doc={x} />
                              </div>
                            </HoverCardTrigger>
                            <HoverCardContent className="w-[60vw] bg-background text-sm text-foreground dark:border-muted-foreground/60 sm:w-96">
                              <DocDetailView
                                relevantDocument={x}
                                onLinkClick={url => {
                                  openExternal(url)
                                }}
                              />
                            </HoverCardContent>
                          </HoverCard>
                        </div>
                      )
                    })}
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

function DocumentSummaryView({ doc }: { doc: AttachmentDocItem }) {
  const isWebDoc = isAttachmentWebDoc(doc)
  const isIngestedDoc = isAttachmentIngestedDoc(doc)

  if (!isWebDoc && !isIngestedDoc) return null

  return (
    <div>
      {isWebDoc && <WebDocSummaryView doc={doc} />}
      {isIngestedDoc && <IngestedDocSummaryView doc={doc} />}
    </div>
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
    <div className="flex flex-nowrap items-center gap-2 rounded-md px-1.5 py-0.5 font-semibold text-foreground hover:bg-accent hover:text-accent-foreground">
      <SiteFavicon hostname={sourceUrl.hostname} className="m-0 shrink-0" />
      <span className="flex-1 truncate text-foreground">{doc.title}</span>
      <span className="m-0 ml-1 shrink-0 text-muted-foreground">
        {sourceUrl.hostname}
      </span>
    </div>
  )
}

function PageSummaryView({ doc }: { doc: AttachmentDocItem }) {
  if (!isAttachmentPageDoc(doc)) return null

  return (
    <div className="flex flex-nowrap items-center gap-2 rounded-md px-1.5 py-0.5 font-semibold text-foreground hover:bg-accent hover:text-accent-foreground">
      <IconBookOpen className="h-3.5 w-3.5 shrink-0" />
      <span className="flex-1 truncate text-foreground">
        {normalizedMarkdownText(doc.title)}
      </span>
    </div>
  )
}

function IngestedDocSummaryView({ doc }: { doc: AttachmentDocItem }) {
  const isIngestedDoc = isAttachmentIngestedDoc(doc)

  if (!isIngestedDoc) return null

  return (
    <div className="flex flex-nowrap items-center gap-2 rounded-md px-1.5 py-0.5 font-semibold text-foreground hover:bg-accent hover:text-accent-foreground">
      <IconFileUp className="h-3.5 w-3.5 shrink-0" />
      <p>{normalizedMarkdownText(doc.title, 20)}</p>
    </div>
  )
}
