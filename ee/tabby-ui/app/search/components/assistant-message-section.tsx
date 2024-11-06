'use client'

import { MouseEventHandler, useContext, useMemo, useState } from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import DOMPurify from 'dompurify'
import he from 'he'
import { compact, isEmpty } from 'lodash-es'
import { marked } from 'marked'
import { useForm } from 'react-hook-form'
import Textarea from 'react-textarea-autosize'
import { Context } from 'tabby-chat-panel/index'
import * as z from 'zod'

import { MARKDOWN_CITATION_REGEX } from '@/lib/constants/regex'
import { MessageAttachmentCode } from '@/lib/gql/generates/graphql'
import { useEnterSubmit } from '@/lib/hooks/use-enter-submit'
import { AttachmentDocItem, RelevantCodeContext } from '@/lib/types'
import {
  cn,
  formatLineHashForCodeBrowser,
  getRangeFromAttachmentCode,
  getRangeTextFromAttachmentCode
} from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage
} from '@/components/ui/form'
import {
  IconBlocks,
  IconBug,
  IconChevronRight,
  IconEdit,
  IconLayers,
  IconPlus,
  IconRefresh,
  IconRemove,
  IconSparkles,
  IconSpinner,
  IconTrash
} from '@/components/ui/icons'
import { Skeleton } from '@/components/ui/skeleton'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'
import { CodeReferences } from '@/components/chat/code-references'
import { CopyButton } from '@/components/copy-button'
import {
  ErrorMessageBlock,
  MessageMarkdown,
  SiteFavicon
} from '@/components/message-markdown'

import { ConversationMessage, SearchContext, SOURCE_CARD_STYLE } from './search'

export function AssistantMessageSection({
  message,
  showRelatedQuestion,
  isLoading,
  isLastAssistantMessage,
  isDeletable,
  className
}: {
  message: ConversationMessage
  showRelatedQuestion: boolean
  isLoading?: boolean
  isLastAssistantMessage?: boolean
  isDeletable?: boolean
  className?: string
}) {
  const {
    onRegenerateResponse,
    onSubmitSearch,
    setDevPanelOpen,
    setConversationIdForDev,
    enableDeveloperMode,
    contextInfo,
    fetchingContextInfo,
    onDeleteMessage,
    isThreadOwner,
    onUpdateMessage
  } = useContext(SearchContext)

  const [isEditing, setIsEditing] = useState(false)
  const [showMoreSource, setShowMoreSource] = useState(false)
  const [relevantCodeHighlightIndex, setRelevantCodeHighlightIndex] = useState<
    number | undefined
  >(undefined)
  const getCopyContent = (answer: ConversationMessage) => {
    if (isEmpty(answer?.attachment?.doc) && isEmpty(answer?.attachment?.code)) {
      return answer.content
    }

    const content = answer.content
      .replace(MARKDOWN_CITATION_REGEX, match => {
        const citationNumberMatch = match?.match(/\d+/)
        return `[${citationNumberMatch}]`
      })
      .trim()
    const docCitations =
      answer.attachment?.doc
        ?.map((doc, idx) => `[${idx + 1}] ${doc.link}`)
        .join('\n') ?? ''
    const docCitationLen = answer.attachment?.doc?.length ?? 0
    const codeCitations =
      answer.attachment?.code
        ?.map((code, idx) => {
          const lineRangeText = getRangeTextFromAttachmentCode(code)
          const filenameText = compact([code.filepath, lineRangeText]).join(':')
          return `[${idx + docCitationLen + 1}] ${filenameText}`
        })
        .join('\n') ?? ''
    const citations = docCitations + codeCitations

    return `${content}\n\nCitations:\n${citations}`
  }

  const IconAnswer = isLoading ? IconSpinner : IconSparkles

  const messageAttachmentDocs = message?.attachment?.doc
  const messageAttachmentCode = message?.attachment?.code

  const totalHeightInRem = messageAttachmentDocs?.length
    ? Math.ceil(messageAttachmentDocs.length / 4) * SOURCE_CARD_STYLE.expand +
      0.5 * Math.floor(messageAttachmentDocs.length / 4) +
      0.5
    : 0

  const relevantCodeContexts: RelevantCodeContext[] = useMemo(() => {
    return (
      message?.attachment?.code?.map(code => {
        const { startLine, endLine } = getRangeFromAttachmentCode(code)

        return {
          kind: 'file',
          range: {
            start: startLine,
            end: endLine
          },
          filepath: code.filepath,
          content: code.content,
          git_url: code.gitUrl,
          extra: {
            scores: code?.extra?.scores
          }
        }
      }) ?? []
    )
  }, [message?.attachment?.code])

  const onCodeContextClick = (ctx: Context) => {
    if (!ctx.filepath) return
    const url = new URL(`${window.location.origin}/files`)
    const searchParams = new URLSearchParams()
    searchParams.append('redirect_filepath', ctx.filepath)
    searchParams.append('redirect_git_url', ctx.git_url)
    url.search = searchParams.toString()

    const lineHash = formatLineHashForCodeBrowser({
      start: ctx.range.start,
      end: ctx.range.end
    })
    if (lineHash) {
      url.hash = lineHash
    }

    window.open(url.toString())
  }

  const onCodeCitationMouseEnter = (index: number) => {
    setRelevantCodeHighlightIndex(
      index - 1 - (message?.attachment?.doc?.length || 0)
    )
  }

  const onCodeCitationMouseLeave = (index: number) => {
    setRelevantCodeHighlightIndex(undefined)
  }

  const openCodeBrowserTab = (code: MessageAttachmentCode) => {
    const { startLine, endLine } = getRangeFromAttachmentCode(code)

    if (!code.filepath) return
    const url = new URL(`${window.location.origin}/files`)
    const searchParams = new URLSearchParams()
    searchParams.append('redirect_filepath', code.filepath)
    searchParams.append('redirect_git_url', code.gitUrl)
    url.search = searchParams.toString()

    const lineHash = formatLineHashForCodeBrowser({
      start: startLine,
      end: endLine
    })
    if (lineHash) {
      url.hash = lineHash
    }

    window.open(url.toString())
  }

  const onCodeCitationClick = (code: MessageAttachmentCode) => {
    openCodeBrowserTab(code)
  }

  const handleUpdateAssistantMessage = async (message: ConversationMessage) => {
    const errorMessage = await onUpdateMessage(message)
    if (errorMessage) {
      return errorMessage
    } else {
      setIsEditing(false)
    }
  }

  return (
    <div className={cn('flex flex-col gap-y-5', className)}>
      {/* document search hits */}
      {messageAttachmentDocs && messageAttachmentDocs.length > 0 && (
        <div>
          <div className="mb-1 flex items-center gap-x-2">
            <IconBlocks className="relative" style={{ top: '-0.04rem' }} />
            <p className="text-sm font-bold leading-normal">Sources</p>
          </div>
          <div
            className="gap-sm -mx-2 grid grid-cols-3 gap-2 overflow-y-hidden px-2 pt-2 md:grid-cols-4"
            style={{
              transition: 'height 0.25s ease-out',
              height: showMoreSource
                ? `${totalHeightInRem}rem`
                : `${SOURCE_CARD_STYLE.compress + 0.5}rem`
            }}
          >
            {messageAttachmentDocs.map((source, index) => (
              <SourceCard
                key={source.link + index}
                conversationId={message.id}
                source={source}
                showMore={showMoreSource}
                showDevTooltip={enableDeveloperMode}
              />
            ))}
          </div>
          <Button
            variant="ghost"
            className="-ml-1.5 mt-1 flex items-center gap-x-1 px-1 py-2 text-sm font-normal text-muted-foreground"
            onClick={() => setShowMoreSource(!showMoreSource)}
          >
            <IconChevronRight
              className={cn({
                '-rotate-90': showMoreSource,
                'rotate-90': !showMoreSource
              })}
            />
            <p>{showMoreSource ? 'Show less' : 'Show more'}</p>
          </Button>
        </div>
      )}

      {/* Answer content */}
      <div>
        <div className="mb-1 flex h-8 items-center gap-x-1.5">
          <IconAnswer
            className={cn({
              'animate-spinner': isLoading
            })}
          />
          <p className="text-sm font-bold leading-none">Answer</p>
          {enableDeveloperMode && (
            <Button
              variant="ghost"
              size="icon"
              onClick={() => {
                setConversationIdForDev(message.id)
                setDevPanelOpen(true)
              }}
            >
              <IconBug />
            </Button>
          )}
        </div>

        {/* code search hits */}
        {messageAttachmentCode && messageAttachmentCode.length > 0 && (
          <CodeReferences
            contexts={relevantCodeContexts}
            className="mt-1 text-sm"
            onContextClick={onCodeContextClick}
            enableTooltip={enableDeveloperMode}
            showExternalLink={false}
            onTooltipClick={() => {
              setConversationIdForDev(message.id)
              setDevPanelOpen(true)
            }}
            highlightIndex={relevantCodeHighlightIndex}
          />
        )}

        {isLoading && !message.content && (
          <Skeleton className="mt-1 h-40 w-full" />
        )}
        {isEditing ? (
          <MessageContentForm
            message={message}
            onCancel={() => setIsEditing(false)}
            onSubmit={handleUpdateAssistantMessage}
          />
        ) : (
          <>
            <MessageMarkdown
              message={message.content}
              attachmentDocs={messageAttachmentDocs}
              attachmentCode={messageAttachmentCode}
              onCodeCitationClick={onCodeCitationClick}
              onCodeCitationMouseEnter={onCodeCitationMouseEnter}
              onCodeCitationMouseLeave={onCodeCitationMouseLeave}
              contextInfo={contextInfo}
              fetchingContextInfo={fetchingContextInfo}
              canWrapLongLines={!isLoading}
            />
            {/* if isEditing, do not display error message block */}
            {message.error && <ErrorMessageBlock error={message.error} />}

            {!isLoading && !isEditing && (
              <div className="mt-3 flex items-center justify-between text-sm">
                <div className="flex items-center gap-x-3">
                  {isThreadOwner && (
                    <>
                      {!isLoading &&
                        !fetchingContextInfo &&
                        isLastAssistantMessage && (
                          <Button
                            className="flex items-center gap-x-1 px-1 font-normal text-muted-foreground"
                            variant="ghost"
                            onClick={() => onRegenerateResponse(message.id)}
                          >
                            <IconRefresh />
                            <p>Regenerate</p>
                          </Button>
                        )}

                      {isDeletable && (
                        <Button
                          className="flex items-center gap-x-1 px-1 font-normal text-muted-foreground"
                          variant="ghost"
                          onClick={() => onDeleteMessage(message.id)}
                        >
                          <IconTrash />
                          <p>Delete</p>
                        </Button>
                      )}
                    </>
                  )}
                </div>
                <div className="flex items-center gap-x-3">
                  <CopyButton
                    className="-ml-1.5 gap-x-1 px-1 font-normal text-muted-foreground"
                    value={getCopyContent(message)}
                    text="Copy"
                  />
                  {isThreadOwner && (
                    <Button
                      className="flex items-center gap-x-1 px-1 font-normal text-muted-foreground"
                      variant="ghost"
                      onClick={e => setIsEditing(true)}
                    >
                      <IconEdit />
                      <p>Edit</p>
                    </Button>
                  )}
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* Related questions */}
      {showRelatedQuestion &&
        !isEditing &&
        !isLoading &&
        message.threadRelevantQuestions &&
        message.threadRelevantQuestions.length > 0 && (
          <div>
            <div className="flex items-center gap-x-1.5">
              <IconLayers />
              <p className="text-sm font-bold leading-none">Suggestions</p>
            </div>
            <div className="mt-2 flex flex-col gap-y-3">
              {message.threadRelevantQuestions?.map(
                (relevantQuestion, index) => (
                  <div
                    key={index}
                    className="flex cursor-pointer items-center justify-between rounded-lg border p-4 py-3 transition-opacity hover:opacity-70"
                    onClick={onSubmitSearch.bind(null, relevantQuestion)}
                  >
                    <p className="w-full overflow-hidden text-ellipsis text-sm">
                      {relevantQuestion}
                    </p>
                    <IconPlus />
                  </div>
                )
              )}
            </div>
          </div>
        )}
    </div>
  )
}

function SourceCard({
  conversationId,
  source,
  showMore,
  showDevTooltip,
  isDeletable,
  onDelete
}: {
  conversationId: string
  source: AttachmentDocItem
  showMore: boolean
  showDevTooltip?: boolean
  isDeletable?: boolean
  onDelete?: () => void
}) {
  const { setDevPanelOpen, setConversationIdForDev } = useContext(SearchContext)
  const { hostname } = new URL(source.link)
  const [devTooltipOpen, setDevTooltipOpen] = useState(false)

  const onOpenChange = (v: boolean) => {
    if (!showDevTooltip) return
    setDevTooltipOpen(v)
  }

  const onTootipClick: MouseEventHandler<HTMLDivElement> = e => {
    e.stopPropagation()
    setConversationIdForDev(conversationId)
    setDevPanelOpen(true)
  }

  return (
    <Tooltip
      open={devTooltipOpen}
      onOpenChange={onOpenChange}
      delayDuration={0}
    >
      <TooltipTrigger asChild>
        <div
          className="relative flex cursor-pointer flex-col justify-between rounded-lg border bg-card p-3 hover:bg-card/60"
          style={{
            height: showMore
              ? `${SOURCE_CARD_STYLE.expand}rem`
              : `${SOURCE_CARD_STYLE.compress}rem`,
            transition: 'all 0.25s ease-out'
          }}
          onClick={() => window.open(source.link)}
        >
          {isDeletable && (
            <div className="absolute -right-1.5 -top-2">
              <Button
                size="icon"
                variant="secondary"
                className="h-4 w-4 rounded-full border"
                onClick={e => {
                  e.stopPropagation()
                  onDelete?.()
                }}
              >
                <IconRemove className="h-3 w-3" />
              </Button>
            </div>
          )}
          <div className="flex flex-1 flex-col justify-between gap-y-1">
            <div className="flex flex-col gap-y-0.5">
              <p className="line-clamp-1 w-full overflow-hidden text-ellipsis break-all text-xs font-semibold">
                {source.title}
              </p>
              <p
                className={cn(
                  ' w-full overflow-hidden text-ellipsis break-all text-xs text-muted-foreground',
                  {
                    'line-clamp-2': showMore,
                    'line-clamp-1': !showMore
                  }
                )}
              >
                {normalizedText(source.content)}
              </p>
            </div>
            <div className="flex items-center text-xs text-muted-foreground">
              <div className="flex w-full flex-1 items-center">
                <SiteFavicon hostname={hostname} />
                <p className="ml-1 overflow-hidden text-ellipsis">
                  {hostname.replace('www.', '').split('/')[0]}
                </p>
              </div>
            </div>
          </div>
        </div>
      </TooltipTrigger>
      <TooltipContent
        align="start"
        className="cursor-pointer p-2"
        onClick={onTootipClick}
      >
        <p>Score: {source?.extra?.score ?? '-'}</p>
      </TooltipContent>
    </Tooltip>
  )
}

function MessageContentForm({
  message,
  onCancel,
  onSubmit
}: {
  message: ConversationMessage
  onCancel: () => void
  onSubmit: (newMessage: ConversationMessage) => Promise<string | void>
}) {
  const formSchema = z.object({
    content: z.string().trim()
  })
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: { content: message.content }
  })
  const { isSubmitting } = form.formState
  const { content } = form.watch()
  const isEmptyContent = !content || isEmpty(content.trim())
  const [draftMessage] = useState<ConversationMessage>(message)
  const { formRef, onKeyDown } = useEnterSubmit()

  const handleSubmit = async (values: z.infer<typeof formSchema>) => {
    const errorMessage = await onSubmit({
      ...draftMessage,
      content: values.content
    })
    if (errorMessage) {
      form.setError('root', { message: errorMessage })
    }
  }

  return (
    <Form {...form}>
      <form ref={formRef} onSubmit={form.handleSubmit(handleSubmit)}>
        <FormField
          control={form.control}
          name="content"
          render={({ field }) => (
            <FormItem>
              <FormControl>
                <Textarea
                  autoFocus
                  minRows={2}
                  maxRows={20}
                  className="w-full rounded-lg border bg-background p-4 outline-ring"
                  onKeyDown={onKeyDown}
                  {...field}
                />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <div className="my-4 flex items-center justify-between gap-2 px-2">
          <div>
            <FormMessage />
          </div>
          <div className="flex items-center gap-2">
            <Button
              type="button"
              variant="outline"
              onClick={onCancel}
              className="min-w-[2rem]"
            >
              Cancel
            </Button>
            <Button type="submit" disabled={isEmptyContent || isSubmitting}>
              {isSubmitting && (
                <IconSpinner className="mr-2 h-4 w-4 animate-spin" />
              )}
              Save
            </Button>
          </div>
        </div>
      </form>
    </Form>
  )
}

// Remove HTML and Markdown format
const normalizedText = (input: string) => {
  const sanitizedHtml = DOMPurify.sanitize(input, {
    ALLOWED_TAGS: [],
    ALLOWED_ATTR: []
  })
  const parsed = marked.parse(sanitizedHtml) as string
  const decoded = he.decode(parsed)
  const plainText = decoded.replace(/<\/?[^>]+(>|$)/g, '')
  return plainText
}
