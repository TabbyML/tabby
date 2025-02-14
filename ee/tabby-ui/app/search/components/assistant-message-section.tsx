'use client'

import { MouseEventHandler, useContext, useMemo, useState } from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import DOMPurify from 'dompurify'
import he from 'he'
import { compact, isEmpty } from 'lodash-es'
import { marked } from 'marked'
import { useForm } from 'react-hook-form'
import Textarea from 'react-textarea-autosize'
import * as z from 'zod'

import { MARKDOWN_CITATION_REGEX } from '@/lib/constants/regex'
import {
  ContextSource,
  Maybe,
  MessageAttachmentClientCode,
  MessageAttachmentCode
} from '@/lib/gql/generates/graphql'
import { makeFormErrorHandler } from '@/lib/tabby/gql'
import {
  AttachmentDocItem,
  Context,
  ExtendedCombinedError,
  RelevantCodeContext
} from '@/lib/types'
import {
  buildCodeBrowserUrlForContext,
  cn,
  formatLineHashForCodeBrowser,
  getContent,
  getMentionsFromText,
  getRangeFromAttachmentCode,
  getRangeTextFromAttachmentCode,
  isDocSourceContext
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
  HoverCard,
  HoverCardContent,
  HoverCardTrigger
} from '@/components/ui/hover-card'
import {
  IconBug,
  IconCheckCircled,
  IconCircleDot,
  IconEdit,
  IconGitMerge,
  IconGitPullRequest,
  IconLayers,
  IconPlus,
  IconRefresh,
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
import { ChatContext } from '@/components/chat/chat'
import { CopyButton } from '@/components/copy-button'
import {
  ErrorMessageBlock,
  MessageMarkdown
} from '@/components/message-markdown'
import { DocDetailView } from '@/components/message-markdown/doc-detail-view'
import { SiteFavicon } from '@/components/site-favicon'
import { UserAvatar } from '@/components/user-avatar'

import { ReadingCodeStepper } from './reading-code-step'
import { ReadingDocStepper } from './reading-doc-step'
import { SOURCE_CARD_STYLE } from './search'
import { SearchContext } from './search-context'
import { ConversationMessage } from './types'

export function AssistantMessageSection({
  className,
  message,
  userMessage,
  showRelatedQuestion,
  isLoading,
  isLastAssistantMessage,
  isDeletable,
  clientCode
}: {
  className?: string
  message: ConversationMessage
  userMessage: ConversationMessage
  showRelatedQuestion: boolean
  isLoading?: boolean
  isLastAssistantMessage?: boolean
  isDeletable?: boolean
  clientCode?: Maybe<Array<MessageAttachmentClientCode>>
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
    onUpdateMessage,
    repositories
  } = useContext(SearchContext)

  const { supportsOnApplyInEditorV2 } = useContext(ChatContext)

  const docSources: Array<Omit<ContextSource, 'id'>> = useMemo(() => {
    if (!contextInfo?.sources || !userMessage?.content) return []

    const _sources = getMentionsFromText(
      userMessage.content,
      contextInfo?.sources
    )
    return _sources
      .filter(x => isDocSourceContext(x.kind))
      .map(x => ({
        sourceId: x.id,
        sourceKind: x.kind,
        sourceName: x.label
      }))
  }, [contextInfo?.sources, userMessage?.content])

  const [isEditing, setIsEditing] = useState(false)
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

  // match gitUrl for clientCode with codeSourceId
  const clientCodeGitUrl = useMemo(() => {
    if (!message.codeSourceId || !repositories?.length) return ''

    const target = repositories.find(
      info => info.sourceId === message.codeSourceId
    )
    return target?.gitUrl ?? ''
  }, [message.codeSourceId, repositories])

  const clientCodeContexts: RelevantCodeContext[] = useMemo(() => {
    if (!clientCode?.length) return []
    return (
      clientCode.map(code => {
        return {
          kind: 'file',
          range: getRangeFromAttachmentCode(code),
          filepath: code.filepath || '',
          content: code.content,
          git_url: clientCodeGitUrl
        }
      }) ?? []
    )
  }, [clientCode, clientCodeGitUrl])

  const serverCodeContexts: RelevantCodeContext[] = useMemo(() => {
    return (
      message?.attachment?.code?.map(code => {
        return {
          kind: 'file',
          range: getRangeFromAttachmentCode(code),
          filepath: code.filepath,
          content: code.content,
          git_url: code.gitUrl,
          commit: code.commit ?? undefined,
          extra: {
            scores: code?.extra?.scores
          }
        }
      }) ?? []
    )
  }, [message?.attachment?.code])

  const messageAttachmentClientCode = useMemo(() => {
    return clientCode?.map(o => ({
      ...o,
      gitUrl: clientCodeGitUrl
    }))
  }, [clientCode, clientCodeGitUrl])

  const messageAttachmentDocs = message?.attachment?.doc
  const messageAttachmentCodeLen =
    (messageAttachmentClientCode?.length || 0) +
    (message.attachment?.code?.length || 0)

  const issuesAndPRs = useMemo(() => {
    return messageAttachmentDocs?.filter(
      x =>
        x.__typename === 'MessageAttachmentIssueDoc' ||
        x.__typename === 'MessageAttachmentPullDoc'
    )
  }, [messageAttachmentDocs])

  const webDocs = useMemo(() => {
    return messageAttachmentDocs?.filter(
      x => x.__typename === 'MessageAttachmentWebDoc'
    )
  }, [messageAttachmentDocs])

  const onCodeContextClick = (ctx: Context) => {
    if (!ctx.filepath) return
    const url = buildCodeBrowserUrlForContext(window.location.origin, ctx)
    window.open(url, '_blank')
  }

  const openCodeBrowserTab = (code: MessageAttachmentCode) => {
    const range = getRangeFromAttachmentCode(code)

    if (!code.filepath) return
    const url = new URL(`${window.location.origin}/files`)
    const searchParams = new URLSearchParams()
    searchParams.append('redirect_filepath', code.filepath)
    searchParams.append('redirect_git_url', code.gitUrl)
    if (code.commit) {
      searchParams.append('redirect_rev', code.commit)
    }
    url.search = searchParams.toString()

    const lineHash = formatLineHashForCodeBrowser(range)
    if (lineHash) {
      url.hash = lineHash
    }

    window.open(url.toString())
  }

  const onCodeCitationClick = (code: MessageAttachmentCode) => {
    if (code.gitUrl) {
      openCodeBrowserTab(code)
    }
  }

  const handleUpdateAssistantMessage = async (message: ConversationMessage) => {
    const error = await onUpdateMessage(message)
    if (error) {
      return error
    } else {
      setIsEditing(false)
    }
  }

  const showFileListStep =
    !!message.readingCode?.fileList ||
    !!message.attachment?.codeFileList?.fileList?.length
  const showCodeSnippetsStep =
    message.readingCode?.snippet || !!messageAttachmentCodeLen

  const showReadingCodeStep = !!message.codeSourceId
  const showReadingDocStep = !!docSources?.length

  return (
    <div className={cn('flex flex-col gap-y-5', className)}>
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

        {(showReadingCodeStep || showReadingDocStep) && (
          <div className="mb-6 space-y-1.5">
            {showReadingCodeStep && (
              <ReadingCodeStepper
                clientCodeContexts={clientCodeContexts}
                serverCodeContexts={serverCodeContexts}
                isReadingFileList={message.isReadingFileList}
                isReadingCode={message.isReadingCode}
                isReadingDocs={message.isReadingDocs}
                codeSourceId={message.codeSourceId}
                docQuery
                docQueryResources={docSources}
                webResources={issuesAndPRs}
                readingCode={{
                  fileList: showFileListStep,
                  snippet: showCodeSnippetsStep
                }}
                onContextClick={onCodeContextClick}
              />
            )}
            {showReadingDocStep && (
              <ReadingDocStepper
                docQueryResources={docSources}
                isReadingDocs={message.isReadingDocs}
                webResources={webDocs}
              />
            )}
          </div>
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
              attachmentClientCode={messageAttachmentClientCode}
              attachmentCode={message.attachment?.code}
              onCodeCitationClick={onCodeCitationClick}
              contextInfo={contextInfo}
              fetchingContextInfo={fetchingContextInfo}
              canWrapLongLines={!isLoading}
              supportsOnApplyInEditorV2={supportsOnApplyInEditorV2}
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
  showDevTooltip
}: {
  conversationId: string
  source: AttachmentDocItem
  showMore: boolean
  showDevTooltip?: boolean
  isDeletable?: boolean
  onDelete?: () => void
}) {
  const { setDevPanelOpen, setConversationIdForDev } = useContext(SearchContext)
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
    <HoverCard openDelay={100} closeDelay={100}>
      <Tooltip
        open={devTooltipOpen}
        onOpenChange={onOpenChange}
        delayDuration={0}
      >
        <HoverCardTrigger asChild>
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
              <SourceCardContent source={source} showMore={showMore} />
            </div>
          </TooltipTrigger>
        </HoverCardTrigger>
        <TooltipContent
          align="start"
          className="cursor-pointer p-2"
          onClick={onTootipClick}
        >
          <p>Score: {source?.extra?.score ?? '-'}</p>
        </TooltipContent>
      </Tooltip>
      <HoverCardContent className="w-96 bg-background text-sm text-foreground dark:border-muted-foreground/60">
        <DocDetailView relevantDocument={source} />
      </HoverCardContent>
    </HoverCard>
  )
}

function SourceCardContent({
  source,
  showMore
}: {
  source: AttachmentDocItem
  showMore: boolean
}) {
  const { hostname } = new URL(source.link)

  const isIssue = source.__typename === 'MessageAttachmentIssueDoc'
  const isPR = source.__typename === 'MessageAttachmentPullDoc'
  const author =
    source.__typename === 'MessageAttachmentWebDoc' ? undefined : source.author

  const showAvatar = (isIssue || isPR) && !!author

  return (
    <div className="flex flex-1 flex-col justify-between gap-y-1">
      <div className="flex flex-col gap-y-0.5">
        <p className="line-clamp-1 w-full overflow-hidden text-ellipsis break-all text-xs font-semibold">
          {source.title}
        </p>

        {showAvatar && (
          <div className="flex items-center gap-1 overflow-x-hidden">
            <UserAvatar user={author} className="h-3.5 w-3.5 shrink-0" />
            <p className="truncate text-xs font-medium text-muted-foreground">
              {author?.name}
            </p>
          </div>
        )}
        {(!showAvatar || showMore) && (
          <p
            className={cn(
              ' w-full overflow-hidden text-ellipsis break-all text-xs text-muted-foreground',
              !showAvatar && showMore ? 'line-clamp-2' : 'line-clamp-1'
            )}
          >
            {normalizedText(getContent(source))}
          </p>
        )}
      </div>
      <div className="flex items-center text-xs text-muted-foreground">
        <div className="flex w-full flex-1 items-center justify-between gap-1">
          <div className="flex items-center">
            <SiteFavicon hostname={hostname} />
            <p className="ml-1 truncate">
              {hostname.replace('www.', '').split('/')[0]}
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-1">
            {isIssue && (
              <>
                {source.closed ? (
                  <IconCheckCircled className="h-3.5 w-3.5" />
                ) : (
                  <IconCircleDot className="h-3.5 w-3.5" />
                )}
                <span>{source.closed ? 'Closed' : 'Open'}</span>
              </>
            )}
            {isPR && (
              <>
                {source.merged ? (
                  <IconGitMerge className="h-3.5 w-3.5" />
                ) : (
                  <IconGitPullRequest className="h-3.5 w-3.5" />
                )}
                {source.merged ? 'Merged' : 'Open'}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function MessageContentForm({
  message,
  onCancel,
  onSubmit
}: {
  message: ConversationMessage
  onCancel: () => void
  onSubmit: (
    newMessage: ConversationMessage
  ) => Promise<ExtendedCombinedError | void>
}) {
  const formSchema = z.object({
    content: z.string().trim()
  })
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: { content: message.content }
  })
  const { isSubmitting } = form.formState
  const [draftMessage] = useState<ConversationMessage>(message)

  const handleSubmit = async (values: z.infer<typeof formSchema>) => {
    const error = await onSubmit({
      ...draftMessage,
      content: values.content
    })

    if (error) {
      makeFormErrorHandler(form)(error)
    }
  }

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(handleSubmit)}>
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
            <Button type="submit" disabled={isSubmitting}>
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
