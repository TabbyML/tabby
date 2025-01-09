'use client'

import { MouseEventHandler, useContext, useMemo, useState } from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import DOMPurify from 'dompurify'
import he from 'he'
import { compact, concat, isEmpty } from 'lodash-es'
import { marked } from 'marked'
import { useForm } from 'react-hook-form'
import Textarea from 'react-textarea-autosize'
import { Context } from 'tabby-chat-panel/index'
import * as z from 'zod'

import { MARKDOWN_CITATION_REGEX } from '@/lib/constants/regex'
import {
  Maybe,
  MessageAttachmentClientCode,
  MessageAttachmentCode
} from '@/lib/gql/generates/graphql'
import { makeFormErrorHandler } from '@/lib/tabby/gql'
import {
  AttachmentCodeItem,
  AttachmentDocItem,
  ExtendedCombinedError,
  RelevantCodeContext
} from '@/lib/types'
import {
  cn,
  formatLineHashForCodeBrowser,
  getContent,
  getRangeFromAttachmentCode,
  getRangeTextFromAttachmentCode
} from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
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
  IconCheckCircled,
  IconChevronRight,
  IconCircleDot,
  IconEdit,
  IconGitMerge,
  IconGitPullRequest,
  IconLayers,
  IconMore,
  IconPlus,
  IconRefresh,
  IconSparkles,
  IconSpinner,
  IconTrash
} from '@/components/ui/icons'
import {
  Sheet,
  SheetClose,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
  SheetTrigger
} from '@/components/ui/sheet'
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

import { ConversationMessage, PageContext, SOURCE_CARD_STYLE } from './page'

export function SectionContent({
  className,
  message,
  showRelatedQuestion,
  isLoading,
  clientCode
}: {
  className?: string
  message: ConversationMessage
  showRelatedQuestion: boolean
  isLoading?: boolean
  isLastAssistantMessage?: boolean
  isDeletable?: boolean
  clientCode?: Maybe<Array<MessageAttachmentClientCode>>
}) {
  const {
    onRegenerateResponse,
    onSubmitSearch,
    enableDeveloperMode,
    contextInfo,
    fetchingContextInfo,
    onDeleteMessage,
    isThreadOwner,
    onUpdateMessage,
    mode
  } = useContext(PageContext)

  const { supportsOnApplyInEditorV2, onNavigateToContext } =
    useContext(ChatContext)

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

  const relevantCodeGitURL = message?.attachment?.code?.[0]?.gitUrl || ''

  const clientCodeContexts: RelevantCodeContext[] = useMemo(() => {
    if (!clientCode?.length) return []
    return (
      clientCode.map(code => {
        const { startLine, endLine } = getRangeFromAttachmentCode(code)

        return {
          kind: 'file',
          range: {
            start: startLine,
            end: endLine
          },
          filepath: code.filepath || '',
          content: code.content,
          git_url: relevantCodeGitURL
        }
      }) ?? []
    )
  }, [clientCode, relevantCodeGitURL])

  const serverCodeContexts: RelevantCodeContext[] = useMemo(() => {
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
  }, [clientCode, message?.attachment?.code])

  const messageAttachmentClientCode = useMemo(() => {
    return clientCode?.map(o => ({
      ...o,
      gitUrl: relevantCodeGitURL
    }))
  }, [clientCode, relevantCodeGitURL])

  const messageAttachmentDocs = message?.attachment?.doc

  const sources = useMemo(() => {
    return concat<AttachmentDocItem | AttachmentCodeItem>(
      [],
      messageAttachmentDocs,
      messageAttachmentClientCode,
      message.attachment?.code
    )
  }, [
    messageAttachmentDocs,
    messageAttachmentClientCode,
    message.attachment?.code
  ])
  const sourceLen = sources.length

  // todo context
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

  const onCodeCitationMouseEnter = (index: number) => {}

  const onCodeCitationMouseLeave = (index: number) => {}

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

  return (
    <div className={cn('flex flex-col gap-y-5', className)}>
      {/* Section content */}
      <div>
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
              onCodeCitationMouseEnter={onCodeCitationMouseEnter}
              onCodeCitationMouseLeave={onCodeCitationMouseLeave}
              contextInfo={contextInfo}
              fetchingContextInfo={fetchingContextInfo}
              canWrapLongLines={!isLoading}
              supportsOnApplyInEditorV2={supportsOnApplyInEditorV2}
              className="prose-p:my-0.5 prose-ol:my-1 prose-ul:my-1"
            />
            {/* if isEditing, do not display error message block */}
            {message.error && <ErrorMessageBlock error={message.error} />}

            {!isLoading && !isEditing && (
              <div className="mt-3 flex items-center gap-3 text-sm">
                {sourceLen > 0 && (
                  <Sheet>
                    <SheetTrigger asChild>
                      <div className="border py-1 px-2 rounded-full cursor-pointer">
                        {sourceLen} sources
                      </div>
                    </SheetTrigger>
                    <SheetContent className="w-[50vw] min-w-[300px] flex flex-col">
                      <SheetHeader className="border-b">
                        <SheetTitle>Sources</SheetTitle>
                        <SheetClose />
                      </SheetHeader>
                      <div className="flex-1 overflow-y-auto space-y-3">
                        {sources.map((x, index) => {
                          // FIXME id
                          return <SourceCard source={x} key={index} />
                        })}
                      </div>
                      <SheetFooter>
                        <Button>Remove sources</Button>
                      </SheetFooter>
                    </SheetContent>
                  </Sheet>
                )}
                <div className="flex items-center gap-x-3">
                  {mode === 'view' && (
                    <CopyButton
                      className="-ml-1.5 gap-x-1 px-1 font-normal text-muted-foreground"
                      value={getCopyContent(message)}
                      text="Copy"
                    />
                  )}
                  {isThreadOwner && mode === 'edit' && (
                    <>
                      <Button
                        className="flex items-center gap-x-1 px-1 font-normal text-muted-foreground"
                        variant="ghost"
                        onClick={e => setIsEditing(true)}
                      >
                        <IconEdit />
                        <p>Edit</p>
                      </Button>
                      <Button
                        className="flex items-center gap-x-1 px-1 font-normal text-muted-foreground"
                        variant="ghost"
                      >
                        <IconRefresh />
                        <p>Regenerate</p>
                      </Button>
                      <DropdownMenu modal={false}>
                        <DropdownMenuTrigger asChild>
                          <Button
                            className="flex items-center gap-x-1 px-1 font-normal text-muted-foreground"
                            variant="ghost"
                          >
                            <IconMore />
                            <p>More</p>
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="start">
                          <DropdownMenuItem>Move Up</DropdownMenuItem>
                          <DropdownMenuItem>Move Down</DropdownMenuItem>
                          <DropdownMenuItem>Delete Section</DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </>
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
  source
}: {
  source: AttachmentDocItem | AttachmentCodeItem
}) {
  const { mode } = useContext(PageContext)
  const isEditMode = mode === 'edit'

  const isDoc =
    source.__typename === 'MessageAttachmentIssueDoc' ||
    source.__typename === 'MessageAttachmentPullDoc' ||
    source.__typename === 'MessageAttachmentWebDoc'

  if (isDoc) {
    return (
      <div className="flex items-start gap-2">
        {isEditMode && <Checkbox className="mt-2" />}
        <div
          className="relative flex cursor-pointer flex-col justify-between rounded-lg border bg-card text-card-foreground p-3 hover:bg-card/60"
          onClick={() => window.open(source.link)}
        >
          <DocSourceCard source={source} />
        </div>
      </div>
    )
  }

  return (
    <div className="flex items-start gap-2 w-full">
      {isEditMode && <Checkbox className="mt-2" />}
      <div className="relative flex flex-1 cursor-pointer flex-col justify-between rounded-lg border bg-card text-card-foreground p-3 hover:bg-card/60">
        <div className="flex flex-1 flex-col justify-between gap-y-1">
          <div className="flex flex-col gap-y-0.5">
            <p className="line-clamp-1 w-full overflow-hidden text-ellipsis break-all text-xs font-semibold">
              {source.filepath}
            </p>
          </div>
          <div className="flex items-center text-xs text-muted-foreground">
            <div className="flex w-full flex-1 items-center justify-between gap-1">
              <div className="flex items-center">
                <SiteFavicon hostname={source.gitUrl} />
                <p className="ml-1 truncate">{source.gitUrl}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function DocSourceCard({ source }: { source: AttachmentDocItem }) {
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
        {!showAvatar && (
          <p
            className={cn(
              ' w-full overflow-hidden text-ellipsis break-all text-xs text-muted-foreground',
              !showAvatar ? 'line-clamp-2' : 'line-clamp-1'
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
