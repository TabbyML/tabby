'use client'

import { useContext, useMemo, useState } from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { compact, isEmpty } from 'lodash-es'
import { useForm } from 'react-hook-form'
import Textarea from 'react-textarea-autosize'
import * as z from 'zod'

import { MARKDOWN_CITATION_REGEX } from '@/lib/constants/regex'
import {
  ContextSource,
  ContextSourceKind,
  Maybe,
  MessageAttachmentClientCode
} from '@/lib/gql/generates/graphql'
import { makeFormErrorHandler } from '@/lib/tabby/gql'
import {
  AttachmentCodeItem,
  ExtendedCombinedError,
  RelevantCodeContext
} from '@/lib/types'
import {
  attachmentCodeToTerminalContext,
  buildCodeBrowserUrlForContext,
  cn,
  formatLineHashForCodeBrowser,
  getMentionsFromText,
  getRangeFromAttachmentCode,
  getRangeTextFromAttachmentCode,
  isAttachmentCommitDoc,
  isAttachmentIngestedDoc,
  isAttachmentIssueDoc,
  isAttachmentPageDoc,
  isAttachmentPullDoc,
  isAttachmentWebDoc,
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
  IconBug,
  IconEdit,
  IconLayers,
  IconPlus,
  IconRefresh,
  IconSparkles,
  IconSpinner,
  IconTrash
} from '@/components/ui/icons'
import { Skeleton } from '@/components/ui/skeleton'
import { ChatContext } from '@/components/chat/chat-context'
import { CopyButton } from '@/components/copy-button'
import {
  ErrorMessageBlock,
  MessageMarkdown
} from '@/components/message-markdown'

import { ReadingCodeStepper } from './reading-code-step'
import { ReadingDocStepper } from './reading-doc-step'
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
  clientCode,
  enableSearchPages
}: {
  className?: string
  message: ConversationMessage
  userMessage: ConversationMessage
  showRelatedQuestion: boolean
  isLoading?: boolean
  isLastAssistantMessage?: boolean
  isDeletable?: boolean
  clientCode?: Maybe<Array<MessageAttachmentClientCode>>
  enableSearchPages: boolean
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
        ?.map((doc, idx) => {
          if (isAttachmentCommitDoc(doc)) {
            return `[${idx + 1}] ${doc.sha}`
          } else if (isAttachmentIngestedDoc(doc)) {
            return `[${idx + 1}] ${doc.ingestedDocLink ?? ''}`
          } else {
            return `[${idx + 1}] ${doc.link}`
          }
        })
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
          gitUrl: clientCodeGitUrl
        }
      }) ?? []
    )
  }, [clientCode, clientCodeGitUrl])

  const serverCodeContexts: RelevantCodeContext[] = useMemo(() => {
    return (
      message?.attachment?.code?.map(code => {
        const terminalContext = attachmentCodeToTerminalContext(code)
        if (terminalContext) {
          return terminalContext
        }
        return {
          kind: 'file',
          range: getRangeFromAttachmentCode(code),
          filepath: code.filepath,
          content: code.content,
          gitUrl: code.gitUrl,
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

  const codebaseDocs = useMemo(() => {
    return messageAttachmentDocs?.filter(
      x =>
        isAttachmentPullDoc(x) ||
        isAttachmentIssueDoc(x) ||
        isAttachmentCommitDoc(x)
    )
  }, [messageAttachmentDocs])

  const webDocs = useMemo(() => {
    return messageAttachmentDocs?.filter(
      x => isAttachmentWebDoc(x) || isAttachmentIngestedDoc(x)
    )
    // return messageAttachmentDocs?.filter(x => isAttachmentWebDoc(x))
  }, [messageAttachmentDocs])

  const pages = useMemo(() => {
    return messageAttachmentDocs?.filter(x => isAttachmentPageDoc(x))
  }, [messageAttachmentDocs])

  const docQuerySources: Array<Omit<ContextSource, 'id'>> = useMemo(() => {
    if (!contextInfo?.sources || !userMessage?.content) return []

    const _sources = getMentionsFromText(
      userMessage.content,
      contextInfo?.sources
    )

    const result = _sources
      .filter(x => isDocSourceContext(x.kind))
      .map(x => ({
        sourceId: x.id,
        sourceKind: x.kind,
        sourceName: x.label
      }))

    if (enableSearchPages || pages?.length) {
      result.unshift({
        sourceId: 'page',
        sourceKind: ContextSourceKind.Page,
        sourceName: 'Pages'
      })
    }

    return result
  }, [
    contextInfo?.sources,
    userMessage?.content,
    enableSearchPages,
    pages?.length
  ])

  const onCodeContextClick = (ctx: RelevantCodeContext) => {
    if (ctx.kind !== 'file') {
      return
    }
    if (!ctx.filepath) return
    const url = buildCodeBrowserUrlForContext(window.location.origin, ctx)
    window.open(url, '_blank')
  }

  const openCodeBrowserTab = (code: AttachmentCodeItem) => {
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

  const onCodeCitationClick = (code: AttachmentCodeItem) => {
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

  const showReadingCodeStepper = !!message.codeSourceId
  const showReadingDocStepper = !!docQuerySources?.length

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

        {(showReadingCodeStepper || showReadingDocStepper) && (
          <div className="mb-6 space-y-1.5">
            {showReadingCodeStepper && (
              <ReadingCodeStepper
                clientCodeContexts={clientCodeContexts}
                serverCodeContexts={serverCodeContexts}
                isReadingFileList={message.isReadingFileList}
                isReadingCode={message.isReadingCode}
                isReadingDocs={message.isReadingDocs}
                codeSourceId={message.codeSourceId}
                docQuery
                docQueryResources={docQuerySources}
                docs={codebaseDocs}
                codeFileList={message.attachment?.codeFileList}
                readingCode={message.readingCode}
                readingDoc={message.readingDoc}
                onContextClick={onCodeContextClick}
              />
            )}
            {showReadingDocStepper && (
              <ReadingDocStepper
                codeSourceId={message.codeSourceId}
                docQuerySources={docQuerySources}
                isReadingDocs={message.isReadingDocs}
                readingDoc={message.readingDoc}
                webDocs={webDocs}
                pages={pages}
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
              isStreaming={isLoading}
              supportsOnApplyInEditorV2={supportsOnApplyInEditorV2}
              onLinkClick={url => {
                window.open(url)
              }}
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
