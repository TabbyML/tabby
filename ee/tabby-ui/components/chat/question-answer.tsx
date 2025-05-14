// Inspired by Chatbot-UI and modified to fit the needs of this project
// @see https://github.com/mckaywrigley/chatbot-ui/blob/main/components/Chat/ChatMessage.tsx

import React, { useMemo } from 'react'
import Image from 'next/image'
import tabbyLogo from '@/assets/tabby.png'
import { compact, isEmpty, isEqual, isNil, uniqWith } from 'lodash-es'

import { MARKDOWN_CITATION_REGEX } from '@/lib/constants/regex'
import { useEnableSearchPages } from '@/lib/experiment-flags'
import { ContextSource, ContextSourceKind } from '@/lib/gql/generates/graphql'
import { useMe } from '@/lib/hooks/use-me'
import { filename2prism } from '@/lib/language-utils'
import {
  AssistantMessage,
  AttachmentCodeItem,
  Context,
  FileContext,
  QuestionAnswerPair,
  RelevantCodeContext,
  UserMessage
} from '@/lib/types/chat'
import {
  attachmentCodeToTerminalContext,
  buildCodeBrowserUrlForContext,
  buildMarkdownCodeBlock,
  cn,
  getFileLocationFromContext,
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
import { convertContextBlockToPlaceholder } from '@/lib/utils/markdown'

import { CodeRangeLabel } from '../code-range-label'
import { CopyButton } from '../copy-button'
import { ErrorMessageBlock, MessageMarkdown } from '../message-markdown'
import { Button } from '../ui/button'
import {
  IconEdit,
  IconFile,
  IconRefresh,
  IconTerminalSquare,
  IconTrash,
  IconUser
} from '../ui/icons'
import { Separator } from '../ui/separator'
import { Skeleton } from '../ui/skeleton'
import { MyAvatar } from '../user-avatar'
import { ChatContext } from './chat-context'
import { CodeReferences } from './code-references'
import { ReadingDocStepper } from './reading-doc-stepper'
import { ReadingRepoStepper } from './reading-repo-stepper'

interface QuestionAnswerListProps {
  messages: QuestionAnswerPair[]
}
function QuestionAnswerList({ messages }: QuestionAnswerListProps) {
  const { isLoading } = React.useContext(ChatContext)
  return (
    <div className="relative">
      {messages?.map((message, index) => {
        const isLastItem = index === messages.length - 1
        return (
          // use userMessageId as QuestionAnswerItem ID
          <React.Fragment key={message.user.id}>
            <QuestionAnswerItem
              isLoading={isLastItem ? isLoading : false}
              message={message}
              isLastItem={isLastItem}
            />
            {!isLastItem && <Separator className="my-4" />}
          </React.Fragment>
        )
      })}
    </div>
  )
}

interface QuestionAnswerItemProps {
  message: QuestionAnswerPair
  isLoading: boolean
  isLastItem?: boolean
}

type SelectCode = {
  filepath: string
  isMultiLine: boolean
}

function QuestionAnswerItem({
  message,
  isLoading,
  isLastItem
}: QuestionAnswerItemProps) {
  const { user, assistant } = message

  return (
    <>
      <UserMessageCard message={user} />
      {!!assistant && (
        <>
          <Separator className="my-4" />
          <AssistantMessageCard
            message={assistant}
            userMessage={user}
            isLoading={isLoading}
            userMessageId={user.id}
            enableRegenerating={isLastItem}
          />
        </>
      )}
    </>
  )
}

function UserMessageCard(props: { message: UserMessage }) {
  const { message } = props
  const [{ data }] = useMe()
  const selectContext = message.selectContext
  const {
    openInEditor,
    supportsOnApplyInEditorV2,
    contextInfo,
    fetchingContextInfo
  } = React.useContext(ChatContext)
  const selectCodeSnippet = React.useMemo(() => {
    if (selectContext?.kind === 'terminal') {
      return buildMarkdownCodeBlock(selectContext?.selection, 'shell')
    }
    if (!selectContext?.content) return ''
    const language = selectContext?.filepath
      ? filename2prism(selectContext?.filepath)[0] ?? ''
      : ''
    return buildMarkdownCodeBlock(selectContext?.content, language)
  }, [selectContext])

  let selectCode: SelectCode | null = null
  if (
    selectCodeSnippet &&
    message.selectContext &&
    message.selectContext.kind === 'file'
  ) {
    const { range, filepath } = message.selectContext
    selectCode = {
      filepath,
      isMultiLine:
        !!range &&
        !isNil(range?.start) &&
        !isNil(range?.end) &&
        range.start < range.end
    }
  }

  const processedContent = useMemo(() => {
    return convertContextBlockToPlaceholder(message.content)
  }, [message.content])

  return (
    <div
      className={cn('group relative mb-4 flex flex-col items-start gap-y-2')}
      {...props}
    >
      <div
        className={cn('flex min-h-[2rem] w-full items-center justify-between', {
          hidden: !data?.me.name
        })}
      >
        <div className="flex items-center gap-x-2">
          <div className="shrink-0 select-none rounded-full border bg-background shadow">
            <MyAvatar
              className="h-8 w-8"
              fallback={
                <div className="flex h-8 w-8 items-center justify-center">
                  <IconUser className="h-8 w-8" />
                </div>
              }
            />
          </div>
          <p className="block text-sm font-bold">{data?.me.name}</p>
        </div>

        <UserMessageCardActions {...props} />
      </div>

      <div className="group relative flex w-full justify-between gap-x-2">
        <div className="flex-1 space-y-2 overflow-hidden px-1">
          <MessageMarkdown
            message={processedContent}
            supportsOnApplyInEditorV2={supportsOnApplyInEditorV2}
            openInEditor={openInEditor}
            contextInfo={contextInfo}
            fetchingContextInfo={fetchingContextInfo}
          />

          {selectCode &&
            message.selectContext &&
            message.selectContext.kind === 'file' && (
              <div
                className="flex cursor-pointer items-center gap-1 overflow-x-auto text-xs text-muted-foreground hover:underline"
                onClick={() => {
                  const context = message.selectContext!
                  if (context.kind === 'file') {
                    openInEditor(getFileLocationFromContext(context))
                  }
                }}
              >
                <IconFile className="h-3 w-3" />
                <p className="flex-1 truncate pr-1">
                  <span>{selectCode.filepath}</span>
                  <CodeRangeLabel range={message.selectContext.range} />
                </p>
              </div>
            )}
          {message.selectContext?.kind === 'terminal' && (
            <div className="flex cursor-pointer items-center gap-1 overflow-x-auto text-xs text-muted-foreground hover:underline">
              <IconTerminalSquare className="h-3 w-3" />
              <p
                className="flex-1 truncate pr-1"
                title={message.selectContext.selection}
              >
                <span>{message.selectContext.name}</span>
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function UserMessageCardActions(props: { message: UserMessage }) {
  const { message } = props
  const { handleMessageAction, isLoading } = React.useContext(ChatContext)
  return (
    <ChatMessageActionsWrapper className="opacity-0 group-hover:opacity-100">
      {!isLoading && (
        <Button
          variant="ghost"
          size="icon"
          onClick={e => handleMessageAction(message.id, 'edit')}
        >
          <IconEdit />
          <span className="sr-only">Edit message</span>
        </Button>
      )}
      {!isLoading && (
        <Button
          variant="ghost"
          size="icon"
          onClick={e => handleMessageAction(message.id, 'delete')}
        >
          <IconTrash />
          <span className="sr-only">Delete message</span>
        </Button>
      )}
    </ChatMessageActionsWrapper>
  )
}

interface AssistantMessageCardProps {
  userMessageId: string
  isLoading: boolean
  message: AssistantMessage
  userMessage: UserMessage
  enableRegenerating?: boolean
}

interface AssistantMessageActionProps {
  userMessageId: string
  message: AssistantMessage
  enableRegenerating?: boolean
  attachmentCode?: Array<AttachmentCodeItem>
}

function AssistantMessageCard(props: AssistantMessageCardProps) {
  const {
    message,
    userMessage,
    isLoading,
    userMessageId,
    enableRegenerating,
    ...rest
  } = props
  const {
    onApplyInEditor,
    onCopyContent,
    onLookupSymbol,
    openInEditor,
    openExternal,
    supportsOnApplyInEditorV2,
    runShell,
    contextInfo
  } = React.useContext(ChatContext)

  const [enableSearchPages] = useEnableSearchPages()

  const clientCode: Array<Context> = React.useMemo(() => {
    return uniqWith(
      compact([
        userMessage.activeContext,
        ...(userMessage?.relevantContext ?? [])
      ]).map(item => {
        if (item.kind === 'terminal') {
          return item
        }
        const terminalContext = attachmentCodeToTerminalContext(item)
        if (terminalContext) {
          return terminalContext
        }
        return {
          kind: 'file',
          range: getRangeFromAttachmentCode(item),
          filepath: item.filepath,
          content: item.content,
          gitUrl: item.gitUrl,
          commit: item.commit ?? undefined
        }
      }),
      isEqual
    )
  }, [userMessage.activeContext, userMessage.relevantContext])

  const serverCode: Array<Context> = React.useMemo(() => {
    return (
      message?.attachment?.code?.map<Context>(code => {
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
          commit: code.commit ?? undefined
        }
      }) ?? []
    )
  }, [message?.attachment?.code])

  const attachmentClientCode: Array<
    Omit<AttachmentCodeItem, '__typename' | 'startLine' | 'gitUrl'> & {
      startLine?: number | undefined
      gitUrl?: string | undefined
    }
  > = useMemo(() => {
    const formattedAttachmentClientCode =
      clientCode?.map(o => {
        if (o.kind === 'terminal') {
          return {
            content: o.selection,
            filepath: '',
            gitUrl: '',
            baseDir: '',
            startLine: undefined,
            language: 'shell',
            isClient: true
          }
        }
        return {
          content: o.content,
          filepath: o.filepath,
          gitUrl: o.gitUrl,
          baseDir: o.baseDir,
          startLine: o.range ? o.range.start : undefined,
          language: filename2prism(o.filepath ?? '')[0],
          isClient: true
        }
      }) ?? []
    return formattedAttachmentClientCode
  }, [clientCode])

  const attachmentServerCode: Array<Omit<AttachmentCodeItem, '__typename'>> =
    useMemo(() => {
      const formattedServerAttachmentCode =
        serverCode?.map(o => {
          if (o.kind === 'terminal') {
            return {
              content: o.selection,
              filepath: '',
              gitUrl: '',
              baseDir: '',
              startLine: undefined,
              language: 'shell',
              isClient: false
            }
          }
          return {
            content: o.content,
            filepath: o.filepath,
            gitUrl: o.gitUrl ?? '',
            startLine: o.range?.start,
            language: filename2prism(o.filepath ?? '')[0],
            isClient: false
          }
        }) ?? []
      return compact([...formattedServerAttachmentCode])
    }, [serverCode])

  const messageAttachmentDocs = message?.attachment?.doc
  // pulls / issues / commits
  const codebaseDocs = useMemo(() => {
    return messageAttachmentDocs?.filter(
      x =>
        isAttachmentPullDoc(x) ||
        isAttachmentIssueDoc(x) ||
        isAttachmentCommitDoc(x)
    )
  }, [messageAttachmentDocs])
  // web docs
  const webDocs = useMemo(() => {
    return messageAttachmentDocs?.filter(
      x => isAttachmentWebDoc(x) || isAttachmentIngestedDoc(x)
    )
  }, [messageAttachmentDocs])
  // pages
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

    if (enableSearchPages.value || pages?.length) {
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
    enableSearchPages.value,
    pages?.length
  ])

  // When onApplyInEditor is null, it means isInEditor === false, thus there's no need to showExternalLink
  const isInEditor = !!onApplyInEditor

  const onContextClick = (context: RelevantCodeContext, isClient?: boolean) => {
    if (context.kind !== 'file') {
      return
    }
    // When isInEditor is false, we are in the code browser.
    // The `openInEditor` function implementation as `openInCodeBrowser`,
    // and will navigate to target without opening a new tab.
    // So we use `openInEditor` here.
    if (isClient || !isInEditor) {
      openInEditor(getFileLocationFromContext(context))
    } else {
      const url = buildCodeBrowserUrlForContext(window.location.href, context)
      openExternal(url)
    }
  }

  const onCodeCitationClick = (code: AttachmentCodeItem) => {
    const ctx: FileContext = {
      gitUrl: code.gitUrl,
      content: code.content,
      filepath: code.filepath,
      kind: 'file',
      range: getRangeFromAttachmentCode(code)
    }
    onContextClick(ctx, code.isClient)
  }

  const onLinkClick = (url: string) => {
    openExternal(url)
  }

  return (
    <div
      className={cn('group relative mb-4 flex flex-col items-start gap-y-2')}
      {...rest}
    >
      <div className="flex min-h-[2rem] w-full items-center justify-between">
        <div className="flex items-center gap-x-2">
          <div className="shrink-0 select-none rounded-full border bg-background shadow">
            <IconTabby className="h-8 w-8" />
          </div>
          <p className="block text-sm font-bold">Tabby</p>
        </div>

        <AssistantMessageCardActions
          message={message}
          userMessageId={userMessageId}
          enableRegenerating={enableRegenerating}
          attachmentCode={attachmentServerCode}
        />
      </div>

      <div className="mb-1 w-full flex-1 space-y-2 overflow-hidden px-1">
        {!!message.codeSourceId ? (
          <ReadingRepoStepper
            codeSourceId={message.codeSourceId}
            isReadingCode={message.isReadingCode}
            isReadingDocs={message.isReadingDocs}
            isReadingFileList={message.isReadingFileList}
            clientCodeContexts={clientCode}
            serverCodeContexts={serverCode}
            codeFileList={message.attachment?.codeFileList}
            docs={codebaseDocs}
            readingCode={message.readingCode}
            readingDoc={message.readingDoc}
            onContextClick={onContextClick}
            openExternal={openExternal}
          />
        ) : (
          <CodeReferences
            contexts={serverCode}
            clientContexts={clientCode}
            onContextClick={onContextClick}
            showExternalLink={isInEditor}
            supportsOpenInEditor={!!openInEditor}
            showClientCodeIcon={!isInEditor}
          />
        )}

        {!!docQuerySources?.length && (
          <ReadingDocStepper
            codeSourceId={message.codeSourceId}
            docQuerySources={docQuerySources}
            readingDoc={message.readingDoc}
            isReadingDocs={message.isReadingDocs}
            webDocs={webDocs}
            pages={pages}
            openExternal={openExternal}
          />
        )}

        {isLoading && !message?.content ? (
          <MessagePendingIndicator />
        ) : (
          <>
            <MessageMarkdown
              message={message.content}
              onApplyInEditor={onApplyInEditor}
              onCopyContent={onCopyContent}
              attachmentClientCode={attachmentClientCode}
              attachmentCode={attachmentServerCode}
              attachmentDocs={messageAttachmentDocs}
              onCodeCitationClick={onCodeCitationClick}
              onLinkClick={onLinkClick}
              isStreaming={isLoading}
              onLookupSymbol={onLookupSymbol}
              openInEditor={openInEditor}
              supportsOnApplyInEditorV2={supportsOnApplyInEditorV2}
              activeSelection={userMessage.activeContext}
              runShell={runShell}
            />
            {!!message.error && <ErrorMessageBlock error={message.error} />}
          </>
        )}
      </div>
    </div>
  )
}

function getCopyContent(
  content: string,
  attachmentCode?: Array<AttachmentCodeItem>
) {
  if (!attachmentCode || isEmpty(attachmentCode)) return content

  const parsedContent = content
    .replace(MARKDOWN_CITATION_REGEX, match => {
      const citationNumberMatch = match?.match(/\d+/)
      return `[${citationNumberMatch}]`
    })
    .trim()

  const codeCitations =
    attachmentCode
      .map((code, idx) => {
        const lineRangeText = getRangeTextFromAttachmentCode(code)
        const filenameText = compact([code.filepath, lineRangeText]).join(':')
        return `[${idx + 1}] ${filenameText}`
      })
      .join('\n') ?? ''

  return `${parsedContent}\n\nCitations:\n${codeCitations}`
}

function AssistantMessageCardActions(props: AssistantMessageActionProps) {
  const {
    handleMessageAction,
    isLoading: isGenerating,
    onCopyContent
  } = React.useContext(ChatContext)
  const { message, userMessageId, enableRegenerating, attachmentCode } = props
  const copyContent = useMemo(() => {
    return getCopyContent(message.content, attachmentCode)
  }, [message.content, attachmentCode])

  return (
    <ChatMessageActionsWrapper>
      {!isGenerating && enableRegenerating && (
        <Button
          variant="ghost"
          size="icon"
          onClick={e => handleMessageAction(userMessageId, 'regenerate')}
        >
          <IconRefresh />
          <span className="sr-only">Regenerate message</span>
        </Button>
      )}
      <CopyButton value={copyContent} onCopyContent={onCopyContent} />
    </ChatMessageActionsWrapper>
  )
}

function MessagePendingIndicator() {
  return (
    <div className="space-y-2 py-2">
      <Skeleton className="h-3 w-full" />
      <Skeleton className="h-3 w-full" />
    </div>
  )
}

function IconTabby({ className }: { className?: string }) {
  return (
    <Image
      style={{ backgroundColor: '#E8E2D2' }}
      className={cn('rounded-full p-0.5', className)}
      src={tabbyLogo}
      alt="tabby"
    />
  )
}

function ChatMessageActionsWrapper({
  className,
  ...props
}: React.ComponentProps<'div'>) {
  return (
    <div
      className={cn(
        'flex items-center justify-end transition-opacity',
        className
      )}
      {...props}
    />
  )
}

export { QuestionAnswerList }
