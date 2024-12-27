// Inspired by Chatbot-UI and modified to fit the needs of this project
// @see https://github.com/mckaywrigley/chatbot-ui/blob/main/components/Chat/ChatMessage.tsx

import React, { useMemo } from 'react'
import Image from 'next/image'
import tabbyLogo from '@/assets/tabby.png'
import { compact, isEmpty, isEqual, isNil, uniqWith } from 'lodash-es'

import { MARKDOWN_CITATION_REGEX } from '@/lib/constants/regex'
import { useMe } from '@/lib/hooks/use-me'
import { filename2prism } from '@/lib/language-utils'
import {
  AssistantMessage,
  AttachmentCodeItem,
  Context,
  QuestionAnswerPair,
  UserMessage
} from '@/lib/types/chat'
import {
  buildCodeBrowserUrlForContext,
  cn,
  getFileLocationFromContext,
  getRangeFromAttachmentCode,
  getRangeTextFromAttachmentCode
} from '@/lib/utils'

import { CopyButton } from '../copy-button'
import { ErrorMessageBlock, MessageMarkdown } from '../message-markdown'
import { Button } from '../ui/button'
import {
  IconEdit,
  IconFile,
  IconRefresh,
  IconTrash,
  IconUser
} from '../ui/icons'
import { Separator } from '../ui/separator'
import { Skeleton } from '../ui/skeleton'
import { MyAvatar } from '../user-avatar'
import { ChatContext } from './chat'
import { CodeReferences } from './code-references'

interface QuestionAnswerListProps {
  messages: QuestionAnswerPair[]
  chatMaxWidthClass: string
}
function QuestionAnswerList({
  messages,
  chatMaxWidthClass
}: QuestionAnswerListProps) {
  const { isLoading } = React.useContext(ChatContext)
  return (
    <div className={`relative mx-auto px-4 ${chatMaxWidthClass}`}>
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
            {!isLastItem && <Separator className="my-4 md:my-8" />}
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
          <Separator className="my-4 md:my-8" />
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
  const { openInEditor, supportsOnApplyInEditorV2 } =
    React.useContext(ChatContext)
  const selectCodeSnippet = React.useMemo(() => {
    if (!selectContext?.content) return ''
    const language = selectContext?.filepath
      ? filename2prism(selectContext?.filepath)[0] ?? ''
      : ''
    return `\n${'```'}${language}\n${selectContext?.content ?? ''}\n${'```'}\n`
  }, [selectContext])

  let selectCode: SelectCode | null = null
  if (selectCodeSnippet && message.selectContext) {
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
  return (
    <div
      className={cn(
        'group relative mb-4 flex flex-col items-start gap-y-2 md:-ml-4 md:flex-row'
      )}
      {...props}
    >
      <div
        className={cn(
          'flex min-h-[2rem] w-full items-center justify-between md:w-auto',
          {
            'hidden md:flex': !data?.me.name
          }
        )}
      >
        <div className="flex items-center gap-x-2">
          <div className="shrink-0 select-none rounded-full border bg-background shadow">
            <MyAvatar
              className="h-6 w-6 md:h-8 md:w-8"
              fallback={
                <div className="flex h-6 w-6 items-center justify-center md:h-8 md:w-8">
                  <IconUser className="h-6 w-6" />
                </div>
              }
            />
          </div>
          <p className="block text-xs font-bold md:hidden">{data?.me.name}</p>
        </div>

        <div className="block opacity-0 transition-opacity group-hover:opacity-100 md:hidden">
          <UserMessageCardActions {...props} />
        </div>
      </div>

      <div className="group relative flex w-full justify-between gap-x-2">
        <div className="flex-1 space-y-2 overflow-hidden px-1 md:ml-4">
          <MessageMarkdown
            message={message.message}
            canWrapLongLines
            supportsOnApplyInEditorV2={supportsOnApplyInEditorV2}
          />
          <div className="hidden md:block">
            <UserMessageCardActions {...props} />
          </div>

          {selectCode && message.selectContext && (
            <div
              className="flex cursor-pointer items-center gap-1 overflow-x-auto text-xs text-muted-foreground hover:underline"
              onClick={() => {
                const context = message.selectContext!
                openInEditor(getFileLocationFromContext(context))
              }}
            >
              <IconFile className="h-3 w-3" />
              <p className="flex-1 truncate pr-1">
                <span>{selectCode.filepath}</span>
                {message.selectContext?.range?.start && (
                  <span>:{message.selectContext?.range.start}</span>
                )}
                {selectCode.isMultiLine && (
                  <span>-{message.selectContext?.range?.end}</span>
                )}
              </p>
            </div>
          )}
        </div>
        {!data?.me.name && (
          <div className="editor-bg absolute right-0 top-0 -mt-0.5 block opacity-0 transition-opacity group-hover:opacity-100 md:hidden">
            <UserMessageCardActions {...props} />
          </div>
        )}
      </div>
    </div>
  )
}

function UserMessageCardActions(props: { message: UserMessage }) {
  const { message } = props
  const { handleMessageAction, isLoading } = React.useContext(ChatContext)
  return (
    <ChatMessageActionsWrapper>
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
    supportsOnApplyInEditorV2
  } = React.useContext(ChatContext)
  const [relevantCodeHighlightIndex, setRelevantCodeHighlightIndex] =
    React.useState<number | undefined>(undefined)
  const serverCode: Array<Context> = React.useMemo(() => {
    return (
      message?.relevant_code?.map(code => ({
        kind: 'file',
        range: getRangeFromAttachmentCode(code),
        filepath: code.filepath,
        content: code.content,
        git_url: code.gitUrl,
        commit: code.commit ?? undefined
      })) ?? []
    )
  }, [message?.relevant_code])

  const clientCode: Array<Context> = React.useMemo(() => {
    return uniqWith(
      compact([
        userMessage.activeContext,
        ...(userMessage?.relevantContext ?? [])
      ]),
      isEqual
    )
  }, [userMessage.activeContext, userMessage.relevantContext])

  const attachmentDocsLen = 0

  const attachmentClientCode: Array<
    Omit<AttachmentCodeItem, '__typename' | 'startLine'> & {
      startLine: number | undefined
    }
  > = useMemo(() => {
    const formatedAttachmentClientCode =
      clientCode?.map(o => ({
        content: o.content,
        filepath: o.filepath,
        gitUrl: o.git_url,
        startLine: o.range ? o.range.start : undefined,
        language: filename2prism(o.filepath ?? '')[0],
        isClient: true
      })) ?? []
    return formatedAttachmentClientCode
  }, [clientCode])

  const attachmentCode: Array<Omit<AttachmentCodeItem, '__typename'>> =
    useMemo(() => {
      const formatedServerAttachmentCode =
        serverCode?.map(o => ({
          content: o.content,
          filepath: o.filepath,
          gitUrl: o.git_url,
          startLine: o.range?.start,
          language: filename2prism(o.filepath ?? '')[0],
          isClient: false
        })) ?? []
      return compact([...formatedServerAttachmentCode])
    }, [serverCode])

  const onCodeCitationMouseEnter = (index: number) => {
    setRelevantCodeHighlightIndex(index - 1 - (attachmentDocsLen || 0))
  }

  const onCodeCitationMouseLeave = (index: number) => {
    setRelevantCodeHighlightIndex(undefined)
  }

  // When onApplyInEditor is null, it means isInEditor === false, thus there's no need to showExternalLink
  const isInEditor = !!onApplyInEditor

  const onContextClick = (context: Context, isClient?: boolean) => {
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
    const ctx: Context = {
      git_url: code.gitUrl,
      content: code.content,
      filepath: code.filepath,
      kind: 'file',
      range: getRangeFromAttachmentCode(code)
    }
    onContextClick(ctx, code.isClient)
  }

  return (
    <div
      className={cn(
        'group relative mb-4 flex flex-col items-start gap-y-2 md:-ml-4 md:flex-row'
      )}
      {...rest}
    >
      <div className="flex min-h-[2rem] w-full items-center justify-between md:w-auto">
        <div className="flex items-center gap-x-2">
          <div className="shrink-0 select-none rounded-full border bg-background shadow">
            <IconTabby className="h-6 w-6 md:h-8 md:w-8" />
          </div>
          <p className="block text-xs font-bold md:hidden">Tabby</p>
        </div>

        <div className="block opacity-0 transition-opacity group-hover:opacity-100 md:hidden">
          <AssistantMessageCardActions
            message={message}
            userMessageId={userMessageId}
            enableRegenerating={enableRegenerating}
            attachmentCode={attachmentCode}
          />
        </div>
      </div>

      <div className="w-full flex-1 space-y-2 overflow-hidden px-1 md:ml-4">
        <CodeReferences
          contexts={serverCode}
          clientContexts={clientCode}
          onContextClick={onContextClick}
          showExternalLink={isInEditor}
          isInEditor={isInEditor}
          showClientCodeIcon={!isInEditor}
          highlightIndex={relevantCodeHighlightIndex}
          triggerClassname="md:pt-0"
        />
        {isLoading && !message?.message ? (
          <MessagePendingIndicator />
        ) : (
          <>
            <MessageMarkdown
              message={message.message}
              onApplyInEditor={onApplyInEditor}
              onCopyContent={onCopyContent}
              attachmentClientCode={attachmentClientCode}
              attachmentCode={attachmentCode}
              onCodeCitationClick={onCodeCitationClick}
              onCodeCitationMouseEnter={onCodeCitationMouseEnter}
              onCodeCitationMouseLeave={onCodeCitationMouseLeave}
              canWrapLongLines={!isLoading}
              onLookupSymbol={onLookupSymbol}
              openInEditor={openInEditor}
              supportsOnApplyInEditorV2={supportsOnApplyInEditorV2}
              activeSelection={userMessage.activeContext}
            />
            {!!message.error && <ErrorMessageBlock error={message.error} />}
          </>
        )}
        <div className="hidden md:block">
          <AssistantMessageCardActions
            message={message}
            userMessageId={userMessageId}
            enableRegenerating={enableRegenerating}
            attachmentCode={attachmentCode}
          />
        </div>
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
    return getCopyContent(message.message, attachmentCode)
  }, [message.message, attachmentCode])

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
    <div className="space-y-2 py-2 md:px-1 md:py-0">
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
        'flex items-center justify-end transition-opacity group-hover:opacity-100 md:absolute md:-right-[4rem] md:-top-2 md:opacity-0',
        className
      )}
      {...props}
    />
  )
}

export { QuestionAnswerList }
