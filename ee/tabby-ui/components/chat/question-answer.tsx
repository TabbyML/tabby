// Inspired by Chatbot-UI and modified to fit the needs of this project
// @see https://github.com/mckaywrigley/chatbot-ui/blob/main/components/Chat/ChatMessage.tsx

import React, { useMemo } from 'react'
import Image from 'next/image'
import tabbyLogo from '@/assets/tabby.png'
import { compact, concat, isNil } from 'lodash-es'
import type { Context } from 'tabby-chat-panel'

import { MessageAttachmentCode } from '@/lib/gql/generates/graphql'
import { useMe } from '@/lib/hooks/use-me'
import { filename2prism } from '@/lib/language-utils'
import {
  AssistantMessage,
  AttachmentCodeItem,
  QuestionAnswerPair,
  UserMessage
} from '@/lib/types/chat'
import { cn, formatLineHashForCodeBrowser } from '@/lib/utils'

import { CopyButton } from '../copy-button'
import { ErrorMessageBlock, MessageMarkdown } from '../message-markdown'
import { Button } from '../ui/button'
import { IconFile, IconRefresh, IconTrash, IconUser } from '../ui/icons'
import { Separator } from '../ui/separator'
import { Skeleton } from '../ui/skeleton'
import { UserAvatar } from '../user-avatar'
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
  const { onNavigateToContext, client } = React.useContext(ChatContext)
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
        !isNil(range?.start) && !isNil(range?.end) && range.start < range.end
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
        className={cn('flex w-full items-center justify-between md:w-auto', {
          'hidden md:flex': !data?.me.name
        })}
      >
        <div className="flex items-center gap-x-2">
          <div className="shrink-0 select-none rounded-full border bg-background shadow">
            <UserAvatar
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
          <MessageMarkdown message={message.message} />
          <div className="hidden md:block">
            <UserMessageCardActions {...props} />
          </div>

          {selectCode && message.selectContext && (
            <div
              className="flex cursor-pointer items-center gap-1 overflow-x-auto text-xs text-muted-foreground hover:underline"
              onClick={() =>
                onNavigateToContext?.(message.selectContext!, {
                  openInEditor: client === 'vscode'
                })
              }
            >
              <IconFile className="h-3 w-3" />
              <p className="flex-1 truncate pr-1">
                <span>{selectCode.filepath}</span>
                {message.selectContext?.range?.start && (
                  <span>:{message.selectContext?.range.start}</span>
                )}
                {selectCode.isMultiLine && (
                  <span>-{message.selectContext?.range.end}</span>
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
          onClick={e => handleMessageAction?.(message.id, 'delete')}
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
  const { onNavigateToContext, client, onApplyInEditor, onCopyContent } =
    React.useContext(ChatContext)
  const [relevantCodeHighlightIndex, setRelevantCodeHighlightIndex] =
    React.useState<number | undefined>(undefined)
  const serverCode: Array<Context> = React.useMemo(() => {
    return (
      message?.relevant_code?.map(code => {
        const start_line = code?.startLine ?? 0
        const lineCount = code.content.split('\n').length
        const end_line = start_line + lineCount - 1

        return {
          kind: 'file',
          range: {
            start: start_line,
            end: end_line
          },
          filepath: code.filepath,
          content: code.content,
          git_url: code.gitUrl
        }
      }) ?? []
    )
  }, [message?.relevant_code])

  const clientCode: Array<Context> = React.useMemo(() => {
    return compact([
      userMessage.activeContext,
      ...(userMessage?.relevantContext ?? [])
    ])
  }, [userMessage.activeContext, userMessage.relevantContext])

  const attachmentDocsLen = 0

  const attachmentCode: Array<AttachmentCodeItem> = useMemo(() => {
    return concat(clientCode, serverCode).map(o => ({
      content: o.content,
      filepath: o.filepath,
      gitUrl: o.git_url,
      startLine: o.range.start,
      // FIXME
      language: ''
    }))
  }, [clientCode, serverCode])

  const onCodeCitationMouseEnter = (index: number) => {
    setRelevantCodeHighlightIndex(index - 1 - (attachmentDocsLen || 0))
  }

  const onCodeCitationMouseLeave = (index: number) => {
    setRelevantCodeHighlightIndex(undefined)
  }

  const openCodeBrowserTab = (code: MessageAttachmentCode) => {
    const start_line = code?.startLine ?? 0
    const lineCount = code.content.split('\n').length
    const end_line = start_line + lineCount - 1

    if (!code.filepath) return
    const url = new URL(`${window.location.origin}/files`)
    const searchParams = new URLSearchParams()
    searchParams.append('redirect_filepath', code.filepath)
    searchParams.append('redirect_git_url', code.gitUrl)
    url.search = searchParams.toString()

    const lineHash = formatLineHashForCodeBrowser({
      start: start_line,
      end: end_line
    })
    if (lineHash) {
      url.hash = lineHash
    }

    window.open(url.toString())
  }

  const onCodeCitationClick = (code: MessageAttachmentCode) => {
    openCodeBrowserTab(code)
  }

  return (
    <div
      className={cn(
        'group relative mb-4 flex flex-col items-start gap-y-2 md:-ml-4 md:flex-row'
      )}
      {...rest}
    >
      <div className="flex w-full items-center justify-between md:w-auto">
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
          />
        </div>
      </div>

      <div className="w-full flex-1 space-y-2 overflow-hidden px-1 md:ml-4">
        <CodeReferences
          contexts={serverCode}
          userContexts={clientCode}
          onContextClick={(ctx, isInWorkspace) => {
            onNavigateToContext?.(ctx, {
              openInEditor: client === 'vscode' && isInWorkspace
            })
          }}
          isExternalLink={!!client}
          highlightIndex={relevantCodeHighlightIndex}
        />
        {isLoading && !message?.message ? (
          <MessagePendingIndicator />
        ) : (
          <>
            <MessageMarkdown
              message={message.message}
              onApplyInEditor={onApplyInEditor}
              onCopyContent={onCopyContent}
              attachmentCode={attachmentCode}
              // FIXME
              // onCodeCitationClick={onCodeCitationClick}
              onCodeCitationMouseEnter={onCodeCitationMouseEnter}
              onCodeCitationMouseLeave={onCodeCitationMouseLeave}
            />
            {!!message.error && <ErrorMessageBlock error={message.error} />}
          </>
        )}
        <div className="hidden md:block">
          <AssistantMessageCardActions
            message={message}
            userMessageId={userMessageId}
          />
        </div>
      </div>
    </div>
  )
}

function AssistantMessageCardActions(props: AssistantMessageActionProps) {
  const {
    handleMessageAction,
    isLoading: isGenerating,
    onCopyContent
  } = React.useContext(ChatContext)
  const { message, userMessageId, enableRegenerating } = props
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
      <CopyButton value={message.message} onCopyContent={onCopyContent} />
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
        'flex items-center justify-end transition-opacity group-hover:opacity-100 md:absolute md:-right-[5rem] md:-top-2 md:opacity-0',
        className
      )}
      {...props}
    />
  )
}

export { QuestionAnswerList }
