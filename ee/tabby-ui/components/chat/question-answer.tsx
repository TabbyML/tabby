// Inspired by Chatbot-UI and modified to fit the needs of this project
// @see https://github.com/mckaywrigley/chatbot-ui/blob/main/components/Chat/ChatMessage.tsx

import React, { useState } from 'react'
import Image from 'next/image'
import tabbyLogo from '@/assets/tabby.png'
import { isNil } from 'lodash-es'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import type { Context } from 'tabby-chat-panel'

import { useMe } from '@/lib/hooks/use-me'
import { filename2prism } from '@/lib/language-utils'
import {
  AssistantMessage,
  QuestionAnswerPair,
  RelevantCodeContext,
  UserMessage
} from '@/lib/types/chat'
import { cn } from '@/lib/utils'
import { CodeBlock } from '@/components/ui/codeblock'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'
import { MemoizedReactMarkdown } from '@/components/markdown'

import { CopyButton } from '../copy-button'
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger
} from '../ui/accordion'
import { Button } from '../ui/button'
import {
  IconExternalLink,
  IconFile,
  IconRefresh,
  IconTrash,
  IconUser
} from '../ui/icons'
import { Separator } from '../ui/separator'
import { Skeleton } from '../ui/skeleton'
import { UserAvatar } from '../user-avatar'
import { ChatContext } from './chat'

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
}

type SelectCode = {
  filepath: string
  isMultiLine: boolean
}

function QuestionAnswerItem({ message, isLoading }: QuestionAnswerItemProps) {
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
          {!!selectCodeSnippet && (
            <MessageMarkdown message={selectCodeSnippet} />
          )}
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
  const { handleMessageAction } = React.useContext(ChatContext)
  return (
    <ChatMessageActionsWrapper>
      <Button
        variant="ghost"
        size="icon"
        onClick={e => handleMessageAction?.(message.id, 'delete')}
      >
        <IconTrash />
        <span className="sr-only">Delete message</span>
      </Button>
    </ChatMessageActionsWrapper>
  )
}

interface AssistantMessageCardProps {
  userMessageId: string
  isLoading: boolean
  message: AssistantMessage
  userMessage: UserMessage
}

interface AssistantMessageActionProps {
  userMessageId: string
  message: AssistantMessage
}

function AssistantMessageCard(props: AssistantMessageCardProps) {
  const { message, userMessage, isLoading, userMessageId, ...rest } = props
  const { onNavigateToContext, client } = React.useContext(ChatContext)
  const contexts: Array<Context> = React.useMemo(() => {
    return (
      message?.relevant_code?.map(code => {
        const start_line = code.doc?.start_line ?? 0
        const lineCount = code.doc.body.split('\n').length
        const end_line = start_line + lineCount - 1

        return {
          kind: 'file',
          range: {
            start: start_line,
            end: end_line
          },
          filepath: code.doc.filepath,
          content: code.doc.body,
          git_url: code.doc.git_url
        }
      }) ?? []
    )
  }, [message?.relevant_code])

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
          />
        </div>
      </div>

      <div className="w-full flex-1 space-y-2 overflow-hidden px-1 md:ml-4">
        <CodeReferences
          contexts={contexts}
          userContexts={userMessage.relevantContext}
          onContextClick={(ctx, isInWorkspace) => {
            onNavigateToContext?.(ctx, {
              openInEditor: client === 'vscode' && isInWorkspace
            })
          }}
        />
        {isLoading && !message?.message ? (
          <MessagePendingIndicator />
        ) : (
          <>
            <MessageMarkdown message={message.message} />
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
  const { message, userMessageId } = props
  return (
    <ChatMessageActionsWrapper>
      {!isGenerating && (
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

function MessageMarkdown({ message }: { message: string }) {
  const { onCopyContent, onApplyInEditor } = React.useContext(ChatContext)
  return (
    <MemoizedReactMarkdown
      className="prose max-w-none break-words dark:prose-invert prose-p:leading-relaxed prose-pre:mt-1 prose-pre:p-0"
      remarkPlugins={[remarkGfm, remarkMath]}
      components={{
        p({ children }) {
          return <p className="mb-2 last:mb-0">{children}</p>
        },
        code({ node, inline, className, children, ...props }) {
          if (children.length) {
            if (children[0] == '▍') {
              return (
                <span className="mt-1 animate-pulse cursor-default">▍</span>
              )
            }

            children[0] = (children[0] as string).replace('`▍`', '▍')
          }

          const match = /language-(\w+)/.exec(className || '')
          const metaObject = parseMetaDataString(node.data?.meta as string)
          const isReference = metaObject?.['is_reference'] === '1'

          if (isReference) {
            return null
          }

          if (inline) {
            return (
              <code className={className} {...props}>
                {children}
              </code>
            )
          }

          return (
            <CodeBlock
              key={Math.random()}
              language={(match && match[1]) || ''}
              value={String(children).replace(/\n$/, '')}
              onCopyContent={onCopyContent}
              onApplyInEditor={onApplyInEditor}
              {...props}
            />
          )
        }
      }}
    >
      {message}
    </MemoizedReactMarkdown>
  )
}

function ErrorMessageBlock({ error = 'Fail to fetch' }: { error?: string }) {
  const errorMessage = React.useMemo(() => {
    let jsonString = JSON.stringify(
      {
        error: true,
        message: error
      },
      null,
      2
    )
    const markdownJson = '```\n' + jsonString + '\n```'
    return markdownJson
  }, [error])
  return (
    <MemoizedReactMarkdown
      className="prose break-words text-sm dark:prose-invert prose-p:leading-relaxed prose-pre:mt-1 prose-pre:p-0"
      remarkPlugins={[remarkGfm, remarkMath]}
      components={{
        code({ node, inline, className, children, ...props }) {
          return (
            <div {...props} className={cn(className, 'bg-zinc-950 p-2')}>
              {children}
            </div>
          )
        }
      }}
    >
      {errorMessage}
    </MemoizedReactMarkdown>
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

interface ContextReferencesProps {
  contexts: RelevantCodeContext[]
  userContexts?: RelevantCodeContext[]
  className?: string
  onContextClick?: (
    context: RelevantCodeContext,
    isInWorkspace?: boolean
  ) => void
  defaultOpen?: boolean
  enableTooltip?: boolean
  onTooltipClick?: () => void
}
export const CodeReferences = ({
  contexts,
  userContexts,
  className,
  onContextClick,
  defaultOpen,
  enableTooltip,
  onTooltipClick
}: ContextReferencesProps) => {
  const totalContextLength = (userContexts?.length || 0) + contexts.length
  const isMultipleReferences = totalContextLength > 1

  if (totalContextLength === 0) return null
  return (
    <Accordion
      type="single"
      collapsible
      className={cn('bg-transparent text-foreground', className)}
      defaultValue={defaultOpen ? 'references' : undefined}
    >
      <AccordionItem value="references" className="my-0 border-0">
        <AccordionTrigger className="my-0 py-2 font-semibold">
          <span className="mr-2">{`Read ${totalContextLength} file${
            isMultipleReferences ? 's' : ''
          }`}</span>
        </AccordionTrigger>
        <AccordionContent className="space-y-2">
          {userContexts?.map((item, index) => {
            return (
              <ContextItem
                key={`user-${index}`}
                context={item}
                onContextClick={ctx => onContextClick?.(ctx, true)}
              />
            )
          })}
          {contexts.map((item, index) => {
            return (
              <ContextItem
                key={`assistant-${index}`}
                context={item}
                onContextClick={ctx => onContextClick?.(ctx, false)}
                enableTooltip={enableTooltip}
                onTooltipClick={onTooltipClick}
                isServerSide
              />
            )
          })}
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  )
}

function ContextItem({
  context,
  clickable = true,
  onContextClick,
  enableTooltip,
  onTooltipClick,
  isServerSide
}: {
  context: RelevantCodeContext
  clickable?: boolean
  onContextClick?: (context: RelevantCodeContext) => void
  enableTooltip?: boolean
  onTooltipClick?: () => void
  isServerSide?: boolean
}) {
  const [tooltipOpen, setTooltipOpen] = useState(false)
  const isMultiLine =
    !isNil(context.range?.start) &&
    !isNil(context.range?.end) &&
    context.range.start < context.range.end
  const pathSegments = context.filepath.split('/')
  const fileName = pathSegments[pathSegments.length - 1]
  const path = pathSegments.slice(0, pathSegments.length - 1).join('/')
  const scores = context?.extra?.scores
  const onTooltipOpenChange = (v: boolean) => {
    if (!enableTooltip || !scores) return

    setTooltipOpen(v)
  }

  return (
    <Tooltip
      open={tooltipOpen}
      onOpenChange={onTooltipOpenChange}
      delayDuration={0}
    >
      <TooltipTrigger asChild>
        <div
          className={cn('rounded-md border p-2', {
            'cursor-pointer hover:bg-accent': clickable,
            'cursor-default pointer-events-auto': !clickable
          })}
          onClick={e => clickable && onContextClick?.(context)}
        >
          <div className="flex items-center gap-1 overflow-hidden">
            <IconFile className="shrink-0" />
            <div className="flex-1 truncate" title={context.filepath}>
              <span>{fileName}</span>
              {context.range?.start && (
                <span className="text-muted-foreground">
                  :{context.range.start}
                </span>
              )}
              {isMultiLine && (
                <span className="text-muted-foreground">
                  -{context.range.end}
                </span>
              )}
              <span className="ml-2 text-xs text-muted-foreground">{path}</span>
            </div>
            {isServerSide && (
              <IconExternalLink className="shrink-0 text-muted-foreground" />
            )}
          </div>
        </div>
      </TooltipTrigger>
      <TooltipContent
        align="start"
        onClick={onTooltipClick}
        className="cursor-pointer p-2"
      >
        <div>
          <div className="mb-2 font-semibold">Scores</div>
          <div className="space-y-1">
            <div className="flex">
              <span className="w-20">rrf:</span>
              {scores?.rrf}
            </div>
            <div className="flex">
              <span className="w-20">bm25:</span>
              {scores?.bm25}
            </div>
            <div className="flex">
              <span className="w-20">embedding:</span>
              {scores?.embedding}
            </div>
          </div>
        </div>
      </TooltipContent>
    </Tooltip>
  )
}

function parseMetaDataString(metaData: string | undefined) {
  const metadataObj: Record<string, string> = {}
  if (!metaData) return metadataObj

  const keyValuePairs = metaData.split(' ')
  keyValuePairs.forEach(pair => {
    const [key, value] = pair.split('=')
    metadataObj[key] = value
  })

  return metadataObj
}

export { QuestionAnswerList }
