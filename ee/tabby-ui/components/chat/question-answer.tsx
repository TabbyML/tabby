// Inspired by Chatbot-UI and modified to fit the needs of this project
// @see https://github.com/mckaywrigley/chatbot-ui/blob/main/components/Chat/ChatMessage.tsx

import React from 'react'
import Image from 'next/image'
import tabbyLogo from '@/assets/tabby.png'
import { compact, isNil } from 'lodash-es'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'

import {
  AssistantMessage,
  Context,
  QuestionAnswerPair,
  UserMessage
} from '@/lib/types/chat'
import { cn } from '@/lib/utils'
import { CodeBlock } from '@/components/ui/codeblock'
import { MemoizedReactMarkdown } from '@/components/markdown'

import { CopyButton } from '../copy-button'
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger
} from '../ui/accordion'
import { Button } from '../ui/button'
import { IconFile, IconRefresh, IconTrash } from '../ui/icons'
import { Separator } from '../ui/separator'
import { Skeleton } from '../ui/skeleton'
import { UserAvatar } from '../user-avatar'
import { ChatContext } from './chat'

interface QuestionAnswerListProps {
  messages: QuestionAnswerPair[]
}
function QuestionAnswerList({ messages }: QuestionAnswerListProps) {
  const { isLoading } = React.useContext(ChatContext)
  return (
    <div className="relative mx-auto max-w-2xl px-4">
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

function QuestionAnswerItem({ message, isLoading }: QuestionAnswerItemProps) {
  const { user, assistant } = message
  const selectContext = user.selectContext
  const relevantContext = user.relevantContext

  return (
    <>
      <UserMessageCard message={user} />
      {!!assistant && (
        <>
          <Separator className="my-4 md:my-8" />
          <AssistantMessageCard
            message={assistant}
            isLoading={isLoading}
            userMessageId={user.id}
            selectContext={selectContext}
            relevantContext={relevantContext}
          />
        </>
      )}
    </>
  )
}

function UserMessageCard(props: { message: UserMessage }) {
  const { message } = props
  const { handleMessageAction } = React.useContext(ChatContext)
  return (
    <div
      className={cn('group relative mb-4 flex items-start md:-ml-12')}
      {...props}
    >
      <div className="shrink-0 select-none rounded-full border bg-background shadow">
        <UserAvatar className="h-8 w-8" />
      </div>
      <div className="ml-4 flex-1 space-y-2 overflow-hidden px-1">
        <MessageMarkdown message={message.message} />
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
      </div>
    </div>
  )
}

interface AssistantMessageCardProps {
  userMessageId: string
  isLoading: boolean
  message: AssistantMessage
  selectContext?: Context
  relevantContext?: Array<Context>
}

function AssistantMessageCard(props: AssistantMessageCardProps) {
  const { handleMessageAction, isLoading: isGenerating } =
    React.useContext(ChatContext)
  const {
    message,
    selectContext,
    relevantContext,
    isLoading,
    userMessageId,
    ...rest
  } = props

  const contexts = React.useMemo(() => {
    return compact([selectContext, ...(relevantContext ?? [])])
  }, [selectContext, relevantContext])

  return (
    <div
      className={cn('group relative mb-4 flex items-start md:-ml-12')}
      {...rest}
    >
      <div className="shrink-0 select-none rounded-full border bg-background shadow">
        <IconTabby className="h-8 w-8" />
      </div>
      <div className="ml-4 flex-1 space-y-2 overflow-hidden px-1">
        <CodeReferences contexts={contexts} />
        {isLoading && !message?.message ? (
          <MessagePendingIndicator />
        ) : (
          <>
            <MessageMarkdown message={message.message} />
            {!!message.error && <ErrorMessageBlock error={message.error} />}
          </>
        )}
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
          <CopyButton value={message.message} />
        </ChatMessageActionsWrapper>
      </div>
    </div>
  )
}

function MessageMarkdown({ message }: { message: string }) {
  return (
    <MemoizedReactMarkdown
      className="prose break-words dark:prose-invert prose-p:leading-relaxed prose-pre:mt-1 prose-pre:p-0"
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
    <div className="space-y-2 px-1">
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
  contexts: Context[]
}
const CodeReferences = ({ contexts }: ContextReferencesProps) => {
  const { onNavigateToContext } = React.useContext(ChatContext)
  const isMultipleReferences = contexts?.length > 1

  if (!contexts?.length) return null

  return (
    <Accordion
      type="single"
      collapsible
      className="bg-background text-foreground"
    >
      <AccordionItem value="references" className="my-0 border-0">
        <AccordionTrigger className="my-0 py-2">
          <span className="mr-2">{`Used ${contexts.length} reference${
            isMultipleReferences ? 's' : ''
          }`}</span>
        </AccordionTrigger>
        <AccordionContent className="space-y-2">
          {contexts?.map(item => {
            const isMultiLine =
              !isNil(item.range?.start) && item.range.start < item.range.end
            const pathSegments = item.filePath.split('/')
            const fileName = pathSegments[pathSegments.length - 1]
            const path = pathSegments
              .slice(0, pathSegments.length - 1)
              .join('/')
            return (
              <div
                className="cursor-pointer rounded-md border p-2 hover:bg-accent"
                key={item.filePath}
                onClick={e => onNavigateToContext?.(item)}
              >
                <div className="flex items-center gap-1 overflow-x-auto">
                  <IconFile className="shrink-0" />
                  <span>
                    <span>{fileName}</span>
                    {item.range?.start && (
                      <span className="text-muted-foreground">
                        :{item.range.start}
                      </span>
                    )}
                    {isMultiLine && (
                      <span className="text-muted-foreground">
                        -{item.range.end}
                      </span>
                    )}
                  </span>
                  <span className="ml-2 text-xs text-muted-foreground">
                    {path}
                  </span>
                </div>
              </div>
            )
          })}
        </AccordionContent>
      </AccordionItem>
    </Accordion>
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
