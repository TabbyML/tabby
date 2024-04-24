// Inspired by Chatbot-UI and modified to fit the needs of this project
// @see https://github.com/mckaywrigley/chatbot-ui/blob/main/components/Chat/ChatMessage.tsx

import { useMemo } from 'react'
import Image from 'next/image'
import tabbyLogo from '@/assets/tabby.png'
import { Message } from 'ai'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'

import { MessageActionType } from '@/lib/types'
import { cn } from '@/lib/utils'
import { CodeBlock } from '@/components/ui/codeblock'
import { ChatMessageActions } from '@/components/chat-message-actions'
import { MemoizedReactMarkdown } from '@/components/markdown'

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger
} from './ui/accordion'
import { IconFile } from './ui/icons'
import { Skeleton } from './ui/skeleton'
import { UserAvatar } from './user-avatar'

export interface ChatMessageProps {
  message: Message
  handleMessageAction: (messageId: string, action: MessageActionType) => void
}

export function ChatMessage({
  message,
  handleMessageAction,
  ...props
}: ChatMessageProps) {
  return (
    <div
      className={cn('group relative mb-4 flex items-start md:-ml-12')}
      {...props}
    >
      <div className="shrink-0 select-none rounded-full border bg-background shadow">
        {message.role === 'user' ? (
          <UserAvatar className="h-8 w-8" />
        ) : (
          <IconTabby className="h-8 w-8" />
        )}
      </div>
      <div className="ml-4 flex-1 space-y-2 overflow-hidden px-1">
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
              const isReference = metaObject['is_reference'] === '1'

              if (isReference) {
                return (
                  <CodeReferences
                    references={[metaObject as CodeReferenceItem]}
                  />
                )
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
          {message.content}
        </MemoizedReactMarkdown>
        <ChatMessageActions
          message={message}
          handleMessageAction={handleMessageAction}
        />
      </div>
    </div>
  )
}

export function MessagePendingIndicator() {
  return (
    <div className="mb-4 flex items-start md:-ml-12">
      <div className="shrink-0 select-none rounded-full border bg-background shadow">
        <IconTabby className="h-8 w-8" />
      </div>
      <div className="ml-4 flex-1 space-y-2 px-1">
        <Skeleton className="h-3 w-full" />
        <Skeleton className="h-3 w-full" />
      </div>
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

type CodeReferenceItem = {
  path: string
  line_from?: string
  line_to?: string
}

interface CodeReferencesProps {
  references: CodeReferenceItem[]
}
const CodeReferences = ({ references }: CodeReferencesProps) => {
  const formatedReferences = useMemo(() => {
    return references?.map(item => {
      const path = item.path
      const lastSlashIndex = path.lastIndexOf('/')
      const pathName = path.substring(0, lastSlashIndex)
      const fileName = path.substring(lastSlashIndex + 1)
      return {
        ...item,
        fileName,
        pathName
      }
    })
  }, [references])

  return (
    <Accordion
      type="single"
      collapsible
      className="bg-background text-foreground"
    >
      <AccordionItem value="reference" className="my-0 border-0">
        <AccordionTrigger className="my-0 py-2">
          <span className="mr-2">Used 1 reference</span>
        </AccordionTrigger>
        <AccordionContent>
          {formatedReferences?.map(item => {
            return (
              <div className="rounded-md border p-2" key={item.path}>
                <div className="flex items-center gap-1 overflow-x-auto">
                  <IconFile className="shrink-0" />
                  <span>
                    <span>{item.fileName}</span>
                    {item.line_from && (
                      <span className="text-muted-foreground">
                        :{item.line_from}
                      </span>
                    )}
                    {item.line_to && item.line_from !== item.line_to && (
                      <span className="text-muted-foreground">
                        -{item.line_to}
                      </span>
                    )}
                  </span>
                  <span className="ml-2 text-xs text-muted-foreground">
                    {item.pathName}
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
