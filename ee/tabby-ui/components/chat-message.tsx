// Inspired by Chatbot-UI and modified to fit the needs of this project
// @see https://github.com/mckaywrigley/chatbot-ui/blob/main/components/Chat/ChatMessage.tsx

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

import { IconSpinner } from './ui/icons'
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
      <div
        className={cn(
          'bg-background shrink-0 select-none rounded-full border shadow'
        )}
      >
        {message.role === 'user' ? (
          <UserAvatar className="h-8 w-8" />
        ) : (
          <IconTabby className="h-8 w-8" />
        )}
      </div>
      <div className="ml-4 flex-1 space-y-2 overflow-hidden px-1">
        <MemoizedReactMarkdown
          className="prose dark:prose-invert prose-p:leading-relaxed prose-pre:p-0 break-words"
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
      <div
        className={cn(
          'bg-background shrink-0 select-none rounded-full border shadow'
        )}
      >
        <IconTabby className="h-8 w-8" />
      </div>
      <div className="ml-4 flex h-8 items-center">
        <IconSpinner className="h-6 w-6" />
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
