import { type Message } from 'ai'

import { MessageActionType } from '@/lib/types'
import { Separator } from '@/components/ui/separator'
import { ChatMessage, MessagePendingIndicator } from './chat-message'
import { QuestionAnswerPair } from './chat'

export interface ChatList {
  messages: QuestionAnswerPair[]
  handleMessageAction: (messageId: string, action: 'delete' | 'regenerate') => void
  isStreamResponsePending?: boolean
}

export function ChatList({
  messages,
  handleMessageAction,
  isStreamResponsePending
}: ChatList) {
  if (!messages.length) {
    return null
  }

  const len = messages.length

  return (
    <div className="relative mx-auto max-w-2xl px-4">
      {messages.map((message, index) => (
        <div key={index}>
          <ChatMessage
            message={message}
            handleMessageAction={handleMessageAction}
          />
          {index < messages.length - 1 && (
            <Separator className="my-4 md:my-8" />
          )}
        </div>
      ))}
      {isStreamResponsePending && (
        <>
          <Separator className="my-4 md:my-8" />
          <MessagePendingIndicator />
        </>
      )}
    </div>
  )
}
