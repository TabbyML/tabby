'use client'

import React from 'react'
import { useChat } from 'ai/react'
import type { Message, UseChatHelpers } from 'ai/react'
import { find, findIndex } from 'lodash-es'
import { toast } from 'sonner'

import { usePatchFetch } from '@/lib/hooks/use-patch-fetch'
import { useStore } from '@/lib/hooks/use-store'
import { addChat, updateMessages } from '@/lib/stores/chat-actions'
import { useChatStore } from '@/lib/stores/chat-store'
import type { MessageActionType } from '@/lib/types'
import { cn, nanoid, truncateText } from '@/lib/utils'
import { ChatList } from '@/components/chat-list'
import { ChatPanel } from '@/components/chat-panel'
import { ChatScrollAnchor } from '@/components/chat-scroll-anchor'
import { EmptyScreen } from '@/components/empty-screen'

export interface ChatProps extends React.ComponentProps<'div'> {
  initialMessages?: Message[]
  id?: string
  chatPanelClassName?: string
}

export interface ChatRef extends UseChatHelpers {}

function ChatRenderer(
  { id, initialMessages, className, chatPanelClassName }: ChatProps,
  ref: React.ForwardedRef<ChatRef>
) {
  usePatchFetch()
  const chats = useStore(useChatStore, state => state.chats)

  const useChatHelpers = useChat({
    initialMessages,
    id,
    body: {
      id
    },
    onResponse(response) {
      if (response.status === 401) {
        toast.error(response.statusText)
      }
    }
  })

  const {
    messages,
    append,
    reload,
    stop,
    isLoading,
    input,
    setInput,
    setMessages
  } = useChatHelpers

  const [selectedMessageId, setSelectedMessageId] = React.useState<string>()

  const onRegenerateResponse = (messageId: string) => {
    const messageIndex = findIndex(messages, { id: messageId })
    const prevMessage = messages?.[messageIndex - 1]
    if (prevMessage?.role === 'user') {
      setMessages(messages.slice(0, messageIndex - 1))
      append(prevMessage)
    }
  }

  const onDeleteMessage = (messageId: string) => {
    const message = find(messages, { id: messageId })
    if (message) {
      setMessages(messages.filter(m => m.id !== messageId))
    }
  }

  const onEditMessage = (messageId: string) => {
    const message = find(messages, { id: messageId })
    if (message) {
      setInput(message.content)
      setSelectedMessageId(messageId)
    }
  }

  const handleMessageAction = (
    messageId: string,
    actionType: MessageActionType
  ) => {
    switch (actionType) {
      case 'edit':
        onEditMessage(messageId)
        break
      case 'delete':
        onDeleteMessage(messageId)
        break
      case 'regenerate':
        onRegenerateResponse(messageId)
        break
      default:
        break
    }
  }

  const handleSubmit = async (value: string) => {
    if (findIndex(chats, { id }) === -1) {
      addChat(id, truncateText(value))
    } else if (selectedMessageId) {
      let messageIdx = findIndex(messages, { id: selectedMessageId })
      setMessages(messages.slice(0, messageIdx))
      setSelectedMessageId(undefined)
    }
    await append({
      id: nanoid(),
      content: value,
      role: 'user'
    })
  }

  React.useEffect(() => {
    if (id) {
      updateMessages(id, messages)
    }
  }, [messages])

  React.useEffect(() => {
    const scrollHeight = document.documentElement.scrollHeight
    window.scrollTo(0, scrollHeight)

    return () => stop()
  }, [])

  React.useImperativeHandle(
    ref,
    () => {
      return useChatHelpers
    },
    [useChatHelpers]
  )

  return (
    <div className="flex justify-center overflow-x-hidden">
      <div className="w-full max-w-2xl px-4">
        <div className={cn('pb-[200px] pt-4 md:pt-10', className)}>
          {messages.length ? (
            <>
              <ChatList
                messages={messages}
                handleMessageAction={handleMessageAction}
              />
              <ChatScrollAnchor trackVisibility={isLoading} />
            </>
          ) : (
            <EmptyScreen setInput={setInput} />
          )}
        </div>
        <ChatPanel
          onSubmit={handleSubmit}
          className={cn(
            'fixed inset-x-0 bottom-0 lg:ml-[280px]',
            chatPanelClassName
          )}
          id={id}
          isLoading={isLoading}
          stop={stop}
          append={append}
          reload={reload}
          messages={messages}
          input={input}
          setInput={setInput}
        />
      </div>
    </div>
  )
}

export const Chat = React.forwardRef<ChatRef, ChatProps>(ChatRenderer)
