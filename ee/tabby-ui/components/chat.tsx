'use client'

import React from 'react'
import { useChat } from 'ai/react'
import type { Message } from 'ai/react'
import { find, findIndex } from 'lodash-es'
import { toast } from 'react-hot-toast'

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
import { ListSkeleton } from '@/components/skeleton'

export interface ChatProps extends React.ComponentProps<'div'> {
  initialMessages?: Message[]
  id?: string
  loading?: boolean
}

export function Chat({ id, initialMessages, loading, className }: ChatProps) {
  usePatchFetch()
  const chats = useStore(useChatStore, state => state.chats)

  const {
    messages,
    append,
    reload,
    stop,
    isLoading,
    input,
    setInput,
    setMessages
  } = useChat({
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

  return (
    <div className="flex justify-center overflow-x-hidden">
      <div className="w-full max-w-2xl px-4">
        <div className={cn('pb-[200px] pt-4 md:pt-10', className)}>
          {loading ? (
            <div className="group relative mb-4 flex animate-pulse items-start md:-ml-12">
              <div className="shrink-0">
                <span className="block h-8 w-8 rounded-md bg-gray-200 dark:bg-gray-700"></span>
              </div>
              <div className="ml-4 flex-1 space-y-2 overflow-hidden px-1">
                <ListSkeleton />
              </div>
            </div>
          ) : messages.length ? (
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
          className="fixed inset-x-0 bottom-0 lg:ml-[280px]"
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
