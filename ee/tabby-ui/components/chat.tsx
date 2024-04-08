'use client'

import React from 'react'
import { useChat } from 'ai/react'
import type { Message, UseChatHelpers } from 'ai/react'
import { find, findIndex } from 'lodash-es'
import { toast } from 'sonner'

import { useLatest } from '@/lib/hooks/use-latest'
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
}

export interface ChatRef extends UseChatHelpers {}

function ChatRenderer(
  { id, initialMessages, className }: ChatProps,
  ref: React.ForwardedRef<ChatRef>
) {
  const chats = useStore(useChatStore, state => state.chats)
  // When the response status text is 200, the variable should be false
  const [isStreamResponsePending, setIsStreamResponsePending] =
    React.useState(false)

  const onStreamToken = useLatest(() => {
    if (isStreamResponsePending) {
      setIsStreamResponsePending(false)
    }
  })

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

  usePatchFetch({
    onStart: () => {
      setIsStreamResponsePending(true)
    },
    onToken: () => {
      onStreamToken.current()
    },
    processRequestBody(body) {
      return mergeMessagesByRole(body)
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

  const onStop = () => {
    setIsStreamResponsePending(false)
    stop()
  }

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

  const scrollToBottom = (behavior?: ScrollBehavior) => {
    window.scrollTo({
      top: document.body.offsetHeight,
      behavior
    })
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
    scrollToBottom()

    return () => stop()
  }, [])

  React.useLayoutEffect(() => {
    if (isStreamResponsePending) {
      scrollToBottom('smooth')
    }
  }, [isStreamResponsePending])

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
                isStreamResponsePending={isStreamResponsePending}
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
          stop={onStop}
          append={append}
          reload={reload}
          messages={messages}
          input={input}
          setInput={setInput}
          setMessages={setMessages}
        />
      </div>
    </div>
  )
}

function mergeMessagesByRole(body: BodyInit | null | undefined) {
  if (typeof body !== 'string') return body
  try {
    const bodyObject = JSON.parse(body)
    let messages: Message[] = bodyObject.messages?.slice()
    if (Array.isArray(messages) && messages.length > 1) {
      let previewCursor = 0
      let curCursor = 1
      while (curCursor < messages.length) {
        let prevMessage = messages[previewCursor]
        let curMessage = messages[curCursor]
        if (curMessage.role === prevMessage.role) {
          messages = [
            ...messages.slice(0, previewCursor),
            {
              ...prevMessage,
              content: [prevMessage.content, curMessage.content].join('\n')
            },
            ...messages.slice(curCursor + 1)
          ]
        } else {
          previewCursor = curCursor++
        }
      }
      return JSON.stringify({
        ...bodyObject,
        messages
      })
    } else {
      return body
    }
  } catch (e) {
    return body
  }
}

export const Chat = React.forwardRef<ChatRef, ChatProps>(ChatRenderer)
