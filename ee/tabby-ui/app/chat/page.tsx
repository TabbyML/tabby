'use client'

import { useState, useRef } from 'react'
import { useServer } from 'tabby-chat-panel/react'
import type { FetcherOptions, ChatMessage } from 'tabby-chat-panel'

import { nanoid } from '@/lib/utils'
import { Chat, ChatRef } from '@/components/chat/chat'

export default function ChatPage () {
  const [isInit, setIsInit] = useState(false)
  const [fetcherOptions, setFetcherOptions] = useState<FetcherOptions | null>(null)
  const chatRef = useRef<ChatRef>(null);
  const activeChatId = nanoid()
  let messageQueueBeforeInit: ChatMessage[] = [];

  const sendMessage = (message: ChatMessage) => {
    if (chatRef.current) {
      chatRef.current.sendUserChat(message);
    } else {
      messageQueueBeforeInit.push(message)
    }
  }

  useServer({
    init: (request) => {
      if (chatRef.current) return
      setIsInit(true)
      setFetcherOptions(request.fetcherOptions)

      messageQueueBeforeInit.forEach(sendMessage)
      messageQueueBeforeInit = []
    },
    sendMessage: (message: ChatMessage) => {
      return sendMessage(message)
    }
  })

  if (!isInit || !fetcherOptions) return <></>
  const headers = {
    'Authorization': `Bearer ${fetcherOptions.authorization}`
  }
  return (
    <Chat
      chatId={activeChatId}
      key={activeChatId}
      ref={chatRef}
      headers={headers}
      onThreadUpdates={() => {}}
      onNavigateToContext={() => {}}
    />
  )
}