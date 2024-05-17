'use client'

import { useRef, useState } from 'react'
import type { ChatMessage, FetcherOptions } from 'tabby-chat-panel'
import { useServer } from 'tabby-chat-panel/react'

import { nanoid } from '@/lib/utils'
import { Chat, ChatRef } from '@/components/chat/chat'

export default function ChatPage() {
  const [isInit, setIsInit] = useState(false)
  const [fetcherOptions, setFetcherOptions] = useState<FetcherOptions | null>(
    null
  )
  const [activeChatId, setActiveChatId] = useState("")
  const chatRef = useRef<ChatRef>(null)
  let messageQueueBeforeInit: ChatMessage[] = []

  const sendMessage = (message: ChatMessage) => {
    if (chatRef.current) {
      chatRef.current.sendUserChat(message)
    } else {
      messageQueueBeforeInit.push(message)
    }
  }

  useServer({
    init: request => {
      if (chatRef.current) return
      setActiveChatId(nanoid())
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
    Authorization: `Bearer ${fetcherOptions.authorization}`
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
