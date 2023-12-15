'use client'

import React from 'react'
import type { Message } from 'ai'

import { useStore } from '@/lib/hooks/use-store'
import { useChatStore } from '@/lib/stores/chat-store'
import { getChatById } from '@/lib/stores/utils'
import { Chat } from '@/components/chat'

import { ChatSessions } from './chat-sessions'

const emptyMessages: Message[] = []

export default function Chats() {
  const _hasHydrated = useStore(useChatStore, state => state._hasHydrated)
  const chats = useStore(useChatStore, state => state.chats)
  const activeChatId = useStore(useChatStore, state => state.activeChatId)

  const chatId = activeChatId
  const chat = getChatById(chats, chatId)

  return (
    <div className="grid flex-1 overflow-hidden lg:grid-cols-[280px_1fr]">
      <ChatSessions className="hidden w-[280px] border-r lg:block" />
      <Chat
        loading={!_hasHydrated}
        id={chatId}
        key={chatId}
        initialMessages={chat?.messages ?? emptyMessages}
      />
    </div>
  )
}
