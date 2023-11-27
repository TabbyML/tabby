'use client'

import React from 'react'
import { Chat } from '@/components/chat'
import { useChatStore } from '@/lib/stores/chat-store'
import { getChatById } from '@/lib/stores/utils'
import { ChatSessions } from './chat-sessions'
import { useStore } from '@/lib/hooks/use-store'
import type { Message } from 'ai'

const emptyMessages: Message[] = []

export default function Chats() {
  const _hasHydrated = useStore(useChatStore, state => state._hasHydrated)
  const chats = useStore(useChatStore, state => state.chats)
  const activeChatId = useStore(useChatStore, state => state.activeChatId)

  const chatId = activeChatId
  const chat = getChatById(chats, chatId)

  return (
    <div className="grid flex-1 overflow-hidden lg:grid-cols-[280px_1fr]">
      <ChatSessions className="hidden w-[280px] border-r bg-zinc-100/40 dark:bg-zinc-800/40 lg:block" />
      <Chat
        loading={!_hasHydrated}
        id={chatId}
        key={chatId}
        initialMessages={chat?.messages ?? emptyMessages}
      />
    </div>
  )
}
