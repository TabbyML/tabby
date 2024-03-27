'use client'

import React from 'react'
import { useSearchParams } from 'next/navigation'
import type { Message } from 'ai'

import { useStore } from '@/lib/hooks/use-store'
import { addChat } from '@/lib/stores/chat-actions'
import { useChatStore } from '@/lib/stores/chat-store'
import { getChatById } from '@/lib/stores/utils'
import { truncateText } from '@/lib/utils'
import { Chat, ChatRef } from '@/components/chat'
import LoadingWrapper from '@/components/loading-wrapper'
import { ListSkeleton } from '@/components/skeleton'

import { ChatSessions } from './chat-sessions'

const emptyMessages: Message[] = []

export default function Chats() {
  const searchParams = useSearchParams()
  const defaultPrompt = searchParams.get('prompt')?.toString()
  const shouldConsumeDefaultPrompt = React.useRef(!!defaultPrompt)
  const chatRef = React.useRef<ChatRef>(null)

  const _hasHydrated = useStore(useChatStore, state => state._hasHydrated)
  const chats = useStore(useChatStore, state => state.chats)
  const activeChatId = useStore(useChatStore, state => state.activeChatId)
  const chat = getChatById(chats, activeChatId)

  React.useEffect(() => {
    if (!shouldConsumeDefaultPrompt.current) return
    if (!chatRef.current?.append) return

    if (activeChatId && defaultPrompt) {
      // append default prompt
      chatRef.current.append({
        role: 'user',
        content: defaultPrompt
      })
      // store as a new chat
      addChat(activeChatId, truncateText(defaultPrompt))

      shouldConsumeDefaultPrompt.current = false
    }
  }, [chatRef.current?.append])

  return (
    <div className="grid flex-1 overflow-hidden lg:grid-cols-[280px_1fr]">
      <ChatSessions className="hidden w-[280px] border-r lg:block" />
      <LoadingWrapper
        delay={0}
        loading={!_hasHydrated || !activeChatId}
        fallback={
          <div className="mx-auto w-full max-w-2xl pt-4 md:pt-10">
            <ListSkeleton />
          </div>
        }
      >
        <Chat
          id={activeChatId}
          key={activeChatId}
          initialMessages={chat?.messages ?? emptyMessages}
          ref={chatRef}
        />
      </LoadingWrapper>
    </div>
  )
}
