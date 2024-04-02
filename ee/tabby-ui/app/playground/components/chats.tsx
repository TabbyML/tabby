'use client'

import React from 'react'
import { type Message } from 'ai'

import useRouterStuff from '@/lib/hooks/use-router-stuff'
import { useStore } from '@/lib/hooks/use-store'
import { addChat } from '@/lib/stores/chat-actions'
import { useChatStore } from '@/lib/stores/chat-store'
import { getChatById } from '@/lib/stores/utils'
import { nanoid, truncateText } from '@/lib/utils'
import { Chat, ChatRef } from '@/components/chat'
import LoadingWrapper from '@/components/loading-wrapper'
import { ListSkeleton } from '@/components/skeleton'

import { ChatSessions } from './chat-sessions'

const emptyMessages: Message[] = []

export default function Chats() {
  const { searchParams, updateSearchParams } = useRouterStuff()
  const initialMessage = searchParams.get('initialMessage')?.toString()
  const shouldConsumeInitialMessage = React.useRef(!!initialMessage)
  const chatRef = React.useRef<ChatRef>(null)

  const _hasHydrated = useStore(useChatStore, state => state._hasHydrated)
  const chats = useStore(useChatStore, state => state.chats)
  const activeChatId = useStore(useChatStore, state => state.activeChatId)
  const chat = getChatById(chats, activeChatId)

  React.useEffect(() => {
    if (!shouldConsumeInitialMessage.current) return
    if (!chatRef.current?.append) return

    if (activeChatId && initialMessage) {
      // request initialMessage
      chatRef.current
        .append({
          role: 'user',
          content: initialMessage
        })
        .then(() => {
          // Remove the initialMessage params after the request is completed.
          updateSearchParams({
            del: 'initialMessage'
          })
        })
      // store as a new chat
      addChat(activeChatId, truncateText(initialMessage))

      shouldConsumeInitialMessage.current = false
    }
  }, [chatRef.current?.append])

  React.useEffect(() => {
    const onMessage = async (event: MessageEvent) => {
      if (event.origin !== window.origin || !event.data) {
        return
      }
      if (!chatRef.current || chatRef.current.isLoading) return

      const { data } = event
      if (data.action === 'append') {
        chatRef.current.append({
          id: nanoid(),
          role: 'user',
          content: data.payload
        })
        return
      }

      if (data.action === 'stop') {
        chatRef.current.stop()
      }
    }

    window.addEventListener('message', onMessage)

    return () => {
      window.removeEventListener('message', onMessage)
    }
  }, [chatRef.current])

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
