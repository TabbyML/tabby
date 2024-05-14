'use client'

import React from 'react'

import { usePatchFetch } from '@/lib/hooks/use-patch-fetch'
import useRouterStuff from '@/lib/hooks/use-router-stuff'
import { useStore } from '@/lib/hooks/use-store'
import { useChatStore } from '@/lib/stores/chat-store'
import { nanoid } from '@/lib/utils'
import LoadingWrapper from '@/components/loading-wrapper'
import {
  Chat,
  Context as ChatContext,
  ChatRef,
  QuestionAnswerPair
} from '@/components/new-chat/chat'
import { ListSkeleton } from '@/components/skeleton'

import { ChatSessions } from './chat-sessions'

const emptyQaParise: QuestionAnswerPair[] = []

export default function Chats() {
  usePatchFetch()
  const { searchParams, updateSearchParams } = useRouterStuff()
  const initialMessage = searchParams.get('initialMessage')?.toString()
  const shouldConsumeInitialMessage = React.useRef(!!initialMessage)
  const chatRef = React.useRef<ChatRef>(null)

  const _hasHydrated = useStore(useChatStore, state => state._hasHydrated)

  // todo Use localStorage to save chat data.
  const activeChatId = useStore(useChatStore, state => state.activeChatId)

  React.useEffect(() => {
    const onMessage = async (event: MessageEvent) => {
      if (event.origin !== window.origin || !event.data) {
        return
      }
      if (!chatRef.current || chatRef.current.isLoading) return

      const { data } = event
      if (data.action === 'append') {
        chatRef.current.sendUserChat({
          id: nanoid(),
          // todo should be context and content
          message: data.payload
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
  }, [])

  const onThreadUpdates = (messages: QuestionAnswerPair[]) => {
    console.log(messages)
  }

  const onNavigateToContext = (context: ChatContext) => {
    console.log(context)
  }

  const onChatLoaded = () => {
    if (!shouldConsumeInitialMessage.current) return
    if (!chatRef.current?.sendUserChat) return
    if (activeChatId && initialMessage) {
      // request initialMessage
      chatRef.current
        .sendUserChat({
          id: nanoid(),
          message: initialMessage
        })
        .then(() => {
          // Remove the initialMessage params after the request is completed.
          updateSearchParams({
            del: 'initialMessage'
          })
        })
      // todo store as a new chat with localstorage
      // addChat(activeChatId, truncateText(initialMessage))

      shouldConsumeInitialMessage.current = false
    }
  }

  // const sent = React.useRef(false)
  // React.useEffect(() => {
  //   setTimeout(() => {
  //     if (sent.current) return
  //     chatRef.current?.sendUserChat({
  //       id: nanoid(),
  //       message: 'explain',
  //       selectContext: {
  //         kind: 'file',
  //         content: 'do not display',
  //         range: {
  //           start: 4,
  //           end: 10
  //         },
  //         filePath: 'ee/tabby-ui/layout.tsx',
  //         link: 'xxxxxx'
  //       },
  //       relevantContext: [
  //         {
  //           kind: 'file',
  //           content: 'do not display',
  //           range: {
  //             start: 99,
  //             end: 100
  //           },
  //           filePath: 'ee/tabby-ui/path/to/file/test.tsx',
  //           link: 'xxxxxx'
  //         }
  //       ]
  //     })
  //     sent.current = true
  //   }, 5000)
  // }, [])

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
          chatId={activeChatId as string}
          key={activeChatId}
          initialMessages={emptyQaParise}
          ref={chatRef}
          onThreadUpdates={onThreadUpdates}
          onNavigateToContext={onNavigateToContext}
          onLoaded={onChatLoaded}
        />
      </LoadingWrapper>
    </div>
  )
}
