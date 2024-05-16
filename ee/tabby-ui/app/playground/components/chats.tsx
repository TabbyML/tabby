'use client'

import React from 'react'

import { useDebounceCallback } from '@/lib/hooks/use-debounce'
import useRouterStuff from '@/lib/hooks/use-router-stuff'
import { useStore } from '@/lib/hooks/use-store'
import { addChat, updateMessages } from '@/lib/stores/chat-actions'
import { useChatStore } from '@/lib/stores/chat-store'
import { getChatById } from '@/lib/stores/utils'
import fetcher from '@/lib/tabby/fetcher'
import { Context as ChatContext, QuestionAnswerPair } from '@/lib/types/chat'
import { truncateText } from '@/lib/utils'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Chat, ChatRef } from '@/components/chat/chat'
import { BANNER_HEIGHT, useShowDemoBanner } from '@/components/demo-banner'
import LoadingWrapper from '@/components/loading-wrapper'
import { ListSkeleton } from '@/components/skeleton'

import { ChatSessions } from './chat-sessions'

const emptyQaParise: QuestionAnswerPair[] = []

export default function Chats() {
  const { searchParams, updateSearchParams } = useRouterStuff()
  const initialMessage = searchParams.get('initialMessage')?.toString()
  const shouldConsumeInitialMessage = React.useRef(!!initialMessage)
  const chatRef = React.useRef<ChatRef>(null)
  const chatContainerRef = React.useRef<HTMLDivElement>(null)
  const [isShowDemoBanner] = useShowDemoBanner()

  const _hasHydrated = useStore(useChatStore, state => state._hasHydrated)

  const activeChatId = useStore(useChatStore, state => state.activeChatId)
  const storedChatList = useStore(useChatStore, state => state.chats)
  const storedChat = getChatById(storedChatList, activeChatId)

  const initialMessages = React.useMemo(() => {
    return storedChat?.messages?.filter(o => !!o.user)
  }, [activeChatId])

  React.useEffect(() => {
    const onMessage = async (event: MessageEvent) => {
      if (event.origin !== window.origin || !event.data) {
        return
      }
      if (!chatRef.current || chatRef.current.isLoading) return

      const { data } = event
      if (data.action === 'sendUserChat') {
        chatRef.current.sendUserChat({
          ...data.payload
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

  const persistChat = useDebounceCallback(
    (chatId: string, messages: QuestionAnswerPair[]) => {
      if (!storedChat && messages?.length) {
        addChat(activeChatId, truncateText(messages?.[0]?.user?.message))
      } else if (storedChat) {
        updateMessages(chatId, messages)
      }
    },
    1000
  )

  const onThreadUpdates = (messages: QuestionAnswerPair[]) => {
    if (activeChatId) {
      persistChat.run(activeChatId, messages)
    }
  }

  const onNavigateToContext = (context: ChatContext) => {
    // todo check if is embed
    if (!context.filePath || !context.link) return
    const isInIframe = window.self !== window.top
    if (isInIframe) {
      window.top?.postMessage({
        action: 'navigateToContext',
        path: context.filePath,
        line: context.range.start
      })
    } else {
      window.open(context.link)
    }
  }

  const onChatLoaded = () => {
    if (!shouldConsumeInitialMessage.current) return
    if (!chatRef.current?.sendUserChat) return
    if (activeChatId && initialMessage) {
      // request initialMessage
      chatRef.current
        .sendUserChat({
          message: initialMessage
        })
        .then(() => {
          // Remove the initialMessage params after the request is completed.
          updateSearchParams({
            del: 'initialMessage'
          })
        })

      shouldConsumeInitialMessage.current = false
    }
  }

  React.useEffect(() => {
    return () => persistChat.flush()
  }, [])

  const style = isShowDemoBanner
    ? { height: `calc(100vh - ${BANNER_HEIGHT})` }
    : { height: '100vh' }
  return (
    <div className="grid flex-1 overflow-hidden lg:grid-cols-[280px_1fr]">
      <ChatSessions
        className="hidden w-[280px] border-r lg:block"
        style={style}
      />
      <LoadingWrapper
        delay={0}
        loading={!_hasHydrated || !activeChatId}
        fallback={
          <div className="mx-auto w-full max-w-2xl pt-4 md:pt-10">
            <ListSkeleton />
          </div>
        }
      >
        <ScrollArea
          className="transition-all"
          style={style}
          ref={chatContainerRef}
        >
          <div>
            <Chat
              chatId={activeChatId as string}
              key={activeChatId}
              initialMessages={initialMessages ?? emptyQaParise}
              ref={chatRef}
              container={
                (chatContainerRef?.current?.children[1] as HTMLDivElement) ||
                undefined
              }
              onThreadUpdates={onThreadUpdates}
              onNavigateToContext={onNavigateToContext}
              onLoaded={onChatLoaded}
              fetcher={
                ((url: string, init?: RequestInit) =>
                  fetcher(url, {
                    ...init,
                    responseFormatter(response) {
                      return response
                    },
                    errorHandler(response) {
                      throw new Error(
                        response ? String(response.status) : 'Fail to fetch'
                      )
                    }
                  })) as typeof fetch
              }
              // api="/v1/chat/completion"
            />
          </div>
        </ScrollArea>
      </LoadingWrapper>
    </div>
  )
}
