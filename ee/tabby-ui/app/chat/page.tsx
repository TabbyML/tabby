'use client'

import { useRef, useState, useEffect } from 'react'
import type { ChatMessage, Context, FetcherOptions } from 'tabby-chat-panel'
import { useServer } from 'tabby-chat-panel/react'

import { cn } from '@/lib/utils'
import { nanoid } from '@/lib/utils'
import { Chat, ChatRef } from '@/components/chat/chat'
import { QuestionAnswerPair } from '@/lib/types/chat'
import Color from "color";

import './page.css'

const convertToHSLColor = (style: string) => {
  return Color(style)
    .hsl()
    .toString()
    .replace(/hsla?\(/, '')
    .replace(')', '')
    .split(',')
    .slice(0, 3)
    .map((item, idx) => {
      if (idx === 0) return parseFloat(item).toFixed(1)
      return item
    })
    .join('')
}

export default function ChatPage() {
  const [isInit, setIsInit] = useState(false)
  const [fetcherOptions, setFetcherOptions] = useState<FetcherOptions | null>(
    null
  )
  const [activeChatId, setActiveChatId] = useState('')
  const [initialMessages, setInitialMessages] = useState<QuestionAnswerPair[]>([])
  const chatRef = useRef<ChatRef>(null)

  useEffect(() => {
    window.addEventListener('message', ({ data }) => {
      // Sync with VSCode CSS variable
      if (data.style) {
        const styleWithHslValue = data.style
          .split(';')
          .filter((style: string) => style)
          .map((style: string) => {
            const [key, value] = style.split(':')
            const styleValue = value.trim()
            const isColorValue = styleValue.startsWith('#') || styleValue.startsWith('rgb')
            if (!isColorValue) return `${key}: ${value}`
            const hslValue = convertToHSLColor(styleValue)
            return `${key}: ${hslValue}`
          })
          .join(';')
        document.documentElement.style.cssText = styleWithHslValue
      }

      // Sync with edit theme
      if (data.themeClass) {
        document.documentElement.className = data.themeClass;
      }
    })
  }, []);

  const sendMessage = (message: ChatMessage) => {
    if (chatRef.current) {
      chatRef.current.sendUserChat(message)
    } else {
      const newInitialMessages = [...initialMessages]
      newInitialMessages.push({
        user: {
          ...message,
          id: nanoid()
        }
      })
      setInitialMessages(newInitialMessages)
    }
  }

  const server = useServer({
    init: request => {
      if (chatRef.current) return
      setActiveChatId(nanoid())
      setIsInit(true)
      setFetcherOptions(request.fetcherOptions)
    },
    sendMessage: (message: ChatMessage) => {
      return sendMessage(message)
    }
  })

  const onNavigateToContext = (context: Context) => {
    server?.navigate(context)
  }

  if (!isInit || !fetcherOptions) return <></>
  const headers = {
    Authorization: `Bearer ${fetcherOptions.authorization}`
  }
  return (
    <div className="content">
      <Chat
        chatId={activeChatId}
        key={activeChatId}
        ref={chatRef}
        headers={headers}
        initialMessages={initialMessages}
        onThreadUpdates={() => {}}
        onNavigateToContext={onNavigateToContext}
      />
    </div>
  )
}
