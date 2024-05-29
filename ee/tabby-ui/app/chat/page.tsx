'use client'

import { useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'next/navigation'
import Color from 'color'
import type { ChatMessage, Context, FetcherOptions } from 'tabby-chat-panel'
import { useServer } from 'tabby-chat-panel/react'

import { nanoid } from '@/lib/utils'
import { Chat, ChatRef } from '@/components/chat/chat'

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
  const [pendingMessages, setPendingMessages] = useState<ChatMessage[]>([])

  const chatRef = useRef<ChatRef>(null)

  const searchParams = useSearchParams()
  const from = searchParams.get('from') || undefined
  const isFromVSCode = from === 'vscode'
  const maxWidth = isFromVSCode ? '5xl' : undefined

  useEffect(() => {
    const onMessage = ({
      data
    }: {
      data: {
        style?: string
        themeClass?: string
      }
    }) => {
      // Sync with VSCode CSS variable
      if (data.style) {
        const styleWithHslValue = data.style
          .split(';')
          .filter((style: string) => style)
          .map((style: string) => {
            const [key, value] = style.split(':')
            const styleValue = value.trim()
            const isColorValue =
              styleValue.startsWith('#') || styleValue.startsWith('rgb')
            if (!isColorValue) return `${key}: ${value}`
            const hslValue = convertToHSLColor(styleValue)
            return `${key}: ${hslValue}`
          })
          .join(';')
        document.documentElement.style.cssText = styleWithHslValue
      }

      // Sync with edit theme
      if (data.themeClass) {
        document.documentElement.className = data.themeClass
      }
    }

    window.addEventListener('message', onMessage)
    return () => {
      window.removeEventListener('message', onMessage)
    }
  }, [])

  // VSCode bug: not support shortcuts like copy/paste
  // @see - https://github.com/microsoft/vscode/issues/129178
  useEffect(() => {
    if (!isFromVSCode) return

    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.code === 'KeyC') {
        document.execCommand('copy')
      } else if ((event.ctrlKey || event.metaKey) && event.code === 'KeyX') {
        document.execCommand('cut')
      } else if ((event.ctrlKey || event.metaKey) && event.code === 'KeyV') {
        document.execCommand('paste')
      } else if ((event.ctrlKey || event.metaKey) && event.code === 'KeyA') {
        document.execCommand('selectAll')
      }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
    }
  }, [])

  const sendMessage = (message: ChatMessage) => {
    if (chatRef.current) {
      chatRef.current.sendUserChat(message)
    } else {
      const newPendingMessages = [...pendingMessages]
      newPendingMessages.push(message)
      setPendingMessages(newPendingMessages)
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

  const onChatLoaded = () => {
    pendingMessages.forEach(sendMessage)
    setPendingMessages([])
  }

  const onNavigateToContext = (context: Context) => {
    server?.navigate(context)
  }

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
      onNavigateToContext={onNavigateToContext}
      onLoaded={onChatLoaded}
      maxWidth={maxWidth}
    />
  )
}
