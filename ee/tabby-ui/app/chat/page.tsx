'use client'

import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import Image from 'next/image'
import { useSearchParams } from 'next/navigation'
import tabbyUrl from '@/assets/tabby.png'
import { useWindowSize } from '@uidotdev/usehooks'
import Color from 'color'
import { ErrorBoundary } from 'react-error-boundary'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import {
  TABBY_CHAT_PANEL_API_VERSION,
  type ChatMessage,
  type Context,
  type ErrorMessage,
  type FetcherOptions,
  type InitRequest,
  type NavigateOpts
} from 'tabby-chat-panel'
import { useServer } from 'tabby-chat-panel/react'

import { nanoid } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { IconSpinner } from '@/components/ui/icons'
import { Chat, ChatRef } from '@/components/chat/chat'
import { MemoizedReactMarkdown } from '@/components/markdown'

import './page.css'

import { saveFetcherOptions } from '@/lib/tabby/token-management'

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
  const [pendingRelevantContexts, setPendingRelevantContexts] = useState<
    Context[]
  >([])
  const [errorMessage, setErrorMessage] = useState<ErrorMessage | null>(null)
  const [isRefreshLoading, setIsRefreshLoading] = useState(false)

  const chatRef = useRef<ChatRef>(null)
  const [chatLoaded, setChatLoaded] = useState(false)
  const { width } = useWindowSize()
  const prevWidthRef = useRef(width)
  const chatInputRef = useRef<HTMLTextAreaElement>(null)

  const searchParams = useSearchParams()
  const client = searchParams.get('client') as ClientType
  const isInEditor = !!client || undefined
  const useMacOSKeyboardEventHandler = useRef<boolean>()

  const sendMessage = (message: ChatMessage) => {
    if (chatRef.current) {
      chatRef.current.sendUserChat(message)
    } else {
      const newPendingMessages = [...pendingMessages]
      newPendingMessages.push(message)
      setPendingMessages(newPendingMessages)
    }
  }

  const addRelevantContext = (ctx: Context) => {
    if (chatRef.current) {
      chatRef.current.addRelevantContext(ctx)
    } else {
      const newPendingRelevantContexts = [...pendingRelevantContexts]
      newPendingRelevantContexts.push(ctx)
      setPendingRelevantContexts(newPendingRelevantContexts)
    }
  }

  const server = useServer({
    init: (request: InitRequest) => {
      if (chatRef.current) return

      // save fetcherOptions to sessionStorage
      if (client) {
        saveFetcherOptions(request.fetcherOptions)
      }

      setActiveChatId(nanoid())
      setIsInit(true)
      setFetcherOptions(request.fetcherOptions)
      useMacOSKeyboardEventHandler.current =
        request.useMacOSKeyboardEventHandler
    },
    sendMessage: (message: ChatMessage) => {
      return sendMessage(message)
    },
    showError: (errorMessage: ErrorMessage) => {
      setErrorMessage(errorMessage)
    },
    cleanError: () => {
      setErrorMessage(null)
    },
    addRelevantContext: context => {
      return addRelevantContext(context)
    },
    updateTheme: (style, themeClass) => {
      const styleWithHslValue = style
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

      // Sync with edit theme
      document.documentElement.className =
        themeClass + ` client client-${client}`
    }
  })

  useEffect(() => {
    const onFocus = () => {
      // When we receive top level focus, just focus chatInputRef
      setTimeout(() => {
        chatInputRef.current?.focus()
      }, 0)
    }

    window.addEventListener('focus', onFocus)
    return () => {
      window.removeEventListener('focus', onFocus)
    }
  })

  useEffect(() => {
    const dispatchKeyboardEvent = (
      type: 'keydown' | 'keyup' | 'keypress',
      event: KeyboardEvent
    ) => {
      server?.onKeyboardEvent(type, {
        code: event.code,
        isComposing: event.isComposing,
        key: event.key,
        altKey: event.altKey,
        ctrlKey: event.ctrlKey,
        metaKey: event.metaKey,
        shiftKey: event.shiftKey,
        location: event.location,
        repeat: event.repeat,
        // keyCode is deprecated, but still required for VSCode on Windows
        keyCode: event.keyCode
      })
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (useMacOSKeyboardEventHandler.current) {
        // Workaround for vscode webview issue:
        // shortcut (cmd+a, cmd+c, cmd+v, cmd+x) not work in nested iframe in vscode webview
        // see https://github.com/microsoft/vscode/issues/129178
        if (event.metaKey && event.code === 'KeyC') {
          document.execCommand('copy')
        } else if (event.metaKey && event.code === 'KeyX') {
          document.execCommand('cut')
        } else if (event.metaKey && event.code === 'KeyV') {
          document.execCommand('paste')
        } else if (event.metaKey && event.code === 'KeyA') {
          document.execCommand('selectAll')
        } else {
          dispatchKeyboardEvent('keydown', event)
        }
      } else {
        dispatchKeyboardEvent('keydown', event)
      }
    }

    const onKeyUp = (event: KeyboardEvent) => {
      dispatchKeyboardEvent('keyup', event)
    }

    const onKeyPress = (event: KeyboardEvent) => {
      dispatchKeyboardEvent('keypress', event)
    }

    window.addEventListener('keydown', onKeyDown)
    window.addEventListener('keyup', onKeyUp)
    window.addEventListener('keypress', onKeyPress)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('keyup', onKeyUp)
      window.removeEventListener('keypress', onKeyPress)
    }
  }, [server, client])

  useEffect(() => {
    if (server) {
      server?.onLoaded({
        apiVersion: TABBY_CHAT_PANEL_API_VERSION
      })
    }
  }, [server])

  useLayoutEffect(() => {
    if (!chatLoaded) return
    if (
      width &&
      isInit &&
      fetcherOptions &&
      !errorMessage &&
      !prevWidthRef.current
    ) {
      chatRef.current?.focus()
    }
    prevWidthRef.current = width
  }, [width, chatLoaded])

  const onChatLoaded = () => {
    pendingRelevantContexts.forEach(addRelevantContext)
    pendingMessages.forEach(sendMessage)
    setPendingRelevantContexts([])
    setPendingMessages([])
    setChatLoaded(true)
  }

  const onNavigateToContext = (context: Context, opts?: NavigateOpts) => {
    server?.navigate(context, opts)
  }

  const refresh = async () => {
    setIsRefreshLoading(true)
    await server?.refresh()
    setIsRefreshLoading(false)
  }

  function StaticContent({ children }: { children: React.ReactNode }) {
    return (
      <div
        className="h-screen w-screen"
        style={{
          padding: client == 'intellij' ? '20px' : '5px 18px'
        }}
      >
        <div className="flex items-center" style={{ marginBottom: '0.55em' }}>
          <Image
            src={tabbyUrl}
            alt="logo"
            className="rounded-full"
            style={{
              background: 'rgb(232, 226, 210)',
              marginRight: '0.375em',
              padding: '0.15em'
            }}
            width={18}
          />
          <p className="font-semibold">Tabby</p>
        </div>
        {children}
      </div>
    )
  }

  function ErrorBoundaryFallback({ error }: { error: Error }) {
    return (
      <StaticContent>
        <p className="mb-1.5 mt-2 font-semibold">Something went wrong</p>
        <p>{error.message}</p>
        <Button
          className="mt-5 flex items-center gap-x-2 text-sm leading-none"
          onClick={refresh}
        >
          {isRefreshLoading && <IconSpinner />}
          Refresh
        </Button>
      </StaticContent>
    )
  }

  if (errorMessage) {
    return (
      <StaticContent>
        <>
          {errorMessage.title && (
            <p className="mb-1.5 mt-2 font-semibold">{errorMessage.title}</p>
          )}
          <MemoizedReactMarkdown
            className="prose max-w-none break-words dark:prose-invert prose-p:leading-relaxed prose-pre:mt-1 prose-pre:p-0"
            remarkPlugins={[remarkGfm, remarkMath]}
          >
            {errorMessage.content}
          </MemoizedReactMarkdown>
          <Button
            className="mt-5 flex items-center gap-x-2 text-sm leading-none"
            onClick={refresh}
          >
            {isRefreshLoading && <IconSpinner />}
            Refresh
          </Button>
        </>
      </StaticContent>
    )
  }

  if (!isInit || !fetcherOptions) {
    return (
      <StaticContent>
        <>
          <p className="opacity-80">
            Welcome to Tabby Chat! Just a moment while we get things ready...
          </p>
          <IconSpinner
            className="mx-auto"
            style={{
              marginTop: '1.25em',
              width: '0.875em',
              height: '0.875em'
            }}
          />
        </>
      </StaticContent>
    )
  }

  return (
    <ErrorBoundary FallbackComponent={ErrorBoundaryFallback}>
      <Chat
        chatId={activeChatId}
        key={activeChatId}
        ref={chatRef}
        chatInputRef={chatInputRef}
        onNavigateToContext={onNavigateToContext}
        onLoaded={onChatLoaded}
        maxWidth={client === 'vscode' ? '5xl' : undefined}
        onCopyContent={isInEditor && server?.onCopy}
        onSubmitMessage={isInEditor && server?.onSubmitMessage}
        onApplyInEditor={isInEditor && server?.onApplyInEditor}
      />
    </ErrorBoundary>
  )
}

type ClientType = 'vscode' | 'intellij' | 'vim' | 'eclipse' | null
