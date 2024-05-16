import React from 'react'
import { Message } from 'ai'
import { useChat, type UseChatHelpers } from 'ai/react'
import { findIndex, omit } from 'lodash-es'

import {
  AssistantMessage,
  Context,
  FileContext,
  MessageActionType,
  QuestionAnswerPair,
  UserMessage,
  UserMessageWithOptionalId
} from '@/lib/types/chat'
import { cn, nanoid } from '@/lib/utils'

import { ChatPanel } from './chat-panel'
import { ChatScrollAnchor } from './chat-scroll-anchor'
import { EmptyScreen } from './empty-screen'
import { QuestionAnswerList } from './question-answer'
import { usePatchFetch } from './use-patch-fetch'

type ChatContextValue = {
  isLoading: boolean
  handleMessageAction: (
    userMessageId: string,
    action: MessageActionType
  ) => void
  onNavigateToContext?: (context: Context) => void
  container?: HTMLDivElement
}

export const ChatContext = React.createContext<ChatContextValue>(
  {} as ChatContextValue
)

function toMessages(qaPairs: QuestionAnswerPair[] | undefined): Message[] {
  if (!qaPairs?.length) return []
  let result: Message[] = []
  for (let pair of qaPairs) {
    let { user, assistant } = pair
    if (user) {
      result.push(userMessageToMessage(user))
    }
    if (assistant) {
      result.push({
        role: 'assistant',
        id: assistant.id,
        content: assistant.message
      })
    }
  }
  return result
}

function userMessageToMessage(userMessage: UserMessage): Message {
  const { selectContext, message, id } = userMessage
  return {
    id,
    role: 'user',
    content: message + fileContextToMessageContent(selectContext)
  }
}

function fileContextToMessageContent(context: FileContext | undefined): string {
  if (!context) return ''
  const { content, language } = context
  return `\n${'```'}${language ?? ''} is_reference=1\n ${content} \n${'```'}\n`
}

export interface ChatRef extends Omit<UseChatHelpers, 'append' | 'messages'> {
  sendUserChat: (
    message: UserMessageWithOptionalId
  ) => Promise<string | null | undefined>
}

interface ChatProps extends React.ComponentProps<'div'> {
  chatId: string
  api?: string
  fetcher?: typeof fetch
  headers?: Record<string, string> | Headers
  initialMessages?: QuestionAnswerPair[]
  onLoaded?: () => void
  onThreadUpdates: (messages: QuestionAnswerPair[]) => void
  onNavigateToContext: (context: Context) => void
  container?: HTMLDivElement
}

function ChatRenderer(
  {
    className,
    chatId,
    initialMessages,
    headers,
    api = '/v1/chat/completions',
    onLoaded,
    onThreadUpdates,
    onNavigateToContext,
    container,
    fetcher
  }: ChatProps,
  ref: React.ForwardedRef<ChatRef>
) {
  usePatchFetch({ api, fetcher })
  const [qaPairs, setQaPairs] = React.useState(initialMessages ?? [])
  const loaded = React.useRef(false)
  const transformedInitialMessages = React.useMemo(() => {
    return toMessages(initialMessages)
  }, [])

  const useChatHelpers = useChat({
    initialMessages: transformedInitialMessages,
    id: chatId,
    headers,
    body: {
      id: chatId
    }
  })

  const {
    messages,
    append,
    stop,
    isLoading,
    input,
    setInput,
    setMessages,
    error
  } = useChatHelpers

  const onDeleteMessage = async (userMessageId: string) => {
    // Stop generating first.
    stop()

    const nextQaPairs = qaPairs.filter(o => o.user.id !== userMessageId)
    setQaPairs(nextQaPairs)
    // setmessage returns by useChatHelpers
    setMessages(toMessages(nextQaPairs))
  }

  const onRegenerateResponse = async (userMessageid: string) => {
    const qaPairIndex = findIndex(qaPairs, o => o.user.id === userMessageid)
    if (qaPairIndex > -1) {
      const qaPair = qaPairs[qaPairIndex]
      let nextQaPairs: QuestionAnswerPair[] = [
        ...qaPairs.slice(0, qaPairIndex),
        {
          ...qaPair,
          assistant: {
            ...qaPair.assistant,
            id: qaPair.assistant?.id || nanoid(),
            // clear assistant message
            message: '',
            // clear error
            error: undefined
          }
        }
      ]
      setQaPairs(nextQaPairs)
      // exclude the last pair
      setMessages(toMessages(nextQaPairs.slice(0, -1)))
      // 'append' the userMessage of last pair to trigger chat api
      return append(userMessageToMessage(qaPair.user))
    }
  }

  // Reload the last AI chat response
  const onReload = async () => {
    if (!qaPairs?.length) return
    const lastUserMessageId = qaPairs[qaPairs.length - 1].user.id
    return onRegenerateResponse(lastUserMessageId)
  }

  const onStop = () => {
    stop()
  }

  const handleMessageAction = (
    userMessageId: string,
    actionType: 'delete' | 'regenerate'
  ) => {
    switch (actionType) {
      case 'delete':
        onDeleteMessage(userMessageId)
        break
      case 'regenerate':
        onRegenerateResponse(userMessageId)
        break
      default:
        break
    }
  }

  React.useEffect(() => {
    if (!isLoading || !qaPairs?.length || !messages?.length) return

    const lastQaPairs = qaPairs[qaPairs.length - 1]
    const lastMessage = messages[messages.length - 1]
    const isAssistantMessageStreaming =
      lastMessage?.role === 'assistant' &&
      messages[messages.length - 2]?.id === lastQaPairs.user.id
    if (isAssistantMessageStreaming) {
      setQaPairs(prev => {
        const assisatntMessage = prev[prev.length - 1].assistant
        const nextAssistantMessage: AssistantMessage = {
          ...assisatntMessage,
          id: assisatntMessage?.id || lastMessage.id,
          message: lastMessage.content,
          error: undefined
        }
        // merge assistantMessage
        return [
          ...prev.slice(0, prev.length - 1),
          {
            ...lastQaPairs,
            assistant: nextAssistantMessage
          }
        ]
      })
    }
  }, [messages, isLoading])

  React.useEffect(() => {
    if (error && qaPairs?.length) {
      setQaPairs(prev => {
        let lastQaPairs = prev[prev.length - 1]
        return [
          ...prev.slice(0, prev.length - 1),
          {
            ...lastQaPairs,
            assistant: {
              ...lastQaPairs.assistant,
              id: lastQaPairs.assistant?.id || nanoid(),
              message: lastQaPairs.assistant?.message ?? '',
              error: error?.message === '401' ? 'Unauthorized' : 'Fail to fetch'
            }
          }
        ]
      })
    }
  }, [error])

  const sendUserChat = (userMessage: UserMessageWithOptionalId) => {
    // If no id is provided, set a fallback id.
    const newUserMessage = {
      ...userMessage,
      id: userMessage.id ?? nanoid()
    }
    setQaPairs(pairs => [
      ...pairs,
      {
        user: newUserMessage,
        // For placeholder, and it also conveniently handles streaming responses and displays reference context.
        assistant: {
          id: nanoid(),
          message: '',
          error: undefined
        }
      }
    ])
    return append(userMessageToMessage(newUserMessage))
  }

  const handleSubmit = async (value: string) => {
    return sendUserChat({
      message: value
    })
  }

  React.useEffect(() => {
    if (!loaded.current) return
    onThreadUpdates(qaPairs)
  }, [qaPairs])

  React.useImperativeHandle(
    ref,
    () => {
      return {
        ...omit(useChatHelpers, ['append', 'messages']),
        sendUserChat
      }
    },
    [useChatHelpers]
  )

  React.useEffect(() => {
    if (loaded?.current) return

    loaded.current = true
    onLoaded?.()
  }, [])

  return (
    <ChatContext.Provider
      value={{
        isLoading: useChatHelpers.isLoading,
        onNavigateToContext,
        handleMessageAction,
        container
      }}
    >
      <div className="flex justify-center overflow-x-hidden">
        <div className="w-full max-w-2xl px-4">
          <div className={cn('pb-[200px] pt-4 md:pt-10', className)}>
            {qaPairs?.length ? (
              <QuestionAnswerList messages={qaPairs} />
            ) : (
              <EmptyScreen setInput={useChatHelpers.setInput} />
            )}
            <ChatScrollAnchor trackVisibility={isLoading} />
          </div>
          <ChatPanel
            onSubmit={handleSubmit}
            className="fixed inset-x-0 bottom-0 lg:ml-[280px]"
            id={chatId}
            isLoading={isLoading}
            stop={onStop}
            append={append}
            reload={onReload}
            messages={messages}
            input={input}
            setInput={setInput}
            setMessages={setMessages}
          />
        </div>
      </div>
    </ChatContext.Provider>
  )
}

export const Chat = React.forwardRef<ChatRef, ChatProps>(ChatRenderer)
