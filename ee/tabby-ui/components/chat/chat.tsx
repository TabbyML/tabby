import React from 'react'
import { Message } from 'ai'
import { findIndex } from 'lodash-es'
import type { Context, NavigateOpts } from 'tabby-chat-panel'

import { useDebounceCallback } from '@/lib/hooks/use-debounce'
import { useLatest } from '@/lib/hooks/use-latest'
import { filename2prism } from '@/lib/language-utils'
import {
  AnswerRequest,
  AssistantMessage,
  MessageActionType,
  QuestionAnswerPair,
  UserMessage,
  UserMessageWithOptionalId
} from '@/lib/types/chat'
import { cn, nanoid } from '@/lib/utils'

import { useTabbyAnswer } from '../../lib/hooks/use-tabby-answer'
import { ListSkeleton } from '../skeleton'
import { ChatPanel } from './chat-panel'
import { ChatScrollAnchor } from './chat-scroll-anchor'
import { EmptyScreen } from './empty-screen'
import { QuestionAnswerList } from './question-answer'

type ChatContextValue = {
  isLoading: boolean
  qaPairs: QuestionAnswerPair[]
  handleMessageAction: (
    userMessageId: string,
    action: MessageActionType
  ) => void
  onNavigateToContext?: (context: Context, opts?: NavigateOpts) => void
  onClearMessages: () => void
  container?: HTMLDivElement
  onCopyContent?: (value: string) => void
  client?: string
  onApplyInEditor?: (value: string) => void
  clientSelectedContext: Context[]
  removeClientSelectedContext: (index: number) => void
}

export const ChatContext = React.createContext<ChatContextValue>(
  {} as ChatContextValue
)

function toMessages(qaPairs: QuestionAnswerPair[] | undefined): Message[] {
  if (!qaPairs?.length) return []
  let result: Message[] = []

  const len = qaPairs.length
  for (let i = 0; i < qaPairs.length; i++) {
    const pair = qaPairs[i]
    let { user, assistant } = pair
    if (user) {
      result.push(
        userMessageToMessage(user, {
          // if it's not the latest user prompt, the message should include the fileContext converted into a code block.
          includeTransformedSelectContext: len > 1 && i !== len - 1
        })
      )
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

function userMessageToMessage(
  userMessage: UserMessage,
  {
    includeTransformedSelectContext
  }: {
    includeTransformedSelectContext?: boolean
  }
): Message {
  const { message, id } = userMessage
  return {
    id,
    role: 'user',
    content: includeTransformedSelectContext
      ? message + selectContextToMessageContent(userMessage.selectContext)
      : message
  }
}

function selectContextToMessageContent(
  context: UserMessage['selectContext']
): string {
  if (!context || !context.content) return ''
  const { content, filepath } = context
  const language = filename2prism(filepath)?.[0]
  return `\n${'```'}${language ?? ''}\n${content ?? ''}\n${'```'}\n`
}

export interface ChatRef {
  sendUserChat: (
    message: UserMessageWithOptionalId
  ) => Promise<string | null | undefined>
  stop: () => void
  isLoading: boolean
  addClientSelectedContext: (context: Context) => void
}

interface ChatProps extends React.ComponentProps<'div'> {
  chatId: string
  api?: string
  fetcher?: typeof fetch
  headers?: Record<string, string> | Headers
  initialMessages?: QuestionAnswerPair[]
  onLoaded?: () => void
  onThreadUpdates: (messages: QuestionAnswerPair[]) => void
  onNavigateToContext: (context: Context, opts?: NavigateOpts) => void
  container?: HTMLDivElement
  docQuery?: boolean
  generateRelevantQuestions?: boolean
  maxWidth?: string
  welcomeMessage?: string
  promptFormClassname?: string
  onCopyContent?: (value: string) => void
  client?: string
  onSubmitMessage?: (msg: string, relevantContext?: Context[]) => Promise<void>
  onApplyInEditor?: (value: string) => void
}

function ChatRenderer(
  {
    className,
    chatId,
    initialMessages,
    headers,
    api = '/v1beta/answer',
    onLoaded,
    onThreadUpdates,
    onNavigateToContext,
    container,
    fetcher,
    docQuery,
    generateRelevantQuestions,
    maxWidth,
    welcomeMessage,
    promptFormClassname,
    onCopyContent,
    client,
    onSubmitMessage,
    onApplyInEditor
  }: ChatProps,
  ref: React.ForwardedRef<ChatRef>
) {
  const [initialized, setInitialzed] = React.useState(false)
  const isOnLoadExecuted = React.useRef(false)
  const [qaPairs, setQaPairs] = React.useState(initialMessages ?? [])
  const [input, setInput] = React.useState<string>('')
  const [clientSelectedContext, setClientSelectedContext] = React.useState<
    Context[]
  >([])

  const { triggerRequest, isLoading, error, answer, stop } = useTabbyAnswer({
    api,
    headers,
    fetcher
  })

  const onDeleteMessage = async (userMessageId: string) => {
    // Stop generating first.
    stop()

    const nextQaPairs = qaPairs.filter(o => o.user.id !== userMessageId)
    setQaPairs(nextQaPairs)
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
      return triggerRequest(generateRequestPayloadFromQaPairs(nextQaPairs))
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

  const onClearMessages = () => {
    stop()
    setQaPairs([])
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
    if (!isLoading || !qaPairs?.length || !answer) return

    const lastQaPairs = qaPairs[qaPairs.length - 1]
    const lastMessage = answer
    setQaPairs(prev => {
      const assisatntMessage = prev[prev.length - 1].assistant
      const nextAssistantMessage: AssistantMessage = {
        ...assisatntMessage,
        id: assisatntMessage?.id || nanoid(),
        message: lastMessage.answer_delta ?? '',
        error: undefined,
        relevant_code: answer?.relevant_code
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
  }, [answer, isLoading])

  const scrollToBottom = useDebounceCallback(() => {
    if (container) {
      container.scrollTo({
        top: container.scrollHeight,
        behavior: 'smooth'
      })
    } else {
      window.scrollTo({
        top: document.body.offsetHeight,
        behavior: 'smooth'
      })
    }
  }, 100)

  React.useLayoutEffect(() => {
    // scroll to bottom when a request is sent
    if (isLoading) {
      scrollToBottom.run()
    }
  }, [isLoading])

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

  const generateRequestPayloadFromQaPairs = (
    qaPairs: QuestionAnswerPair[]
  ): AnswerRequest => {
    const userMessage = qaPairs[qaPairs.length - 1].user
    // FIXME(wwayne): The first context in relevantContext is currently sent as the code query.
    //                Review and update the logic to ensure the appropriate api attribute is used.
    const contextForCodeQuery =
      userMessage?.selectContext || userMessage?.relevantContext?.[0]
    const code_query: AnswerRequest['code_query'] | undefined =
      contextForCodeQuery
        ? {
            content: contextForCodeQuery.content ?? '',
            filepath: contextForCodeQuery.filepath,
            language: contextForCodeQuery.filepath
              ? filename2prism(contextForCodeQuery.filepath)[0] || 'text'
              : 'text',
            git_url: contextForCodeQuery?.git_url ?? ''
          }
        : undefined

    return {
      messages: toMessages(qaPairs).slice(0, -1),
      code_query,
      doc_query: !!docQuery,
      generate_relevant_questions: !!generateRelevantQuestions
    }
  }

  const handleSendUserChat = useLatest(
    async (userMessage: UserMessageWithOptionalId) => {
      if (isLoading) return

      // If no id is provided, set a fallback id.
      const newUserMessage = {
        ...userMessage,
        id: userMessage.id ?? nanoid()
      }

      const nextQaPairs = [
        ...qaPairs,
        {
          user: newUserMessage,
          // For placeholder, and it also conveniently handles streaming responses and displays reference context.
          assistant: {
            id: nanoid(),
            message: '',
            error: undefined
          }
        }
      ]

      setQaPairs(nextQaPairs)

      return triggerRequest(generateRequestPayloadFromQaPairs(nextQaPairs))
    }
  )

  const sendUserChat = async (userMessage: UserMessageWithOptionalId) => {
    return handleSendUserChat.current?.(userMessage)
  }

  const handleSubmit = async (value: string) => {
    if (onSubmitMessage) {
      onSubmitMessage(value, clientSelectedContext)
    } else {
      sendUserChat({
        message: value,
        relevantContext: clientSelectedContext
      })
    }
    setClientSelectedContext([])
  }

  const handleAddClientSelectedContext = useLatest((context: Context) => {
    setClientSelectedContext(clientSelectedContext.concat([context]))
  })

  const addClientSelectedContext = (context: Context) => {
    handleAddClientSelectedContext.current?.(context)
  }

  const removeClientSelectedContext = (index: number) => {
    const newSelectedContext = [...clientSelectedContext]
    newSelectedContext.splice(index, 1)
    setClientSelectedContext(newSelectedContext)
  }

  React.useEffect(() => {
    if (!isOnLoadExecuted.current) return
    onThreadUpdates(qaPairs)
  }, [qaPairs])

  React.useImperativeHandle(
    ref,
    () => {
      return {
        sendUserChat,
        stop,
        isLoading,
        addClientSelectedContext
      }
    },
    []
  )

  React.useEffect(() => {
    if (isOnLoadExecuted.current) return

    isOnLoadExecuted.current = true
    onLoaded?.()
    setInitialzed(true)
  }, [])

  const chatMaxWidthClass = maxWidth ? `max-w-${maxWidth}` : 'max-w-2xl'
  if (!initialized)
    return (
      <ListSkeleton className={`${chatMaxWidthClass} mx-auto pt-4 md:pt-10`} />
    )

  return (
    <ChatContext.Provider
      value={{
        isLoading,
        qaPairs,
        onNavigateToContext,
        handleMessageAction,
        onClearMessages,
        container,
        onCopyContent,
        client,
        onApplyInEditor,
        clientSelectedContext,
        removeClientSelectedContext
      }}
    >
      <div className="flex justify-center overflow-x-hidden">
        <div className={`w-full px-4 ${chatMaxWidthClass}`}>
          {/* FIXME: pb-[200px] might not enough when adding a large number of clientSelectedContext */}
          <div className={cn('pb-[200px] pt-4 md:pt-10', className)}>
            {qaPairs?.length ? (
              <QuestionAnswerList
                messages={qaPairs}
                chatMaxWidthClass={chatMaxWidthClass}
              />
            ) : (
              <EmptyScreen
                setInput={setInput}
                chatMaxWidthClass={chatMaxWidthClass}
                welcomeMessage={welcomeMessage}
              />
            )}
            <ChatScrollAnchor trackVisibility={isLoading} />
          </div>
          <ChatPanel
            onSubmit={handleSubmit}
            className={cn('fixed inset-x-0 bottom-0', promptFormClassname)}
            id={chatId}
            stop={onStop}
            reload={onReload}
            input={input}
            setInput={setInput}
            chatMaxWidthClass={chatMaxWidthClass}
          />
        </div>
      </div>
    </ChatContext.Provider>
  )
}

export const Chat = React.forwardRef<ChatRef, ChatProps>(ChatRenderer)
