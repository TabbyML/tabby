import React from 'react'
import { compact, findIndex } from 'lodash-es'
import type { Context, NavigateOpts } from 'tabby-chat-panel'

import {
  CodeQueryInput,
  CreateMessageInput,
  InputMaybe,
  MessageAttachmentCodeInput,
  ThreadRunItem,
  ThreadRunOptionsInput
} from '@/lib/gql/generates/graphql'
import { useDebounceCallback } from '@/lib/hooks/use-debounce'
import { useLatest } from '@/lib/hooks/use-latest'
import { useThreadRun } from '@/lib/hooks/use-thread-run'
import { filename2prism } from '@/lib/language-utils'
import {
  AssistantMessage,
  MessageActionType,
  QuestionAnswerPair,
  UserMessage,
  UserMessageWithOptionalId
} from '@/lib/types/chat'
import { cn, nanoid } from '@/lib/utils'

import { ListSkeleton } from '../skeleton'
import { ChatPanel, ChatPanelRef } from './chat-panel'
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
  relevantContext: Context[]
  removeRelevantContext: (index: number) => void
}

export const ChatContext = React.createContext<ChatContextValue>(
  {} as ChatContextValue
)

// FIXME remove
function selectContextToMessageContent(
  context: UserMessage['selectContext']
): string {
  if (!context || !context.content) return ''
  const { content, filepath } = context
  const language = filename2prism(filepath)?.[0]
  return `\n${'```'}${language ?? ''}\n${content ?? ''}\n${'```'}\n`
}

export interface ChatRef {
  sendUserChat: (message: UserMessageWithOptionalId) => void
  stop: () => void
  isLoading: boolean
  addRelevantContext: (context: Context) => void
  focus: () => void
}

interface ChatProps extends React.ComponentProps<'div'> {
  chatId: string
  api?: string
  headers?: Record<string, string> | Headers
  isEphemeral?: boolean
  initialMessages?: QuestionAnswerPair[]
  onLoaded?: () => void
  onThreadUpdates?: (messages: QuestionAnswerPair[]) => void
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
    isEphemeral,
    onLoaded,
    onThreadUpdates,
    onNavigateToContext,
    container,
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
  const [threadId, setThreadId] = React.useState<string | undefined>()
  const isOnLoadExecuted = React.useRef(false)
  const [qaPairs, setQaPairs] = React.useState(initialMessages ?? [])
  const [input, setInput] = React.useState<string>('')
  const [relevantContext, setRelevantContext] = React.useState<Context[]>([])
  const chatPanelRef = React.useRef<ChatPanelRef>(null)

  const updateCurrentQaPairIDs = (
    newUserMessageId: string,
    newAssistantMessageId: string
  ) => {
    const qaPairIndex = qaPairs.length - 1
    const qaPair = qaPairs[qaPairIndex]
    const newQaPairs: QuestionAnswerPair[] = [
      ...qaPairs.slice(0, -1),
      {
        user: {
          ...qaPair.user,
          id: newUserMessageId
        },
        assistant: {
          ...qaPair.assistant,
          message: qaPair?.assistant?.message ?? '',
          id: newAssistantMessageId
        }
      }
    ]

    setQaPairs(newQaPairs)
  }

  const onAssistantMessageCompleted = (
    newThreadId: string,
    threadRunItem: ThreadRunItem | undefined
  ) => {
    if (
      threadRunItem?.threadUserMessageCreated &&
      threadRunItem.threadAssistantMessageCreated
    ) {
      updateCurrentQaPairIDs(
        threadRunItem.threadUserMessageCreated,
        threadRunItem.threadAssistantMessageCreated
      )
    }
  }

  const {
    sendUserMessage,
    isLoading,
    error,
    answer,
    stop,
    regenerate,
    deleteThreadMessagePair
  } = useThreadRun({
    threadId,
    headers,
    isEphemeral,
    onAssistantMessageCompleted
  })

  const onDeleteMessage = async (userMessageId: string) => {
    if (!threadId) return

    // Stop generating first.
    stop()
    const qaPair = qaPairs.find(o => o.user.id === userMessageId)
    if (!qaPair?.user || !qaPair.assistant) return

    const nextQaPairs = qaPairs.filter(o => o.user.id !== userMessageId)
    setQaPairs(nextQaPairs)

    deleteThreadMessagePair(threadId, qaPair?.user.id, qaPair?.assistant?.id)
  }

  const onRegenerateResponse = async (userMessageId: string) => {
    if (!threadId) return

    const qaPairIndex = findIndex(qaPairs, o => o.user.id === userMessageId)
    if (qaPairIndex > -1) {
      const qaPair = qaPairs[qaPairIndex]

      if (!qaPair.assistant) return

      const newUserMessageId = nanoid()
      const newAssistantMessgaeid = nanoid()
      let nextQaPairs: QuestionAnswerPair[] = [
        ...qaPairs.slice(0, qaPairIndex),
        {
          user: {
            ...qaPair.user,
            id: newUserMessageId
          },
          assistant: {
            id: newAssistantMessgaeid,
            message: '',
            error: undefined
          }
        }
      ]
      setQaPairs(nextQaPairs)
      const [userMessage, threadRunOptions] = generateRequestPayload(
        qaPair.user
      )
      return regenerate({
        threadId,
        userMessageId: qaPair.user.id,
        assistantMessageId: qaPair.assistant.id,
        userMessage,
        threadRunOptions
      })
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
    stop(true)
    setQaPairs([])
    setThreadId(undefined)
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

    // update threadId
    if (answer?.threadCreated && !threadId) {
      setThreadId(answer.threadCreated)
    }

    setQaPairs(prev => {
      const assisatntMessage = prev[prev.length - 1].assistant
      const nextAssistantMessage: AssistantMessage = {
        ...assisatntMessage,
        id: assisatntMessage?.id || nanoid(),
        message: answer.threadAssistantMessageContentDelta ?? '',
        error: undefined,
        relevant_code:
          answer?.threadAssistantMessageAttachmentsCode?.map(o => o.code) ?? []
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

  const generateRequestPayload = (
    userMessage: UserMessage
  ): [CreateMessageInput, ThreadRunOptionsInput] => {
    const contextForCodeQuery =
      userMessage?.selectContext || userMessage?.activeContext
    const codeQuery: InputMaybe<CodeQueryInput> = contextForCodeQuery
      ? {
          content: contextForCodeQuery.content ?? '',
          filepath: contextForCodeQuery.filepath,
          language: contextForCodeQuery.filepath
            ? filename2prism(contextForCodeQuery.filepath)[0] || 'text'
            : 'text',
          gitUrl: contextForCodeQuery?.git_url ?? ''
        }
      : null

    const attachmentCode: MessageAttachmentCodeInput[] = compact([
      contextForCodeQuery
        ? {
            content: contextForCodeQuery.content,
            filepath: contextForCodeQuery.filepath
          }
        : undefined,
      ...(userMessage?.relevantContext?.map(o => ({
        filepath: o.filepath,
        content: o.content
      })) ?? [])
    ])

    const content = userMessage.message

    return [
      {
        content,
        attachments: {
          code: attachmentCode
        }
      },
      {
        docQuery: docQuery ? { content } : null,
        generateRelevantQuestions: !!generateRelevantQuestions,
        codeQuery
      }
    ]
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

      return sendUserMessage(...generateRequestPayload(newUserMessage))
    }
  )

  const sendUserChat = (userMessage: UserMessageWithOptionalId) => {
    return handleSendUserChat.current?.(userMessage)
  }

  const handleSubmit = async (value: string) => {
    if (onSubmitMessage) {
      onSubmitMessage(value, relevantContext)
    } else {
      sendUserChat({
        message: value,
        relevantContext: relevantContext
      })
    }
    setRelevantContext([])
  }

  const handleAddRelevantContext = useLatest((context: Context) => {
    setRelevantContext(relevantContext.concat([context]))
  })

  const addRelevantContext = (context: Context) => {
    handleAddRelevantContext.current?.(context)
  }

  const removeRelevantContext = (index: number) => {
    const newRelevantContext = [...relevantContext]
    newRelevantContext.splice(index, 1)
    setRelevantContext(newRelevantContext)
  }

  React.useEffect(() => {
    if (!isOnLoadExecuted.current) return
    onThreadUpdates?.(qaPairs)
  }, [qaPairs])

  React.useImperativeHandle(
    ref,
    () => {
      return {
        sendUserChat,
        stop,
        isLoading,
        addRelevantContext,
        focus: () => chatPanelRef.current?.focus()
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
  if (!initialized) {
    return (
      <ListSkeleton className={`${chatMaxWidthClass} mx-auto pt-4 md:pt-10`} />
    )
  }

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
        relevantContext,
        removeRelevantContext
      }}
    >
      <div className="flex justify-center overflow-x-hidden">
        <div className={`w-full px-4 ${chatMaxWidthClass}`}>
          {/* FIXME: pb-[200px] might not enough when adding a large number of relevantContext */}
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
            ref={chatPanelRef}
          />
        </div>
      </div>
    </ChatContext.Provider>
  )
}

export const Chat = React.forwardRef<ChatRef, ChatProps>(ChatRenderer)
