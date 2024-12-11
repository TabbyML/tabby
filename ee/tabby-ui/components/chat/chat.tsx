import React, { RefObject } from 'react'
import { compact, findIndex, isEqual, some, uniqWith } from 'lodash-es'
import type {
  Context,
  FileContext,
  NavigateOpts,
  SymbolInfo
} from 'tabby-chat-panel'

import {
  ERROR_CODE_NOT_FOUND,
  VSCODE_NOTEBOOK_CELL_SCHEME
} from '@/lib/constants'
import {
  CodeQueryInput,
  CreateMessageInput,
  InputMaybe,
  MessageAttachmentCodeInput,
  ThreadRunOptionsInput
} from '@/lib/gql/generates/graphql'
import { useDebounceCallback } from '@/lib/hooks/use-debounce'
import { useLatest } from '@/lib/hooks/use-latest'
import { useThreadRun } from '@/lib/hooks/use-thread-run'
import { filename2prism } from '@/lib/language-utils'
import { useChatStore } from '@/lib/stores/chat-store'
import { ExtendedCombinedError } from '@/lib/types'
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
  threadId: string | undefined
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
  onApplyInEditor?:
    | ((content: string) => void)
    | ((content: string, opts?: { languageId: string; smart: boolean }) => void)
  onLookupSymbol?: (
    filepaths: string[],
    keyword: string
  ) => Promise<SymbolInfo | undefined>
  relevantContext: Context[]
  activeSelection: Context | null
  removeRelevantContext: (index: number) => void
  chatInputRef: RefObject<HTMLTextAreaElement>
  supportsOnApplyInEditorV2: boolean
}

export const ChatContext = React.createContext<ChatContextValue>(
  {} as ChatContextValue
)

export interface ChatRef {
  sendUserChat: (message: UserMessageWithOptionalId) => void
  stop: () => void
  isLoading: boolean
  addRelevantContext: (context: Context) => void
  focus: () => void
  updateActiveSelection: (context: Context | null) => void
}

interface ChatProps extends React.ComponentProps<'div'> {
  chatId: string
  api?: string
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
  onSubmitMessage?: (msg: string, relevantContext?: Context[]) => Promise<void>
  onApplyInEditor?:
    | ((content: string) => void)
    | ((content: string, opts?: { languageId: string; smart: boolean }) => void)
  onLookupSymbol?: (
    filepaths: string[],
    keyword: string
  ) => Promise<SymbolInfo | undefined>
  chatInputRef: RefObject<HTMLTextAreaElement>
  supportsOnApplyInEditorV2: boolean
}

function ChatRenderer(
  {
    className,
    chatId,
    initialMessages,
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
    onSubmitMessage,
    onApplyInEditor,
    onLookupSymbol,
    chatInputRef,
    supportsOnApplyInEditorV2
  }: ChatProps,
  ref: React.ForwardedRef<ChatRef>
) {
  const [initialized, setInitialzed] = React.useState(false)
  const [threadId, setThreadId] = React.useState<string | undefined>()
  const isOnLoadExecuted = React.useRef(false)
  const [qaPairs, setQaPairs] = React.useState(initialMessages ?? [])
  const [input, setInput] = React.useState<string>('')
  const [relevantContext, setRelevantContext] = React.useState<Context[]>([])
  const [activeSelection, setActiveSelection] = React.useState<Context | null>(
    null
  )
  const enableActiveSelection = useChatStore(
    state => state.enableActiveSelection
  )
  const chatPanelRef = React.useRef<ChatPanelRef>(null)

  const {
    sendUserMessage,
    isLoading,
    error,
    answer,
    stop,
    regenerate,
    deleteThreadMessagePair
  } = useThreadRun({
    threadId
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
        qaPair.user,
        enableActiveSelection
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

  const onEditMessage = async (userMessageId: string) => {
    if (!threadId) return

    const qaPair = qaPairs.find(o => o.user.id === userMessageId)
    if (!qaPair?.user || !qaPair.assistant) return

    const userMessage = qaPair.user
    let nextClientContext: Context[] = []

    // restore client context
    if (userMessage.relevantContext?.length) {
      nextClientContext = nextClientContext.concat(userMessage.relevantContext)
    }

    setRelevantContext(uniqWith(nextClientContext, isEqual))

    // delete message pair
    const nextQaPairs = qaPairs.filter(o => o.user.id !== userMessageId)
    setQaPairs(nextQaPairs)
    setInput(userMessage.message)
    if (userMessage.activeContext) {
      onNavigateToContext(userMessage.activeContext, {
        openInEditor: true
      })
    }

    deleteThreadMessagePair(threadId, qaPair?.user.id, qaPair?.assistant?.id)
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
    actionType: MessageActionType
  ) => {
    switch (actionType) {
      case 'delete':
        onDeleteMessage(userMessageId)
        break
      case 'regenerate':
        onRegenerateResponse(userMessageId)
        break
      case 'edit':
        onEditMessage(userMessageId)
        break
      default:
        break
    }
  }

  React.useEffect(() => {
    if (!isLoading || !qaPairs?.length || !answer) return

    const lastQaPairs = qaPairs[qaPairs.length - 1]

    // update threadId
    if (answer.threadId && !threadId) {
      setThreadId(answer.threadId)
    }

    setQaPairs(prev => {
      const assisatntMessage = prev[prev.length - 1].assistant
      const nextAssistantMessage: AssistantMessage = {
        ...assisatntMessage,
        id: answer.assistantMessageId || assisatntMessage?.id || nanoid(),
        message: answer.content,
        error: undefined,
        relevant_code: answer.attachmentsCode?.map(o => o.code) ?? []
      }
      // merge assistantMessage
      return [
        ...prev.slice(0, prev.length - 1),
        {
          user: {
            ...lastQaPairs.user,
            id: answer?.userMessageId || lastQaPairs.user.id
          },
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
              error: formatThreadRunErrorMessage(error)
            }
          }
        ]
      })
    }

    if (error?.message === 'Thread not found' && !qaPairs?.length) {
      onClearMessages()
    }
  }, [error])

  const generateRequestPayload = (
    userMessage: UserMessage,
    enableActiveSelection?: boolean
  ): [CreateMessageInput, ThreadRunOptionsInput] => {
    // use selectContext for code query by default
    let contextForCodeQuery: FileContext | undefined = userMessage.selectContext

    // if enableActiveSelection, use selectContext or activeContext for code query
    if (enableActiveSelection) {
      contextForCodeQuery = contextForCodeQuery || userMessage.activeContext
    }

    // check context for codeQuery
    if (!isValidContextForCodeQuery(contextForCodeQuery)) {
      contextForCodeQuery = undefined
    }

    const codeQuery: InputMaybe<CodeQueryInput> = contextForCodeQuery
      ? {
          content: contextForCodeQuery.content ?? '',
          filepath: contextForCodeQuery.filepath,
          language: contextForCodeQuery.filepath
            ? filename2prism(contextForCodeQuery.filepath)[0] || 'plaintext'
            : 'plaintext',
          gitUrl: contextForCodeQuery?.git_url ?? ''
        }
      : null

    const hasUsableActiveContext =
      enableActiveSelection && !!userMessage.activeContext
    const clientSideFileContexts: FileContext[] = uniqWith(
      compact([
        hasUsableActiveContext && userMessage.activeContext,
        ...(userMessage?.relevantContext || [])
      ]),
      isEqual
    )

    const attachmentCode: MessageAttachmentCodeInput[] =
      clientSideFileContexts.map(o => ({
        content: o.content,
        filepath: o.filepath,
        startLine: o.range.start
      }))

    const content = userMessage.message

    return [
      {
        content,
        attachments: {
          code: attachmentCode
        }
      },
      {
        docQuery: docQuery ? { content, searchPublic: false } : null,
        generateRelevantQuestions: !!generateRelevantQuestions,
        codeQuery
      }
    ]
  }

  const handleSendUserChat = useLatest(
    async (userMessage: UserMessageWithOptionalId) => {
      if (isLoading) return

      let selectCodeSnippet = ''
      const selectCodeContextContent = userMessage?.selectContext?.content
      if (selectCodeContextContent) {
        const language = userMessage?.selectContext?.filepath
          ? filename2prism(userMessage?.selectContext?.filepath)[0] ?? ''
          : ''
        selectCodeSnippet = `\n${'```'}${language}\n${
          selectCodeContextContent ?? ''
        }\n${'```'}\n`
      }

      const finalActiveContext = activeSelection || userMessage.activeContext
      const newUserMessage: UserMessage = {
        ...userMessage,
        message: userMessage.message + selectCodeSnippet,
        // If no id is provided, set a fallback id.
        id: userMessage.id ?? nanoid(),
        selectContext: userMessage.selectContext,
        // For forward compatibility
        activeContext:
          enableActiveSelection && finalActiveContext
            ? finalActiveContext
            : undefined
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

      sendUserMessage(
        ...generateRequestPayload(newUserMessage, enableActiveSelection)
      )
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
    setRelevantContext(oldValue => appendContextAndDedupe(oldValue, context))
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

  const debouncedUpdateActiveSelection = useDebounceCallback(
    (ctx: Context | null) => {
      setActiveSelection(ctx)
    },
    300
  )

  const updateActiveSelection = (ctx: Context | null) => {
    debouncedUpdateActiveSelection.run(ctx)
  }

  React.useImperativeHandle(
    ref,
    () => {
      return {
        sendUserChat,
        stop,
        isLoading,
        addRelevantContext,
        focus: () => chatPanelRef.current?.focus(),
        updateActiveSelection
      }
    },
    []
  )

  React.useEffect(() => {
    setInitialzed(true)
    onLoaded?.()
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
        threadId,
        isLoading,
        qaPairs,
        onNavigateToContext,
        handleMessageAction,
        onClearMessages,
        container,
        onCopyContent,
        onApplyInEditor,
        onLookupSymbol,
        relevantContext,
        removeRelevantContext,
        chatInputRef,
        activeSelection,
        supportsOnApplyInEditorV2
      }}
    >
      <div className="flex justify-center overflow-x-hidden">
        <div
          className={`w-full px-4 md:pl-10 md:pr-[3.75rem] ${chatMaxWidthClass}`}
        >
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
            chatInputRef={chatInputRef}
          />
        </div>
      </div>
    </ChatContext.Provider>
  )
}

function appendContextAndDedupe(
  ctxList: Context[],
  newCtx: Context
): Context[] {
  if (!ctxList.some(ctx => isEqual(ctx, newCtx))) {
    return ctxList.concat([newCtx])
  }
  return ctxList
}

export const Chat = React.forwardRef<ChatRef, ChatProps>(ChatRenderer)

function formatThreadRunErrorMessage(error: ExtendedCombinedError | undefined) {
  if (!error) return 'Failed to fetch'

  if (error.message === '401') {
    return 'Unauthorized'
  }

  if (
    some(error.graphQLErrors, o => o.extensions?.code === ERROR_CODE_NOT_FOUND)
  ) {
    return `The thread has expired, please click ${"'"}Clear${"'"} and try again.`
  }

  return error.message || 'Failed to fetch'
}

function isValidContextForCodeQuery(context: FileContext | undefined) {
  if (!context) return false

  const isUntitledFile =
    context.filepath.startsWith('untitled:') &&
    !filename2prism(context.filepath)[0]

  const isVscodeNotebookCell = context.filepath.startsWith(
    VSCODE_NOTEBOOK_CELL_SCHEME
  )

  return !isUntitledFile && !isVscodeNotebookCell
}
