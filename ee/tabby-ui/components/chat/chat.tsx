import React from 'react'
import { Content, EditorEvents } from '@tiptap/core'
import {
  compact,
  findIndex,
  isEqual,
  isEqualWith,
  some,
  uniqWith
} from 'lodash-es'
import type { ChatCommand, EditorContext } from 'tabby-chat-panel'
import { useQuery } from 'urql'

import { ERROR_CODE_NOT_FOUND } from '@/lib/constants'
import {
  CodeQueryInput,
  CreateMessageInput,
  DocQueryInput,
  InputMaybe,
  ListThreadMessagesQuery,
  MessageAttachmentCodeInput,
  Role,
  ThreadRunOptionsInput
} from '@/lib/gql/generates/graphql'
import { useDebounceCallback } from '@/lib/hooks/use-debounce'
import { useLatest } from '@/lib/hooks/use-latest'
import { useSelectedModel } from '@/lib/hooks/use-models'
import { useThreadRun } from '@/lib/hooks/use-thread-run'
import { filename2prism } from '@/lib/language-utils'
import { useChatStore } from '@/lib/stores/chat-store'
import {
  contextInfoQuery,
  listThreadMessages,
  repositorySourceListQuery
} from '@/lib/tabby/query'
import { ExtendedCombinedError } from '@/lib/types'
import {
  AssistantMessage,
  Context,
  FileContext,
  MessageActionType,
  QuestionAnswerPair,
  UserMessage,
  UserMessageWithOptionalId
} from '@/lib/types/chat'
import {
  buildMarkdownCodeBlock,
  cn,
  convertEditorContext,
  findClosestGitRepository,
  getFileLocationFromContext,
  getPromptForChatCommand,
  nanoid,
  processingPlaceholder,
  terminalContextToAttachmentCode
} from '@/lib/utils'
import { convertContextBlockToPlaceholder } from '@/lib/utils/markdown'

import LoadingWrapper from '../loading-wrapper'
import { ChatContext } from './chat-context'
import { ChatPanel, ChatPanelRef } from './chat-panel'
import { ChatScrollAnchor } from './chat-scroll-anchor'
import { EmptyScreen } from './empty-screen'
import { convertTextToTiptapContent } from './form-editor/utils'

import './git/utils'

import { Maybe } from 'graphql/jsutils/Maybe'

import { QuestionAnswerList } from './question-answer'
import { QaPairSkeleton } from './skeletion'
import { ChatProps, ChatRef } from './types'

export const Chat = React.forwardRef<ChatRef, ChatProps>(
  (
    {
      className,
      threadId: propsThreadId,
      setThreadId: onThreadIdChange,
      initialMessages,
      onLoaded,
      onThreadUpdates,
      container,
      docQuery,
      generateRelevantQuestions,
      welcomeMessage,
      promptFormClassname,
      onCopyContent,
      onApplyInEditor,
      onLookupSymbol,
      openInEditor,
      openExternal,
      chatInputRef,
      supportsOnApplyInEditorV2,
      readWorkspaceGitRepositories,
      getActiveEditorSelection,
      getActiveTerminalSelection,
      fetchSessionState,
      storeSessionState,
      listFileInWorkspace,
      readFileContent,
      listSymbols,
      setShowHistory,
      runShell,
      getChanges,
      ...props
    },
    ref
  ) => {
    const [threadId, setThreadId] = React.useState<string | undefined>()
    const [isDataSetup, setIsDataSetup] = React.useState(false)
    const [initialized, setInitialized] = React.useState(false)
    const isOnLoadExecuted = React.useRef(false)
    const [qaPairs, setQaPairs] = React.useState(initialMessages ?? [])
    const [relevantContext, setRelevantContext] = React.useState<Context[]>([])
    const [activeSelection, setActiveSelection] =
      React.useState<Context | null>(null)

    // sourceId
    const [selectedRepoId, setSelectedRepoId] = React.useState<
      string | undefined
    >()

    React.useEffect(() => {
      if (propsThreadId) {
        setThreadId(propsThreadId)
        setQaPairs([])
      }
    }, [propsThreadId])

    // fetch messages
    const [
      {
        data: threadMessages,
        error: threadMessagesError,
        fetching: fetchingMessages,
        stale: threadMessagesStale
      }
    ] = useQuery({
      query: listThreadMessages,
      variables: {
        threadId: propsThreadId as string
      },
      pause: !propsThreadId
    })

    React.useEffect(() => {
      const formatQaPairs = (
        messages: ListThreadMessagesQuery['threadMessages']['edges']
      ) => {
        const pairs: QuestionAnswerPair[] = []
        let currentPair: Partial<QuestionAnswerPair> = {}
        messages.forEach(x => {
          const message = x.node
          if (message.role === Role.User) {
            currentPair.user = {
              id: message.id,
              content: message.content,
              relevantContext: message.attachment.clientCode?.map(x => {
                return {
                  kind: 'file',
                  content: x.content,
                  filepath: x.filepath ?? '',
                  startLine: x.startLine,
                  gitUrl: ''
                }
              })
            }
          } else if (x.node.role === Role.Assistant) {
            if (!currentPair.assistant) {
              currentPair.assistant = x.node
              if (!!currentPair.user && !!currentPair.assistant) {
                pairs.push(currentPair as QuestionAnswerPair)
                currentPair = {}
              }
            }
          }
        })
        return pairs
      }

      if (threadMessagesStale) return

      if (threadMessages?.threadMessages?.edges?.length && propsThreadId) {
        const nextQair = formatQaPairs(threadMessages.threadMessages.edges)
        setQaPairs(nextQair)
      } else {
        setQaPairs([])
      }
    }, [threadMessages])

    React.useEffect(() => {
      if (isDataSetup) {
        storeSessionState?.({ selectedRepoId })
      }
    }, [selectedRepoId, isDataSetup, storeSessionState])

    const enableActiveSelection = useChatStore(
      state => state.enableActiveSelection
    )

    const chatPanelRef = React.useRef<ChatPanelRef>(null)

    // both set/get input from prompt form
    const setInput = (str: Content) => {
      chatPanelRef.current?.setInput(str)
    }
    const input = chatPanelRef.current?.input ?? ''

    const onUpdate = (p: EditorEvents['update']) => {
      if (isDataSetup) {
        storeSessionState?.({ input: p.editor.getJSON() })
      }
    }

    const [{ data: contextInfoData, fetching: fetchingContextInfo }] = useQuery(
      {
        query: contextInfoQuery
      }
    )

    // fetch indexed repos
    const [{ data: repositoryListData, fetching: fetchingRepos }] = useQuery({
      query: repositorySourceListQuery
    })
    const repos = repositoryListData?.repositoryList

    // fetch models
    const { selectedModel } = useSelectedModel()

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

      const qaPair = qaPairs.find(o => o.user.id === userMessageId)
      if (!qaPair?.user || !qaPair.assistant) return

      const nextQaPairs = qaPairs.filter(o => o.user.id !== userMessageId)
      setQaPairs(nextQaPairs)
      storeSessionState?.({
        qaPairs: nextQaPairs
      })

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
              codeSourceId: qaPair.assistant.codeSourceId,
              id: newAssistantMessgaeid,
              content: '',
              error: undefined
            }
          }
        ]
        setQaPairs(nextQaPairs)

        const [createMessageInput, threadRunOptions] =
          await generateRequestPayload(
            qaPair.user,
            qaPair.assistant.codeSourceId
          )

        return regenerate({
          threadId,
          userMessageId: qaPair.user.id,
          assistantMessageId: qaPair.assistant.id,
          userMessage: createMessageInput,
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
        nextClientContext = nextClientContext.concat(
          userMessage.relevantContext
        )
      }

      const updatedRelevantContext = uniqWith(nextClientContext, isEqual)
      setRelevantContext(updatedRelevantContext)

      // delete message pair
      const nextQaPairs = qaPairs.filter(o => o.user.id !== userMessageId)
      setQaPairs(nextQaPairs)

      storeSessionState?.({
        qaPairs: nextQaPairs
      })

      const inputContent = convertTextToTiptapContent(
        convertContextBlockToPlaceholder(userMessage.content),
        contextInfoData?.contextInfo?.sources ?? []
      )
      setInput({
        type: 'doc',
        content: inputContent
      })

      if (
        userMessage.activeContext &&
        userMessage.activeContext.kind === 'file'
      ) {
        openInEditor(getFileLocationFromContext(userMessage.activeContext))
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

    const switchToNewChat = () => {
      stop(true)
      setQaPairs([])
      setThreadId(undefined)
      onThreadIdChange(undefined)
      storeSessionState?.({
        qaPairs: [],
        threadId: undefined
      })
    }

    const onClearMessages = () => {
      switchToNewChat()
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
      if (!qaPairs?.length || !answer) return

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
          content: answer.content,
          error: undefined
        }

        // update assiatant message status
        if ('isReadingCode' in answer) {
          nextAssistantMessage.isReadingCode = answer.isReadingCode
        }
        if ('isReadingFileList' in answer) {
          nextAssistantMessage.isReadingFileList = answer.isReadingFileList
        }
        if ('isReadingDocs' in answer) {
          nextAssistantMessage.isReadingDocs = answer.isReadingDocs
        }
        // update expose steps
        nextAssistantMessage.readingCode = answer?.readingCode
        nextAssistantMessage.readingDoc = answer?.readingDoc
        // get and format scores from streaming answer
        if (
          !nextAssistantMessage.attachment?.code &&
          !!answer.attachmentsCode
        ) {
          nextAssistantMessage.attachment = {
            clientCode: null,
            doc: nextAssistantMessage.attachment?.doc || null,
            code:
              answer.attachmentsCode.map(hit => ({
                ...hit.code,
                extra: {
                  scores: hit.scores
                }
              })) || null
          }
        }

        // get and format scores from streaming answer
        if (!nextAssistantMessage.attachment?.doc && !!answer.attachmentsDoc) {
          nextAssistantMessage.attachment = {
            clientCode: null,
            doc:
              answer.attachmentsDoc.map(hit => ({
                ...hit.doc,
                extra: {
                  score: hit.score
                }
              })) || null,
            code: nextAssistantMessage.attachment?.code || null
          }
        }

        if (
          !nextAssistantMessage.attachment?.codeFileList &&
          answer?.attachmentsFileList?.codeFileList?.length
        ) {
          nextAssistantMessage.attachment = {
            clientCode: null,
            doc: nextAssistantMessage.attachment?.doc || null,
            codeFileList: {
              fileList: answer.attachmentsFileList.codeFileList,
              truncated: answer.attachmentsFileList.truncated
            },
            code: nextAssistantMessage.attachment?.code || null
          }
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

      if (!isLoading) {
        storeSessionState?.({ qaPairs })
      }
    }, [answer, isLoading])

    const scrollToBottom = useDebounceCallback(
      (behavior: ScrollBehavior = 'smooth') => {
        if (container) {
          container.scrollTo({
            top: container.scrollHeight,
            behavior
          })
        } else {
          window.scrollTo({
            top: document.body.offsetHeight,
            behavior
          })
        }
      },
      100
    )

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
          const nextQaPairs = [
            ...prev.slice(0, prev.length - 1),
            {
              ...lastQaPairs,
              assistant: {
                ...lastQaPairs.assistant,
                id: lastQaPairs.assistant?.id || nanoid(),
                content: lastQaPairs.assistant?.content ?? '',
                error: formatThreadRunErrorMessage(error)
              }
            }
          ]
          storeSessionState?.({
            qaPairs: nextQaPairs
          })
          return nextQaPairs
        })
      }

      if (error?.message === 'Thread not found' && !qaPairs?.length) {
        onClearMessages()
      }
    }, [error])

    const generateRequestPayload = async (
      userMessage: UserMessage,
      codeSourceId: Maybe<string> | undefined
    ): Promise<[CreateMessageInput, ThreadRunOptionsInput]> => {
      const hasUsableActiveContext =
        enableActiveSelection && !!userMessage.activeContext

      const clientContexts: Context[] = uniqWith(
        compact([
          userMessage.selectContext,
          hasUsableActiveContext && userMessage.activeContext,
          ...(userMessage?.relevantContext || [])
        ]),
        isEqual
      )

      const attachmentCode: MessageAttachmentCodeInput[] =
        clientContexts.map<MessageAttachmentCodeInput>(o => {
          if (o.kind === 'terminal') {
            return terminalContextToAttachmentCode(o)
          }
          return {
            content: o.content,
            filepath: o.filepath,
            startLine: o.range?.start
          }
        })

      const content = userMessage.content
      const docQuery: InputMaybe<DocQueryInput> = codeSourceId
        ? {
            content,
            sourceIds: [codeSourceId],
            searchPublic: false
          }
        : null
      const codeQuery: InputMaybe<CodeQueryInput> = codeSourceId
        ? {
            content,
            sourceId: codeSourceId,
            filepath: attachmentCode?.[0]?.filepath
          }
        : null

      return [
        {
          content,
          attachments: {
            code: attachmentCode
          }
        },
        {
          docQuery,
          codeQuery,
          generateRelevantQuestions: !!generateRelevantQuestions,
          modelName: selectedModel
        }
      ]
    }
    const handleSendUserChat = useLatest(
      async (
        userMessage: UserMessageWithOptionalId,
        codeSourceId?: Maybe<string>
      ) => {
        if (isLoading) return

        let selectCodeSnippet = ''
        if (userMessage?.selectContext?.kind === 'file') {
          const selectCodeContextContent = userMessage?.selectContext?.content
          if (selectCodeContextContent) {
            const language = userMessage?.selectContext?.filepath
              ? filename2prism(userMessage?.selectContext?.filepath)[0] ?? ''
              : ''
            selectCodeSnippet = buildMarkdownCodeBlock(
              selectCodeContextContent,
              language
            )
          }
        } else if (userMessage?.selectContext?.kind === 'terminal') {
          const selectCodeContextContent = userMessage?.selectContext?.selection
          if (selectCodeContextContent) {
            selectCodeSnippet = buildMarkdownCodeBlock(
              selectCodeContextContent,
              'shell'
            )
          }
        }

        // processing placeholder like contextCommand, file, symbol, etc.
        userMessage.content = await processingPlaceholder(userMessage.content, {
          getChanges,
          readFileContent
        })

        const newUserMessage: UserMessage = {
          ...userMessage,
          content: userMessage.content + selectCodeSnippet,
          // If no id is provided, set a fallback id.
          id: userMessage.id ?? nanoid(),
          selectContext: userMessage.selectContext,
          activeContext:
            enableActiveSelection && activeSelection
              ? activeSelection
              : undefined,
          relevantContext: [...(userMessage.relevantContext || [])]
        }

        const nextQaPairs = [
          ...qaPairs,
          {
            user: newUserMessage,
            // For placeholder, and it also conveniently handles streaming responses and displays reference context.
            assistant: {
              id: nanoid(),
              content: '',
              error: undefined,
              codeSourceId
            }
          }
        ]
        setQaPairs(nextQaPairs)

        storeSessionState?.({
          qaPairs: nextQaPairs
        })

        const payload = await generateRequestPayload(
          newUserMessage,
          codeSourceId
        )
        sendUserMessage(...payload)
      }
    )

    const sendUserChat = (
      userMessage: UserMessageWithOptionalId,
      codeSourceId: Maybe<string> | undefined
    ) => {
      return handleSendUserChat.current?.(userMessage, codeSourceId)
    }

    const handleExecuteCommand = useLatest(async (command: ChatCommand) => {
      const prompt = getPromptForChatCommand(command)
      sendUserChat(
        {
          content: prompt,
          selectContext:
            command == 'explain-terminal'
              ? (await getActiveTerminalSelection?.()) ?? undefined
              : activeSelection ?? undefined
        },
        selectedRepoId
      )
    })

    const executeCommand = async (command: ChatCommand) => {
      return handleExecuteCommand.current?.(command)
    }

    const handleSubmit = async (value: string) => {
      sendUserChat(
        {
          content: value,
          relevantContext
        },
        selectedRepoId
      )

      setRelevantContext([])
    }

    const handleAddRelevantContext = useLatest((context: Context) => {
      setRelevantContext(oldValue => {
        const updatedValue = appendContextAndDedupe(oldValue, context)
        return updatedValue
      })
    })

    const addRelevantContext = (editorContext: EditorContext) => {
      const context = convertEditorContext(editorContext)
      handleAddRelevantContext.current?.(context)
    }

    React.useEffect(() => {
      if (!isOnLoadExecuted.current) return
      onThreadUpdates?.(qaPairs)
    }, [qaPairs])

    React.useEffect(() => {
      if (isDataSetup) {
        storeSessionState?.({
          relevantContext
        })
      }
    }, [relevantContext, isDataSetup, storeSessionState])

    const debouncedUpdateActiveSelection = useDebounceCallback(
      (ctx: FileContext | null) => {
        setActiveSelection(ctx)
      },
      300
    )

    const updateActiveSelection = (editorContext: EditorContext | null) => {
      const context = editorContext ? convertEditorContext(editorContext) : null
      debouncedUpdateActiveSelection.run(
        context?.kind === 'file' ? context : null
      )
    }

    const fetchWorkspaceGitRepo = () => {
      if (readWorkspaceGitRepositories) {
        return readWorkspaceGitRepositories()
      } else {
        return []
      }
    }

    const initActiveEditorSelection = async () => {
      return getActiveEditorSelection?.()
    }

    React.useEffect(() => {
      const init = async () => {
        const [persistedState, activeEditorSelection] = await Promise.all([
          fetchSessionState?.(),
          initActiveEditorSelection()
        ])

        if (persistedState?.threadId) {
          setThreadId(persistedState.threadId)
        }
        if (persistedState?.qaPairs) {
          setQaPairs(persistedState.qaPairs)
        }
        if (persistedState?.input) {
          setInput(persistedState.input)
        }
        if (persistedState?.relevantContext) {
          setRelevantContext(persistedState.relevantContext)
        }
        scrollToBottom.run('instant')

        // get default repository
        if (
          persistedState?.selectedRepoId &&
          repos?.find(x => x.sourceId === persistedState.selectedRepoId)
        ) {
          setSelectedRepoId(persistedState.selectedRepoId)
        } else {
          const workspaceGitRepositories = await fetchWorkspaceGitRepo()
          if (workspaceGitRepositories?.length && repos?.length) {
            const defaultGitUrl = workspaceGitRepositories[0].url
            const repo = findClosestGitRepository(
              repos.map(x => ({ url: x.gitUrl, sourceId: x.sourceId })),
              defaultGitUrl
            )
            if (repo) {
              setSelectedRepoId(repo.sourceId)
            }
          }
        }

        // update active selection
        if (activeEditorSelection) {
          const context = convertEditorContext(activeEditorSelection)
          setActiveSelection(context.kind === 'file' ? context : null)
        }
      }

      if (!fetchingRepos && !isDataSetup) {
        init().finally(() => {
          setIsDataSetup(true)
        })
      }
    }, [fetchingRepos, isDataSetup])

    React.useEffect(() => {
      if (isDataSetup) {
        onLoaded?.()
        setInitialized(true)
      }
    }, [isDataSetup])

    React.useEffect(() => {
      if (threadMessagesError && !initialMessages) {
        setInitialized(true)
      }
    }, [threadMessagesError])

    React.useEffect(() => {
      storeSessionState?.({
        threadId
      })
    }, [threadId])

    React.useImperativeHandle(
      ref,
      () => {
        return {
          executeCommand,
          stop,
          isLoading,
          addRelevantContext,
          focus: () => chatPanelRef.current?.focus(),
          updateActiveSelection,
          newChat: switchToNewChat
        }
      },
      []
    )

    return (
      <ChatContext.Provider
        value={{
          threadId,
          isLoading,
          qaPairs,
          handleMessageAction,
          onClearMessages,
          container,
          onCopyContent,
          onApplyInEditor,
          onLookupSymbol,
          openInEditor,
          openExternal,
          relevantContext,
          setRelevantContext,
          chatInputRef,
          activeSelection,
          supportsOnApplyInEditorV2,
          selectedRepoId,
          setSelectedRepoId,
          repos,
          fetchingRepos,
          initialized,
          listFileInWorkspace,
          readFileContent,
          listSymbols,
          runShell,
          getChanges,
          contextInfo: contextInfoData?.contextInfo,
          fetchingContextInfo: fetchingContextInfo
        }}
      >
        <div className="flex justify-center overflow-x-hidden" {...props}>
          <div className={`mx-auto w-full max-w-5xl px-[16px]`}>
            {/* FIXME: pb-[200px] might not enough when adding a large number of relevantContext */}
            {initialized && (
              <div className={cn('pb-[200px] pt-4 md:pt-10', className)}>
                <LoadingWrapper
                  loading={fetchingMessages || threadMessagesStale}
                  triggerOnce={false}
                  fallback={<QaPairSkeleton />}
                  delay={propsThreadId ? 0 : 200}
                >
                  {qaPairs?.length ? (
                    <QuestionAnswerList messages={qaPairs} />
                  ) : (
                    <>
                      <EmptyScreen
                        setInput={setInput}
                        welcomeMessage={welcomeMessage}
                        setShowHistory={setShowHistory}
                        onSelectThread={onThreadIdChange}
                      />
                    </>
                  )}
                </LoadingWrapper>
                <ChatScrollAnchor trackVisibility={isLoading} />
              </div>
            )}
            <ChatPanel
              onSubmit={handleSubmit}
              // FIXME(jueliang) remove mb
              className={cn(
                'fixed inset-x-0 bottom-0 z-40 mb-4',
                promptFormClassname
              )}
              stop={onStop}
              reload={onReload}
              input={input}
              setInput={setInput}
              onUpdate={onUpdate}
              ref={chatPanelRef}
              chatInputRef={chatInputRef}
            />
          </div>
        </div>
      </ChatContext.Provider>
    )
  }
)
Chat.displayName = 'Chat'

// get all keys from a union type, eg. T = A | B, then return keyof A | keyof B
type AllKeys<T> = T extends infer U ? keyof U : never

function appendContextAndDedupe(
  ctxList: Context[],
  newCtx: Context
): Context[] {
  const fieldsToIgnore: Array<AllKeys<Context>> = ['content']
  const isEqualIgnoringFields = (obj1: Context, obj2: Context) => {
    return isEqualWith(obj1, obj2, (_value1, _value2, key) => {
      // If the key is in the fieldsToIgnore array, consider the values equal
      if (fieldsToIgnore.includes(key as keyof Context)) {
        return true
      }
    })
  }
  if (!ctxList.some(ctx => isEqualIgnoringFields(ctx, newCtx))) {
    return ctxList.concat([newCtx])
  }
  return ctxList
}

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
