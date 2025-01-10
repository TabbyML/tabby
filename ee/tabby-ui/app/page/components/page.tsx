'use client'

import {
  createContext,
  CSSProperties,
  Dispatch,
  Fragment,
  SetStateAction,
  useEffect,
  useMemo,
  useRef,
  useState
} from 'react'
import Link from 'next/link'
import slugify from '@sindresorhus/slugify'
import { compact, pick, some, uniq, uniqBy } from 'lodash-es'
import { nanoid } from 'nanoid'
import { ImperativePanelHandle } from 'react-resizable-panels'
import { toast } from 'sonner'
import { useQuery } from 'urql'

import { ERROR_CODE_NOT_FOUND, SLUG_TITLE_MAX_LENGTH } from '@/lib/constants'
import { useEnableDeveloperMode } from '@/lib/experiment-flags'
import { graphql } from '@/lib/gql/generates'
import {
  CodeQueryInput,
  ContextInfo,
  DocQueryInput,
  InputMaybe,
  Maybe,
  Message,
  MessageAttachmentClientCode,
  Role
} from '@/lib/gql/generates/graphql'
import { useCopyToClipboard } from '@/lib/hooks/use-copy-to-clipboard'
import { useCurrentTheme } from '@/lib/hooks/use-current-theme'
import { useDebounceValue } from '@/lib/hooks/use-debounce'
import { useLatest } from '@/lib/hooks/use-latest'
import { useMe } from '@/lib/hooks/use-me'
import { useSelectedModel } from '@/lib/hooks/use-models'
import useRouterStuff from '@/lib/hooks/use-router-stuff'
import { useThreadRun } from '@/lib/hooks/use-thread-run'
import { updateSelectedModel } from '@/lib/stores/chat-actions'
import { clearHomeScrollPosition } from '@/lib/stores/scroll-store'
import { useMutation } from '@/lib/tabby/gql'
import {
  contextInfoQuery,
  listThreadMessages,
  listThreads,
  setThreadPersistedMutation
} from '@/lib/tabby/query'
import {
  AttachmentCodeItem,
  AttachmentDocItem,
  ExtendedCombinedError,
  ThreadRunContexts
} from '@/lib/types'
import {
  cn,
  getMentionsFromText,
  getThreadRunContextsFromMentions,
  getTitleFromMessages,
  isCodeSourceContext
} from '@/lib/utils'
import { Button, buttonVariants } from '@/components/ui/button'
import {
  IconClock,
  IconEye,
  IconFileSearch,
  IconList,
  IconPlus,
  IconSheet,
  IconStop
} from '@/components/ui/icons'
import { ScrollArea } from '@/components/ui/scroll-area'
import { ButtonScrollToBottom } from '@/components/button-scroll-to-bottom'
import { BANNER_HEIGHT, useShowDemoBanner } from '@/components/demo-banner'
import { MessageMarkdown } from '@/components/message-markdown'
import NotFoundPage from '@/components/not-found-page'
import TextAreaSearch from '@/components/textarea-search'
import { MyAvatar } from '@/components/user-avatar'

import { DocSelect } from './doc-select'
import { Header } from './header'
import { MessagesSkeleton } from './messages-skeleton'
import { Navbar } from './nav-bar'
import { RepoSelect } from './repo-select'
import { SectionContent } from './section-content'
import { SectionTitle } from './section-title'

export type ConversationMessage = Omit<
  Message,
  '__typename' | 'updatedAt' | 'createdAt' | 'attachment' | 'threadId'
> & {
  threadId?: string
  threadRelevantQuestions?: Maybe<string[]>
  error?: string
  attachment?: {
    clientCode?: Maybe<Array<MessageAttachmentClientCode>> | undefined
    code: Maybe<Array<AttachmentCodeItem>> | undefined
    doc: Maybe<Array<AttachmentDocItem>> | undefined
  }
}

export type ConversationPair = {
  question: ConversationMessage | null
  answer: ConversationMessage | null
}

type PageContextValue = {
  mode: 'edit' | 'view'
  setMode: Dispatch<SetStateAction<'view' | 'edit'>>
  // flag for initialize the pathname
  isPathnameInitialized: boolean
  isLoading: boolean
  onRegenerateResponse: (id: string) => void
  onSubmitSearch: (question: string) => void
  setDevPanelOpen: (v: boolean) => void
  setConversationIdForDev: (v: string | undefined) => void
  enableDeveloperMode: boolean
  contextInfo: ContextInfo | undefined
  fetchingContextInfo: boolean
  onDeleteMessage: (id: string) => void
  isThreadOwner: boolean
  onUpdateMessage: (
    message: ConversationMessage
  ) => Promise<ExtendedCombinedError | undefined>
}

export const PageContext = createContext<PageContextValue>(
  {} as PageContextValue
)

export const SOURCE_CARD_STYLE = {
  compress: 5.3,
  expand: 6.3
}

const PAGE_SIZE = 30

const TEMP_MSG_ID_PREFIX = '_temp_msg_'
const tempNanoId = () => `${TEMP_MSG_ID_PREFIX}${nanoid()}`

export function Page() {
  const [{ data: meData }] = useMe()
  const { updateUrlComponents, pathname } = useRouterStuff()
  const [activePathname, setActivePathname] = useState<string | undefined>()
  const [isPathnameInitialized, setIsPathnameInitialized] = useState(false)
  const [mode, setMode] = useState<'edit' | 'view'>('view')

  const [messages, setMessages] = useState<ConversationMessage[]>([])
  const [stopButtonVisible, setStopButtonVisible] = useState(true)
  const [isReady, setIsReady] = useState(false)
  const [currentUserMessageId, setCurrentUserMessageId] = useState<string>('')
  const [currentAssistantMessageId, setCurrentAssistantMessageId] =
    useState<string>('')
  const contentContainerRef = useRef<HTMLDivElement>(null)
  const [showSearchInput, setShowSearchInput] = useState(false)
  const [isShowDemoBanner] = useShowDemoBanner()
  const initializing = useRef(false)
  const { theme } = useCurrentTheme()
  const [devPanelOpen, setDevPanelOpen] = useState(false)
  const [messageIdForDev, setMessageIdForDev] = useState<string | undefined>()
  const devPanelRef = useRef<ImperativePanelHandle>(null)
  const [devPanelSize, setDevPanelSize] = useState(45)
  const prevDevPanelSize = useRef(devPanelSize)
  const [enableDeveloperMode] = useEnableDeveloperMode()
  const [threadId, setThreadId] = useState<string | undefined>()
  const threadIdFromURL = useMemo(() => {
    const regex = /^\/page\/(.*)/
    if (!activePathname) return undefined

    return activePathname.match(regex)?.[1]?.split('-').pop()
  }, [activePathname])

  const updateThreadMessage = useMutation(updateThreadMessageMutation)

  const onUpdateMessage = async (
    message: ConversationMessage
  ): Promise<ExtendedCombinedError | undefined> => {
    const messageIndex = messages.findIndex(o => o.id === message.id)
    if (messageIndex > -1 && threadId) {
      // 1. call api
      const result = await updateThreadMessage({
        input: {
          threadId,
          id: message.id,
          content: message.content
        }
      })
      if (result?.data?.updateThreadMessage) {
        // 2. set messages
        await setMessages(prev => {
          const newMessages = [...prev]
          newMessages[messageIndex] = message
          return newMessages
        })
      } else {
        return result?.error || new Error('Failed to save')
      }
    } else {
      return new Error('Failed to save')
    }
  }

  useEffect(() => {
    if (threadIdFromURL) {
      setThreadId(threadIdFromURL)
    }
  }, [threadIdFromURL])

  const [{ data: contextInfoData, fetching: fetchingContextInfo }] = useQuery({
    query: contextInfoQuery
  })

  const repos = useMemo(() => {
    return contextInfoData?.contextInfo?.sources.filter(x =>
      isCodeSourceContext(x.sourceKind)
    )
  }, [contextInfoData?.contextInfo?.sources])

  const docs = useMemo(() => {
    return contextInfoData?.contextInfo?.sources.filter(
      x => !isCodeSourceContext(x.sourceKind)
    )
  }, [contextInfoData?.contextInfo?.sources])

  const [afterCursor, setAfterCursor] = useState<string | undefined>()

  const [{ data: threadData, fetching: fetchingThread, error: threadError }] =
    useQuery({
      query: listThreads,
      variables: {
        ids: [threadId as string]
      },
      pause: !threadId
    })

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
      threadId: threadId as string,
      first: PAGE_SIZE,
      after: afterCursor
    },
    pause: !threadId || isReady
  })

  useEffect(() => {
    if (threadMessagesStale) return

    if (threadMessages?.threadMessages?.edges?.length) {
      const messages = threadMessages.threadMessages.edges
        .map(o => o.node)
        .slice()
      setMessages(prev => uniqBy([...prev, ...messages], 'id'))
    }

    if (threadMessages?.threadMessages) {
      const hasNextPage = threadMessages?.threadMessages?.pageInfo?.hasNextPage
      const endCursor = threadMessages?.threadMessages.pageInfo.endCursor
      if (hasNextPage && endCursor) {
        setAfterCursor(endCursor)
      } else {
        setIsReady(true)
      }
    }
  }, [threadMessages])

  const isThreadOwner = useMemo(() => {
    if (!meData) return false
    if (!threadIdFromURL) return true

    const thread = threadData?.threads.edges[0]
    if (!thread) return false

    return meData.me.id === thread.node.userId
  }, [meData, threadData, threadIdFromURL])

  // Compute title
  const sources = contextInfoData?.contextInfo.sources
  const content = messages?.[0]?.content
  const title = useMemo(() => {
    if (sources && content) {
      return getTitleFromMessages(sources, content, {
        maxLength: SLUG_TITLE_MAX_LENGTH
      })
    } else {
      return ''
    }
  }, [sources, content])

  // Update title
  useEffect(() => {
    if (title) {
      document.title = title
    }
  }, [title])

  useEffect(() => {
    if (threadMessagesError && !isReady) {
      setIsReady(true)
    }
  }, [threadMessagesError])

  // `/search` -> `/search/{slug}-{threadId}`
  const updateThreadURL = (threadId: string) => {
    const slug = slugify(title)
    const slugWithThreadId = compact([slug, threadId]).join('-')

    const path = updateUrlComponents({
      pathname: `/page/${slugWithThreadId}`,
      searchParams: {
        del: ['q']
      },
      replace: true
    })

    return location.origin + path
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
    threadId
  })

  const isLoadingRef = useLatest(isLoading)

  const { selectedModel, isModelLoading, models } = useSelectedModel()

  const currentMessageForDev = useMemo(() => {
    return messages.find(item => item.id === messageIdForDev)
  }, [messageIdForDev, messages])

  const valueForDev = useMemo(() => {
    if (currentMessageForDev) {
      return pick(currentMessageForDev?.attachment, 'doc', 'code')
    }
    return {
      answers: messages
        .filter(o => o.role === Role.Assistant)
        .map(o => pick(o, 'doc', 'code'))
    }
  }, [
    messageIdForDev,
    currentMessageForDev?.attachment?.code,
    currentMessageForDev?.attachment?.doc
  ])

  const onPanelLayout = (sizes: number[]) => {
    if (sizes?.[1]) {
      setDevPanelSize(sizes[1])
    }
  }

  // for synchronizing the active pathname
  useEffect(() => {
    setActivePathname(pathname)

    if (!isPathnameInitialized) {
      setIsPathnameInitialized(true)
    }
  }, [pathname])

  useEffect(() => {
    const init = () => {
      if (initializing.current) return

      initializing.current = true

      setIsReady(true)
    }

    if (isPathnameInitialized && !threadIdFromURL) {
      init()
    }
  }, [isPathnameInitialized])

  // Display the input field with a delayed animatio
  useEffect(() => {
    if (isReady) {
      setTimeout(() => {
        setShowSearchInput(true)
      }, 300)
    }
  }, [isReady])

  const persistenceDisabled = useMemo(() => {
    return !threadIdFromURL && some(messages, message => !!message.error)
  }, [threadIdFromURL, messages])

  const { isCopied: isShareLinkCopied, onShare: onClickShare } = useShareThread(
    {
      threadIdFromURL,
      threadIdFromStreaming: threadId,
      streamingDone: !isLoading,
      updateThreadURL
    }
  )

  // Handling the stream response from useThreadRun
  useEffect(() => {
    // update threadId
    if (answer.threadId && answer.threadId !== threadId) {
      setThreadId(answer.threadId)
    }

    let newMessages = [...messages]

    const currentUserMessageIdx = newMessages.findIndex(
      o => o.id === currentUserMessageId
    )
    const currentAssistantMessageIdx = newMessages.findIndex(
      o => o.id === currentAssistantMessageId
    )
    if (currentUserMessageIdx === -1 || currentAssistantMessageIdx === -1) {
      return
    }

    const currentUserMessage = newMessages[currentUserMessageIdx]
    const currentAssistantMessage = newMessages[currentAssistantMessageIdx]

    // update assistant message
    currentAssistantMessage.content = answer.content

    // get and format scores from streaming answer
    if (!currentAssistantMessage.attachment?.code && !!answer.attachmentsCode) {
      currentAssistantMessage.attachment = {
        clientCode: null,
        doc: currentAssistantMessage.attachment?.doc || null,
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
    if (!currentAssistantMessage.attachment?.doc && !!answer.attachmentsDoc) {
      currentAssistantMessage.attachment = {
        clientCode: null,
        doc:
          answer.attachmentsDoc.map(hit => ({
            ...hit.doc,
            extra: {
              score: hit.score
            }
          })) || null,
        code: currentAssistantMessage.attachment?.code || null
      }
    }

    currentAssistantMessage.threadRelevantQuestions = answer?.relevantQuestions

    // update message pair ids
    const newUserMessageId = answer.userMessageId
    const newAssistantMessageId = answer.assistantMessageId
    if (
      newUserMessageId &&
      newAssistantMessageId &&
      newUserMessageId !== currentUserMessage.id &&
      newAssistantMessageId !== currentAssistantMessage.id
    ) {
      currentUserMessage.id = newUserMessageId
      currentAssistantMessage.id = newAssistantMessageId
      setCurrentUserMessageId(newUserMessageId)
      setCurrentAssistantMessageId(newAssistantMessageId)
    }

    // update messages
    setMessages(newMessages)
  }, [isLoading, answer])

  // Handling the error response from useThreadRun
  useEffect(() => {
    if (error) {
      const newConversation = [...messages]
      const currentAnswer = newConversation.find(
        item => item.id === currentAssistantMessageId
      )
      if (currentAnswer) {
        currentAnswer.error = formatThreadRunErrorMessage(error)
      }
    }
  }, [error])

  // Delay showing the stop button
  const showStopTimeoutId = useRef<number>()

  useEffect(() => {
    if (isLoadingRef.current) {
      showStopTimeoutId.current = window.setTimeout(() => {
        if (!isLoadingRef.current) return
        setStopButtonVisible(true)

        // Scroll to the bottom
        const container = contentContainerRef?.current
        if (container) {
          container.scrollTo({
            top: container.scrollHeight,
            behavior: 'smooth'
          })
        }
      }, 300)
    }

    if (!isLoadingRef.current) {
      setStopButtonVisible(false)
    }

    return () => {
      window.clearTimeout(showStopTimeoutId.current)
    }
  }, [isLoading])

  useEffect(() => {
    if (devPanelOpen) {
      devPanelRef.current?.expand()
      devPanelRef.current?.resize(devPanelSize)
    } else {
      devPanelRef.current?.collapse()
    }
  }, [devPanelOpen])

  const onSubmitSearch = (question: string, ctx?: ThreadRunContexts) => {
    const newUserMessageId = tempNanoId()
    const newAssistantMessageId = tempNanoId()
    const newUserMessage: ConversationMessage = {
      id: newUserMessageId,
      role: Role.User,
      content: question
    }
    const newAssistantMessage: ConversationMessage = {
      id: newAssistantMessageId,
      role: Role.Assistant,
      content: ''
    }

    const { sourceIdForCodeQuery, sourceIdsForDocQuery, searchPublic } =
      getSourceInputs(ctx)

    const codeQuery: InputMaybe<CodeQueryInput> = sourceIdForCodeQuery
      ? { sourceId: sourceIdForCodeQuery, content: question }
      : null

    const docQuery: InputMaybe<DocQueryInput> = {
      sourceIds: sourceIdsForDocQuery,
      content: question,
      searchPublic: !!searchPublic
    }

    setCurrentUserMessageId(newUserMessageId)
    setCurrentAssistantMessageId(newAssistantMessageId)
    setMessages([...messages].concat([newUserMessage, newAssistantMessage]))

    sendUserMessage(
      {
        content: question
      },
      {
        generateRelevantQuestions: true,
        codeQuery,
        docQuery,
        modelName: ctx?.modelName
      }
    )
  }

  // regenerate ths last assistant message
  const onRegenerateResponse = () => {
    if (!threadId) return
    // need to get the sources from contextInfo
    if (fetchingContextInfo) return

    const assistantMessageIndex = messages.length - 1
    const userMessageIndex = assistantMessageIndex - 1
    if (assistantMessageIndex === -1 || userMessageIndex <= -1) return

    const prevUserMessageId = messages[userMessageIndex].id
    const prevAssistantMessageId = messages[assistantMessageIndex].id

    const newMessages = messages.slice(0, -2)
    const userMessage = messages[userMessageIndex]
    const newUserMessage: ConversationMessage = {
      ...userMessage,
      id: tempNanoId()
    }
    const newAssistantMessage: ConversationMessage = {
      id: tempNanoId(),
      role: Role.Assistant,
      content: '',
      attachment: {
        code: null,
        doc: null,
        clientCode: null
      },
      error: undefined
    }

    const mentions = getMentionsFromText(
      newUserMessage.content,
      contextInfoData?.contextInfo?.sources
    )

    const { sourceIdForCodeQuery, sourceIdsForDocQuery, searchPublic } =
      getSourceInputs(getThreadRunContextsFromMentions(mentions))

    const codeQuery: InputMaybe<CodeQueryInput> = sourceIdForCodeQuery
      ? { sourceId: sourceIdForCodeQuery, content: newUserMessage.content }
      : null

    const docQuery: InputMaybe<DocQueryInput> = {
      sourceIds: sourceIdsForDocQuery,
      content: newUserMessage.content,
      searchPublic
    }

    setCurrentUserMessageId(newUserMessage.id)
    setCurrentAssistantMessageId(newAssistantMessage.id)
    setMessages([...newMessages, newUserMessage, newAssistantMessage])

    regenerate({
      threadId,
      userMessageId: prevUserMessageId,
      assistantMessageId: prevAssistantMessageId,
      userMessage: {
        content: newUserMessage.content
      },
      threadRunOptions: {
        generateRelevantQuestions: true,
        codeQuery,
        docQuery,
        modelName: selectedModel
      }
    })
  }

  const onToggleFullScreen = (fullScreen: boolean) => {
    let nextSize = prevDevPanelSize.current
    if (fullScreen) {
      nextSize = 100
    } else if (nextSize === 100) {
      nextSize = 45
    }
    devPanelRef.current?.resize(nextSize)
    setDevPanelSize(nextSize)
    prevDevPanelSize.current = devPanelSize
  }

  const onDeleteMessage = (asistantMessageId: string) => {
    if (!threadId) return
    // find userMessageId by assistantMessageId
    const assistantMessageIndex = messages.findIndex(
      message => message.id === asistantMessageId
    )
    const userMessageIndex = assistantMessageIndex - 1
    const userMessage = messages[assistantMessageIndex - 1]

    if (assistantMessageIndex === -1 || userMessage?.role !== Role.User) {
      return
    }

    // message pair not successfully created in threadrun
    if (
      userMessage.id.startsWith(TEMP_MSG_ID_PREFIX) &&
      asistantMessageId.startsWith(TEMP_MSG_ID_PREFIX)
    ) {
      const newMessages = messages
        .slice(0, userMessageIndex)
        .concat(messages.slice(assistantMessageIndex + 1))
      setMessages(newMessages)
      return
    }

    deleteThreadMessagePair(threadId, userMessage.id, asistantMessageId).then(
      errorMessage => {
        if (errorMessage) {
          toast.error(errorMessage)
          return
        }

        // remove userMessage and assistantMessage
        const newMessages = messages
          .slice(0, userMessageIndex)
          .concat(messages.slice(assistantMessageIndex + 1))
        setMessages(newMessages)
      }
    )
  }

  const onModelSelect = (model: string) => {
    updateSelectedModel(model)
  }

  const formatedThreadError: ExtendedCombinedError | undefined = useMemo(() => {
    if (!isReady || fetchingThread || !threadIdFromURL) return undefined
    if (threadError || !threadData?.threads?.edges?.length) {
      return threadError || new Error(ERROR_CODE_NOT_FOUND)
    }
  }, [threadData, fetchingThread, threadError, isReady, threadIdFromURL])

  const [isFetchingMessages] = useDebounceValue(
    fetchingMessages || threadMessages?.threadMessages?.pageInfo?.hasNextPage,
    200
  )

  const qaPairs = useMemo(() => {
    const pairs: Array<ConversationPair> = []
    let currentPair: ConversationPair = { question: null, answer: null }
    messages.forEach(message => {
      if (message.role === Role.User) {
        currentPair.question = message
      } else if (message.role === Role.Assistant) {
        if (!currentPair.answer) {
          // Take the first answer
          currentPair.answer = message
          pairs.push(currentPair)
          currentPair = { question: null, answer: null }
        }
      }
    })

    return pairs
  }, [messages])

  const style = isShowDemoBanner
    ? { height: `calc(100vh - ${BANNER_HEIGHT})` }
    : { height: '100vh' }

  if (isReady && (formatedThreadError || threadMessagesError)) {
    return (
      <ThreadMessagesErrorView
        error={
          (formatedThreadError || threadMessagesError) as ExtendedCombinedError
        }
        threadIdFromURL={threadIdFromURL}
      />
    )
  }

  if (!isReady && (isFetchingMessages || threadMessagesStale)) {
    return (
      <div>
        <Header />
        <div className="mx-auto mt-24 w-full space-y-10 px-4 pb-32 lg:max-w-5xl lg:px-0">
          <MessagesSkeleton />
          <MessagesSkeleton />
        </div>
      </div>
    )
  }

  if (!isReady) {
    return <></>
  }

  return (
    <PageContext.Provider
      value={{
        isLoading,
        onRegenerateResponse,
        onSubmitSearch,
        setDevPanelOpen,
        setConversationIdForDev: setMessageIdForDev,
        isPathnameInitialized,
        enableDeveloperMode: enableDeveloperMode.value,
        contextInfo: contextInfoData?.contextInfo,
        fetchingContextInfo,
        onDeleteMessage,
        isThreadOwner,
        onUpdateMessage,
        mode,
        setMode
      }}
    >
      <div style={style}>
        <Header threadIdFromURL={threadIdFromURL} streamingDone={!isLoading} />
        <main className="h-[calc(100%-4rem)] pb-8 lg:pb-0">
          <ScrollArea className="h-full w-full" ref={contentContainerRef}>
            <div className="mx-auto grid grid-cols-4 gap-2 px-4 pb-32 lg:max-w-5xl lg:px-0">
              <div className="col-span-3">
                {/* page title */}
                <div className="mb-2 mt-4">
                  <h1 className="text-4xl font-semibold">
                    Tailwindcss in TabbyML
                  </h1>
                  <div className="my-4 flex justify-between text-sm text-muted-foreground">
                    <div className="flex items-center gap-1">
                      <MyAvatar className="h-6 w-6" />
                      <div>{meData?.me?.name}</div>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="flex items-center gap-0.5">
                        <IconClock />
                        <span>2 hours ago</span>
                      </div>
                      <div className="flex items-center gap-0.5">
                        <IconEye />
                        <span>345</span>
                      </div>
                    </div>
                  </div>
                </div>
                {/* page summary */}
                {/* FIXME mock */}
                <MessageMarkdown
                  message={
                    'Tailwind CSS is a utility-first CSS framework designed for rapid UI development. Unlike traditional CSS frameworks that provide pre-designed components, Tailwind offers a wide array of utility classes that you can combine to build custom designs without leaving your HTML. This approach gives developers the flexibility and control to create unique and responsive designs efficiently.'
                  }
                />

                {/* sections */}
                <div className="flex flex-col">
                  {qaPairs.map((pair, index) => {
                    const isLastMessage = index === qaPairs.length - 1
                    if (!pair.question) return null

                    return (
                      <Fragment key={pair.question.id}>
                        {!!pair.question && (
                          <SectionTitle
                            className="section-title pt-8"
                            key={pair.question.id}
                            message={pair.question}
                          />
                        )}
                        {!!pair.answer && (
                          <SectionContent
                            key={pair.answer.id}
                            className="pb-8"
                            message={pair.answer}
                            clientCode={pair.question?.attachment?.clientCode}
                            isLoading={isLoading && isLastMessage}
                            isLastAssistantMessage={isLastMessage}
                            showRelatedQuestion={isLastMessage}
                            isDeletable={!isLoading && messages.length > 2}
                          />
                        )}
                        {!isLastMessage && mode === 'edit' && (
                          <div className="mx-auto">
                            <Button variant="outline">Insert Section</Button>
                          </div>
                        )}
                      </Fragment>
                    )
                  })}
                </div>
                {mode === 'edit' && (
                  <div className="rounded-xl border p-2">
                    <div className="flex items-center gap-1">
                      <Button size="icon" variant="ghost">
                        <IconList />
                      </Button>
                      <Button size="icon" variant="ghost">
                        <IconSheet />
                      </Button>
                      <RepoSelect repos={repos} />
                      <DocSelect docs={docs} />
                    </div>
                    <TextAreaSearch
                      onSearch={onSubmitSearch}
                      className="min-h-[5rem] border-0 lg:max-w-5xl"
                      placeholder="What is the section about?"
                      isFollowup
                      isLoading={isLoading}
                      contextInfo={contextInfoData?.contextInfo}
                      fetchingContextInfo={fetchingContextInfo}
                      modelName={selectedModel}
                      onModelSelect={onModelSelect}
                      isModelLoading={isModelLoading}
                      models={models}
                    />
                  </div>
                )}
              </div>
              <div className="relative col-span-1">
                <Navbar qaPairs={qaPairs} />
              </div>
            </div>
          </ScrollArea>

          <ButtonScrollToBottom
            className={cn(
              '!fixed !bottom-[5.4rem] !right-4 !top-auto z-40 border-muted-foreground lg:!bottom-[2.85rem]',
              {
                hidden: devPanelOpen
              }
            )}
            container={contentContainerRef.current as HTMLDivElement}
            offset={100}
            // On mobile browsers(Chrome & Safari) in dark mode, using `background: hsl(var(--background))`
            // result in `rgba(0, 0, 0, 0)`. To prevent this, explicitly set --background
            style={
              theme === 'dark'
                ? ({ '--background': '0 0% 12%' } as CSSProperties)
                : {}
            }
          />

          <div
            className={cn(
              'fixed bottom-5 left-0 z-30 flex min-h-[3rem] w-full flex-col items-center gap-y-2',
              {
                'opacity-100 translate-y-0': showSearchInput,
                'opacity-0 translate-y-10': !showSearchInput,
                hidden: devPanelOpen
              }
            )}
            style={Object.assign(
              { transition: 'all 0.35s ease-out' },
              theme === 'dark'
                ? ({ '--background': '0 0% 12%' } as CSSProperties)
                : {}
            )}
          >
            <div
              className={cn('absolute flex items-center gap-4')}
              style={isThreadOwner ? { top: '-2.5rem' } : undefined}
            >
              {stopButtonVisible && (
                <Button
                  className="bg-background"
                  variant="outline"
                  onClick={() => stop()}
                >
                  <IconStop className="mr-2" />
                  Stop generating
                </Button>
              )}
              {/* {!stopButtonVisible && mode === 'view' && (
                <Tooltip delayDuration={0}>
                  <TooltipTrigger asChild>
                    <span tabIndex={0}>
                      <Button
                        className="gap-2 bg-background"
                        variant="outline"
                        onClick={onClickShare}
                        disabled={persistenceDisabled}
                      >
                        {persistenceDisabled ? (
                          <IconInfoCircled />
                        ) : isShareLinkCopied ? (
                          <IconCheck className="text-green-600" />
                        ) : (
                          <IconShare />
                        )}
                        Share Link
                      </Button>
                    </span>
                  </TooltipTrigger>
                  <TooltipContent hidden={!persistenceDisabled}>
                    Please resolve errors in messages before sharing this
                    thread.
                  </TooltipContent>
                </Tooltip>
              )} */}
            </div>
            {mode === 'view' && (
              <div
                className={cn(
                  'relative z-20 flex justify-center self-stretch px-4'
                )}
              >
                <TextAreaSearch
                  onSearch={onSubmitSearch}
                  className="min-h-[5rem] lg:max-w-4xl"
                  placeholder="Ask a follow up question"
                  isFollowup
                  isLoading={isLoading}
                  contextInfo={contextInfoData?.contextInfo}
                  fetchingContextInfo={fetchingContextInfo}
                  modelName={selectedModel}
                  onModelSelect={onModelSelect}
                  isModelLoading={isModelLoading}
                  models={models}
                />
              </div>
            )}
          </div>
        </main>
      </div>
    </PageContext.Provider>
  )
}

const updateThreadMessageMutation = graphql(/* GraphQL */ `
  mutation UpdateThreadMessage($input: UpdateMessageInput!) {
    updateThreadMessage(input: $input)
  }
`)

interface ThreadMessagesErrorViewProps {
  error: ExtendedCombinedError
  threadIdFromURL?: string
}
function ThreadMessagesErrorView({
  error,
  threadIdFromURL
}: ThreadMessagesErrorViewProps) {
  let title = 'Something went wrong'
  let description =
    'Failed to fetch the thread, please refresh the page or start a new thread'

  if (error.message === ERROR_CODE_NOT_FOUND) {
    return <NotFoundPage />
  }

  return (
    <div className="flex h-screen flex-col">
      <Header threadIdFromURL={threadIdFromURL} />
      <div className="flex-1">
        <div className="flex h-full flex-col items-center justify-center gap-2">
          <div className="flex items-center gap-2">
            <IconFileSearch className="h-6 w-6" />
            <div className="text-xl font-semibold">{title}</div>
          </div>
          <div>{description}</div>
          <Link
            href="/"
            onClick={clearHomeScrollPosition}
            className={cn(buttonVariants(), 'mt-4 gap-2')}
          >
            <IconPlus />
            <span>New Page</span>
          </Link>
        </div>
      </div>
    </div>
  )
}

function getSourceInputs(ctx: ThreadRunContexts | undefined) {
  let sourceIdsForDocQuery: string[] = []
  let sourceIdForCodeQuery: string | undefined
  let searchPublic = false

  if (ctx) {
    sourceIdsForDocQuery = uniq(
      compact([ctx?.codeSourceIds?.[0]].concat(ctx.docSourceIds))
    )
    searchPublic = ctx.searchPublic ?? false
    sourceIdForCodeQuery = ctx.codeSourceIds?.[0] ?? undefined
  }
  return {
    sourceIdsForDocQuery,
    sourceIdForCodeQuery,
    searchPublic
  }
}

interface UseShareThreadOptions {
  threadIdFromURL?: string
  threadIdFromStreaming?: string | null
  streamingDone?: boolean
  updateThreadURL?: (threadId: string) => string
}

function useShareThread({
  threadIdFromURL,
  threadIdFromStreaming,
  streamingDone,
  updateThreadURL
}: UseShareThreadOptions) {
  const { isCopied, copyToClipboard } = useCopyToClipboard({
    timeout: 2000
  })

  const setThreadPersisted = useMutation(setThreadPersistedMutation, {
    onError(err) {
      toast.error(err.message)
    }
  })

  const shouldSetThreadPersisted =
    !threadIdFromURL &&
    streamingDone &&
    threadIdFromStreaming &&
    updateThreadURL

  const onShare = async () => {
    if (isCopied) return

    let url = window.location.href
    if (shouldSetThreadPersisted) {
      await setThreadPersisted({ threadId: threadIdFromStreaming })
      url = updateThreadURL(threadIdFromStreaming)
    }

    copyToClipboard(url)
  }

  return {
    onShare,
    isCopied
  }
}

function formatThreadRunErrorMessage(error?: ExtendedCombinedError) {
  if (!error) return 'Failed to fetch'

  if (error.message === '401') {
    return 'Unauthorized'
  }

  if (
    some(error.graphQLErrors, o => o.extensions?.code === ERROR_CODE_NOT_FOUND)
  ) {
    return `The thread has expired or does not exist.`
  }

  return error.message || 'Failed to fetch'
}
