'use client'

import {
  createContext,
  CSSProperties,
  MouseEventHandler,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState
} from 'react'
import Image from 'next/image'
import { useRouter } from 'next/navigation'
import defaultFavicon from '@/assets/default-favicon.png'
import DOMPurify from 'dompurify'
import he from 'he'
import { marked } from 'marked'
import { nanoid } from 'nanoid'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'

import { SESSION_STORAGE_KEY } from '@/lib/constants'
import { useEnableDeveloperMode, useEnableSearch } from '@/lib/experiment-flags'
import { useCurrentTheme } from '@/lib/hooks/use-current-theme'
import { useLatest } from '@/lib/hooks/use-latest'
import { useIsChatEnabled } from '@/lib/hooks/use-server-info'
import { AnswerEngineExtraContext, RelevantCodeContext } from '@/lib/types'
import { cn, formatLineHashForCodeBrowser } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { CodeBlock } from '@/components/ui/codeblock'
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger
} from '@/components/ui/hover-card'
import {
  IconBlocks,
  IconBug,
  IconChevronLeft,
  IconChevronRight,
  IconLayers,
  IconPlus,
  // IconRefresh,
  IconSparkles,
  IconSpinner,
  IconStop
} from '@/components/ui/icons'
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup
} from '@/components/ui/resizable'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { Skeleton } from '@/components/ui/skeleton'
import { ButtonScrollToBottom } from '@/components/button-scroll-to-bottom'
import { ClientOnly } from '@/components/client-only'
import { CopyButton } from '@/components/copy-button'
import { BANNER_HEIGHT, useShowDemoBanner } from '@/components/demo-banner'
import { MemoizedReactMarkdown } from '@/components/markdown'
import TextAreaSearch from '@/components/textarea-search'
import { ThemeToggle } from '@/components/theme-toggle'
import { UserAvatar } from '@/components/user-avatar'
import UserPanel from '@/components/user-panel'

import './search.css'

import { compact, isEmpty, isNil, pick } from 'lodash-es'
import { ImperativePanelHandle } from 'react-resizable-panels'
import slugify from 'slugify'
import { Context } from 'tabby-chat-panel/index'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import {
  CodeQueryInput,
  InputMaybe,
  Maybe,
  Message,
  MessageAttachmentCode,
  MessageAttachmentDoc,
  MessageCodeSearchHit,
  MessageDocSearchHit,
  RepositoryListQuery,
  Role
} from '@/lib/gql/generates/graphql'
import useRouterStuff from '@/lib/hooks/use-router-stuff'
import { useThreadRun } from '@/lib/hooks/use-thread-run'
import { repositoryListQuery } from '@/lib/tabby/query'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'
import { CodeReferences } from '@/components/chat/question-answer'

import { DevPanel } from './dev-panel'
import { MessagesSkeleton } from './messages-skeleton'

type ConversationMessage = Omit<
  Message,
  '__typename' | 'updatedAt' | 'createdAt' | 'attachment' | 'threadId'
> & {
  threadId?: string
  threadRelevantQuestions?: Maybe<string[]>
  // isLoading?: boolean
  error?: string
  attachment?: {
    code: Array<AttachmentCodeItem>
    doc: Array<AttachmentDocItem>
  }
}

// for rendering, including scores
type AttachmentCodeItem = MessageAttachmentCode & {
  extra?: { scores?: MessageCodeSearchHit['scores'] }
}
// for rendering, including score
type AttachmentDocItem = MessageAttachmentDoc & {
  extra?: { score?: MessageDocSearchHit['score'] }
}

type SearchContextValue = {
  // flag for initialize the pathname
  isPathnameInitialized: boolean
  isLoading: boolean
  onRegenerateResponse: (id: string) => void
  onSubmitSearch: (question: string) => void
  extraRequestContext: Record<string, any>
  repositoryList: RepositoryListQuery['repositoryList'] | undefined
  setDevPanelOpen: (v: boolean) => void
  setConversationIdForDev: (v: string | undefined) => void
}

export const SearchContext = createContext<SearchContextValue>(
  {} as SearchContextValue
)

const SOURCE_CARD_STYLE = {
  compress: 5.3,
  expand: 6.3
}

const listThreadMessages = graphql(/* GraphQL */ `
  query ListThreadMessages(
    $threadId: ID!
    $after: String
    $before: String
    $first: Int
    $last: Int
  ) {
    threadMessages(
      threadId: $threadId
      after: $after
      before: $before
      first: $first
      last: $last
    ) {
      edges {
        node {
          id
          threadId
          role
          content
          attachment {
            code {
              gitUrl
              filepath
              language
              content
              startLine
            }
            doc {
              title
              link
              content
            }
          }
        }
        cursor
      }
      pageInfo {
        hasNextPage
        hasPreviousPage
        startCursor
        endCursor
      }
    }
  }
`)

export function Search() {
  const { updateUrlComponents, pathname, searchParams } = useRouterStuff()
  const isNewThread = searchParams.get('q')?.toString() === 'new'
  const [activePathname, setActivePathname] = useState<string | undefined>()
  const [isPathnameInitialized, setIsPathnameInitialized] = useState(false)
  const isChatEnabled = useIsChatEnabled()
  const [searchFlag] = useEnableSearch()
  const [messages, setMessages] = useState<ConversationMessage[]>([])
  const [stopButtonVisible, setStopButtonVisible] = useState(true)
  const [isReady, setIsReady] = useState(false)
  const [extraContext, setExtraContext] = useState<AnswerEngineExtraContext>({})
  const [currentAssistantMessageId, setCurrentAssistantMessageId] =
    useState<string>('')
  const contentContainerRef = useRef<HTMLDivElement>(null)
  const [showSearchInput, setShowSearchInput] = useState(false)
  const [isShowDemoBanner] = useShowDemoBanner()
  const router = useRouter()
  const initializing = useRef(false)
  const { theme } = useCurrentTheme()
  const [devPanelOpen, setDevPanelOpen] = useState(false)
  const [messageIdForDev, setMessageIdForDev] = useState<string | undefined>()
  const devPanelRef = useRef<ImperativePanelHandle>(null)
  const [devPanelSize, setDevPanelSize] = useState(45)
  const prevDevPanelSize = useRef(devPanelSize)

  const threadId = useMemo(() => {
    const regex = /^\/search\/(.*)/
    if (!activePathname) return undefined

    return activePathname.match(regex)?.[1]?.split('-').pop()
  }, [activePathname])

  const [{ data }] = useQuery({
    query: repositoryListQuery
  })
  const repositoryList = data?.repositoryList

  const [afterCursor, setAfterCursor] = useState<string | undefined>()
  const [{ data: threadMessages, fetching: fetchingMessages }] = useQuery({
    query: listThreadMessages,
    variables: {
      threadId: threadId as string,
      first: 20,
      after: afterCursor
    },
    pause: !threadId || isReady
  })

  // todo scroll and setAfterCursor
  useEffect(() => {
    if (threadMessages?.threadMessages?.edges?.length) {
      const messages = threadMessages.threadMessages.edges
        .map(o => o.node)
        .slice()
      setMessages(messages)
      setIsReady(true)
    }
  }, [threadMessages])

  const onThreadCreated = (threadId: string) => {
    const title = messages?.[0]?.content
    const slug = slugify(title, {
      lower: true
    })
      .split('-')
      .slice(0, 8)
      .join('-')

    if (slug) {
      document.title = slug
    } else {
      document.title = title
    }

    const slugWithThreadId = compact([slug, threadId]).join('-')

    updateUrlComponents({
      pathname: `/search/${slugWithThreadId}`,
      searchParams: {
        del: ['q']
      }
    })
  }

  const { triggerRequest, isLoading, error, answer, stop } = useThreadRun({
    threadId,
    onThreadCreated
  })

  const isLoadingRef = useLatest(isLoading)

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
    // prevActivePath.current = activePath
    setActivePathname(pathname)

    if (!isPathnameInitialized) {
      setIsPathnameInitialized(true)
    }
  }, [pathname])

  // Check sessionStorage for initial message or most recent conversation
  useEffect(() => {
    const init = () => {
      if (initializing.current) return

      initializing.current = true

      if (isNewThread) {
        // initial UserMessage from home page
        const initialMessage = sessionStorage.getItem(
          SESSION_STORAGE_KEY.SEARCH_INITIAL_MSG
        )
        // initial extra context from home page
        const initialExtraContextStr = sessionStorage.getItem(
          SESSION_STORAGE_KEY.SEARCH_INITIAL_EXTRA_CONTEXT
        )
        const initialExtraInfo = initialExtraContextStr
          ? JSON.parse(initialExtraContextStr)
          : undefined

        if (initialMessage) {
          sessionStorage.removeItem(SESSION_STORAGE_KEY.SEARCH_INITIAL_MSG)
          sessionStorage.removeItem(
            SESSION_STORAGE_KEY.SEARCH_INITIAL_EXTRA_CONTEXT
          )
          setIsReady(true)
          setExtraContext(p => ({
            ...p,
            repository: initialExtraInfo?.repository
          }))
          onSubmitSearch(initialMessage, {
            repository: initialExtraInfo?.repository
          })
          return
        }
      }

      if (!threadId) {
        router.replace('/')
      }
    }

    if (isPathnameInitialized) {
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

  // Handling the stream response from useThreadRun
  useEffect(() => {
    // if (!answer) return

    let newMessages = [...messages]
    const currentAssistantMessageIndex = newMessages.findIndex(
      item => item.id === currentAssistantMessageId
    )

    if (currentAssistantMessageIndex <= 0) return

    // updateUserMessageId
    // if (threadRun?.threadUserMessageCreated) {
    //   const currentUserMessage = newMessages[currentAssistantMessageIndex - 1]
    //   newMessages = getMessagesWithNewMessageId(
    //     newMessages,
    //     currentUserMessage.id,
    //     threadRun.threadUserMessageCreated
    //   )
    // }

    const currentAssistantMessage = newMessages[currentAssistantMessageIndex]

    // updateAssistantMessageId
    // if (threadRun?.threadAssistantMessageCreated) {
    //   currentAssistantMessage.id = threadRun.threadAssistantMessageCreated
    //   setCurrentAssistantMessageId(threadRun?.threadAssistantMessageCreated)
    // }

    currentAssistantMessage.content =
      answer?.threadAssistantMessageContentDelta || ''
    currentAssistantMessage.attachment = {
      // format the attachments
      code:
        answer?.threadAssistantMessageAttachmentsCode?.map(hit => ({
          ...hit.code,
          extra: {
            scores: hit.scores
          }
        })) ?? [],
      doc:
        answer?.threadAssistantMessageAttachmentsDoc?.map(hit => ({
          ...hit.doc,
          extra: {
            score: hit.score
          }
        })) ?? []
    }
    currentAssistantMessage.threadRelevantQuestions =
      answer?.threadRelevantQuestions

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
        currentAnswer.error =
          error.message === '401' ? 'Unauthorized' : 'Fail to fetch'
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
        const container = contentContainerRef?.current?.children?.[1]
        if (container) {
          const isLastAnswerLoading =
            currentAssistantMessageId === messages[messages.length - 1].id
          if (isLastAnswerLoading) {
            container.scrollTo({
              top: container.scrollHeight,
              behavior: 'smooth'
            })
          }
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

  // Stop stream before closing the page
  useEffect(() => {
    return () => {
      if (isLoadingRef.current) stop()
    }
  }, [])

  useEffect(() => {
    if (devPanelOpen) {
      devPanelRef.current?.expand()
      devPanelRef.current?.resize(devPanelSize)
    } else {
      devPanelRef.current?.collapse()
    }
  }, [devPanelOpen])

  const onSubmitSearch = (question: string, ctx?: AnswerEngineExtraContext) => {
    const newAssistantId = nanoid()
    const newUserMessage: ConversationMessage = {
      id: nanoid(),
      role: Role.User,
      content: question
    }
    const newAssistantMessage: ConversationMessage = {
      id: newAssistantId,
      role: Role.Assistant,
      content: ''
    }

    const _repository = ctx?.repository || extraContext?.repository
    const codeQuery: InputMaybe<CodeQueryInput> = _repository
      ? { gitUrl: _repository.gitUrl, content: question }
      : null

    setCurrentAssistantMessageId(newAssistantId)
    setMessages([...messages].concat([newUserMessage, newAssistantMessage]))

    triggerRequest(
      {
        content: question
      },
      {
        generateRelevantQuestions: true,
        codeQuery,
        docQuery: { content: question }
      }
    )
  }

  // FIXME
  const onRegenerateResponse = (
    id: string,
    conversationData?: ConversationMessage[]
  ) => {
    const data = conversationData || messages
    const targetAnswerIdx = data.findIndex(item => item.id === id)
    if (targetAnswerIdx < 1) return
    const targetQuestionIdx = targetAnswerIdx - 1
    const targetQuestion = data[targetQuestionIdx]

    const newMessages = [...data]
    let newTargetAnswer = newMessages[targetAnswerIdx]
    newTargetAnswer.content = ''
    newTargetAnswer.attachment = {
      doc: [],
      code: []
    }
    newTargetAnswer.error = undefined
    // newTargetAnswer.isLoading = true

    setCurrentAssistantMessageId(newTargetAnswer.id)
    setMessages(newMessages)
    triggerRequest(
      {
        content: targetQuestion.content
      },
      {
        generateRelevantQuestions: true,
        docQuery: { content: targetQuestion.content }
        // FIXME docQuery
      }
    )
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

  if (!isReady && !isNewThread) {
    return (
      <div className="mx-auto mt-24 w-full space-y-10 px-4 pb-32 lg:max-w-4xl lg:px-0">
        <MessagesSkeleton />
        <MessagesSkeleton />
      </div>
    )
  }

  if (!searchFlag.value || !isChatEnabled || !isReady) {
    return <></>
  }

  const style = isShowDemoBanner
    ? { height: `calc(100vh - ${BANNER_HEIGHT})` }
    : { height: '100vh' }

  return (
    <SearchContext.Provider
      value={{
        isLoading,
        onRegenerateResponse,
        onSubmitSearch,
        extraRequestContext: extraContext,
        repositoryList,
        setDevPanelOpen,
        setConversationIdForDev: setMessageIdForDev,
        isPathnameInitialized
      }}
    >
      <div className="transition-all" style={style}>
        <ResizablePanelGroup direction="vertical" onLayout={onPanelLayout}>
          <ResizablePanel>
            <header className="flex h-16 items-center justify-between px-4">
              <div className="flex items-center gap-x-6">
                <Button
                  variant="ghost"
                  className="-ml-1 pl-0 text-sm text-muted-foreground"
                  onClick={() => router.replace('/')}
                >
                  <IconChevronLeft className="mr-1 h-5 w-5" />
                  Home
                </Button>
              </div>
              <div className="flex items-center gap-x-6">
                {!!threadId && (
                  <Button
                    variant="outline"
                    className="gap-2"
                    onClick={() => router.push('/')}
                  >
                    <IconPlus />
                    New Thread
                  </Button>
                )}
                <ClientOnly>
                  <ThemeToggle />
                </ClientOnly>
                <UserPanel showHome={false} showSetting>
                  <UserAvatar className="h-10 w-10 border" />
                </UserPanel>
              </div>
            </header>

            <main className="h-[calc(100%-4rem)] overflow-auto pb-8 lg:pb-0">
              <ScrollArea className="h-full" ref={contentContainerRef}>
                <div className="mx-auto px-4 pb-32 lg:max-w-4xl lg:px-0">
                  <div className="flex flex-col">
                    {messages.map((item, idx) => {
                      if (item.role === Role.User) {
                        return (
                          <div key={item.id + idx}>
                            {idx !== 0 && <Separator />}
                            <div className="pb-2 pt-8">
                              <MessageMarkdown
                                message={item.content}
                                headline
                              />
                            </div>
                          </div>
                        )
                      }
                      if (item.role === Role.Assistant) {
                        const isLastAssistantMessage =
                          idx === messages.length - 1
                        return (
                          <div key={item.id + idx} className="pb-8 pt-2">
                            <AnswerBlock
                              answer={item}
                              isLastAssistantMessage={isLastAssistantMessage}
                              showRelatedQuestion={isLastAssistantMessage}
                              isLoading={isLoading && isLastAssistantMessage}
                            />
                          </div>
                        )
                      }
                      return <></>
                    })}
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
                container={
                  contentContainerRef.current?.children?.[1] as HTMLDivElement
                }
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
                  'fixed bottom-5 left-0 z-30 flex min-h-[5rem] w-full flex-col items-center gap-y-2',
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
                <Button
                  className={cn('bg-background', {
                    'opacity-0 pointer-events-none': !stopButtonVisible,
                    'opacity-100': stopButtonVisible
                  })}
                  style={{
                    transition: 'opacity 0.55s ease-out'
                  }}
                  variant="outline"
                  onClick={stop}
                >
                  <IconStop className="mr-2" />
                  Stop generating
                </Button>
                <div
                  className={cn(
                    'relative z-20 flex justify-center self-stretch px-4'
                  )}
                >
                  <TextAreaSearch
                    onSearch={onSubmitSearch}
                    className="lg:max-w-4xl"
                    placeholder="Ask a follow up question"
                    isLoading={isLoading}
                    isFollowup
                    extraContext={extraContext}
                  />
                </div>
              </div>
            </main>
          </ResizablePanel>
          <ResizableHandle
            className={cn(
              'hidden !h-[4px] border-none bg-background shadow-[0px_-4px_4px_rgba(0,0,0,0.2)] hover:bg-blue-500 active:bg-blue-500 dark:shadow-[0px_-4px_4px_rgba(255,255,255,0.2)]',
              devPanelOpen && 'block'
            )}
          />
          <ResizablePanel
            collapsible
            collapsedSize={0}
            defaultSize={0}
            ref={devPanelRef}
            onCollapse={() => setDevPanelOpen(false)}
            className="z-50"
          >
            <DevPanel
              onClose={() => setDevPanelOpen(false)}
              value={valueForDev}
              isFullScreen={devPanelSize === 100}
              onToggleFullScreen={onToggleFullScreen}
            />
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>
    </SearchContext.Provider>
  )
}

type AnswerBlockContextValue = {
  onCodeCitationMouseEnter: (index: number) => void
  onCodeCitationMouseLeave: (index: number) => void
  onCodeCitationClick: (data: MessageAttachmentCode) => void
}

const AnswerBlockContext = createContext<AnswerBlockContextValue>(
  {} as AnswerBlockContextValue
)

function AnswerBlock({
  answer,
  showRelatedQuestion,
  isLoading,
  isLastAssistantMessage
}: {
  answer: ConversationMessage
  showRelatedQuestion: boolean
  isLoading?: boolean
  isLastAssistantMessage?: boolean
}) {
  const {
    onRegenerateResponse,
    onSubmitSearch,
    setDevPanelOpen,
    setConversationIdForDev
  } = useContext(SearchContext)
  const [enableDeveloperMode] = useEnableDeveloperMode()

  const [showMoreSource, setShowMoreSource] = useState(false)
  const [relevantCodeHighlightIndex, setRelevantCodeHighlightIndex] = useState<
    number | undefined
  >(undefined)
  const getCopyContent = (answer: ConversationMessage) => {
    // FIXME copy code
    if (isEmpty(answer?.attachment?.doc)) return answer.content

    const citationMatchRegex = /\[\[?citation:\s*\d+\]?\]/g
    const content = answer.content
      .replace(citationMatchRegex, (match, p1) => {
        const citationNumberMatch = match?.match(/\d+/)
        return `[${citationNumberMatch}]`
      })
      .trim()
    const citations = answer.attachment?.doc
      .map((relevent, idx) => `[${idx + 1}] ${relevent.link}`)
      .join('\n')
    return `${content}\n\nCitations:\n${citations}`
  }

  const IconAnswer = isLoading ? IconSpinner : IconSparkles

  const totalHeightInRem = answer.attachment?.doc?.length
    ? Math.ceil(answer.attachment.doc.length / 4) * SOURCE_CARD_STYLE.expand +
      0.5 * Math.floor(answer.attachment.doc.length / 4)
    : 0

  const relevantCodeContexts: RelevantCodeContext[] = useMemo(() => {
    return (
      answer?.attachment?.code?.map(code => {
        const start_line = code?.startLine ?? 0
        const lineCount = code.content.split('\n').length
        const end_line = start_line + lineCount - 1

        return {
          kind: 'file',
          range: {
            start: start_line,
            end: end_line
          },
          filepath: code.filepath,
          content: code.content,
          git_url: code.gitUrl,
          extra: {
            scores: code?.extra?.scores
          }
        }
      }) ?? []
    )
  }, [answer?.attachment?.code])

  const onCodeContextClick = (ctx: Context) => {
    if (!ctx.filepath) return
    const url = new URL(`${window.location.origin}/files`)
    const searchParams = new URLSearchParams()
    searchParams.append('redirect_filepath', ctx.filepath)
    searchParams.append('redirect_git_url', ctx.git_url)
    url.search = searchParams.toString()

    const lineHash = formatLineHashForCodeBrowser({
      start: ctx.range.start,
      end: ctx.range.end
    })
    if (lineHash) {
      url.hash = lineHash
    }

    window.open(url.toString())
  }

  const onCodeCitationMouseEnter = (index: number) => {
    setRelevantCodeHighlightIndex(
      index - 1 - (answer?.attachment?.doc?.length || 0)
    )
  }

  const onCodeCitationMouseLeave = (index: number) => {
    setRelevantCodeHighlightIndex(undefined)
  }

  const onCodeCitationClick = (code: MessageAttachmentCode) => {
    const start_line = code?.startLine ?? 0
    const lineCount = code.content.split('\n').length
    const end_line = start_line + lineCount - 1
    // FIXME utils
    if (!code.filepath) return
    const url = new URL(`${window.location.origin}/files`)
    const searchParams = new URLSearchParams()
    searchParams.append('redirect_filepath', code.filepath)
    searchParams.append('redirect_git_url', code.gitUrl)
    url.search = searchParams.toString()

    const lineHash = formatLineHashForCodeBrowser({
      start: start_line,
      end: end_line
    })
    if (lineHash) {
      url.hash = lineHash
    }

    window.open(url.toString())
  }

  const messageAttachmentDocs = answer?.attachment?.doc
  const messageAttachmentCode = answer?.attachment?.code

  return (
    <AnswerBlockContext.Provider
      value={{
        onCodeCitationClick,
        onCodeCitationMouseEnter,
        onCodeCitationMouseLeave
      }}
    >
      <div className="flex flex-col gap-y-5">
        {/* document search hits */}
        {messageAttachmentDocs && messageAttachmentDocs.length > 0 && (
          <div>
            <div className="mb-1 flex items-center gap-x-2">
              <IconBlocks className="relative" style={{ top: '-0.04rem' }} />
              <p className="text-sm font-bold leading-normal">Sources</p>
            </div>
            <div
              className="gap-sm grid grid-cols-3 gap-2 overflow-hidden md:grid-cols-4"
              style={{
                transition: 'height 0.25s ease-out',
                height: showMoreSource
                  ? `${totalHeightInRem}rem`
                  : `${SOURCE_CARD_STYLE.compress}rem`
              }}
            >
              {messageAttachmentDocs.map((source, index) => (
                <SourceCard
                  key={source.link + index}
                  conversationId={answer.id}
                  source={source}
                  showMore={showMoreSource}
                  showDevTooltip={enableDeveloperMode.value}
                />
              ))}
            </div>
            <Button
              variant="ghost"
              className="-ml-1.5 mt-1 flex items-center gap-x-1 px-1 py-2 text-sm font-normal text-muted-foreground"
              onClick={() => setShowMoreSource(!showMoreSource)}
            >
              <IconChevronRight
                className={cn({
                  '-rotate-90': showMoreSource,
                  'rotate-90': !showMoreSource
                })}
              />
              <p>{showMoreSource ? 'Show less' : 'Show more'}</p>
            </Button>
          </div>
        )}

        {/* Answer content */}
        <div>
          <div className="mb-1 flex items-center gap-x-1.5">
            <IconAnswer
              className={cn({
                'animate-spinner': isLoading
              })}
            />
            <p className="text-sm font-bold leading-none">Answer</p>
            {enableDeveloperMode.value && (
              <Button
                variant="ghost"
                size="icon"
                onClick={() => {
                  setConversationIdForDev(answer.id)
                  setDevPanelOpen(true)
                }}
              >
                <IconBug />
              </Button>
            )}
          </div>

          {/* code search hits */}
          {messageAttachmentCode && messageAttachmentCode.length > 0 && (
            <CodeReferences
              contexts={relevantCodeContexts}
              className="mt-1 text-sm"
              onContextClick={onCodeContextClick}
              defaultOpen
              enableTooltip={enableDeveloperMode.value}
              onTooltipClick={() => {
                setConversationIdForDev(answer.id)
                setDevPanelOpen(true)
              }}
              highlightIndex={relevantCodeHighlightIndex}
            />
          )}

          {isLoading && !answer.content && (
            <Skeleton className="mt-1 h-40 w-full" />
          )}
          <MessageMarkdown
            message={answer.content}
            relevantDocuments={messageAttachmentDocs}
            relevantCode={messageAttachmentCode}
          />
          {answer.error && <ErrorMessageBlock error={answer.error} />}

          {!isLoading && (
            <div className="mt-3 flex items-center gap-x-3 text-sm">
              <CopyButton
                className="-ml-1.5 gap-x-1 px-1 font-normal text-muted-foreground"
                value={getCopyContent(answer)}
                text="Copy"
              />
              {/* FIXME hide regenerate for now */}
              {/* {!isLoading && isLastAssistantMessage && (
                <Button
                  className="flex items-center gap-x-1 px-1 font-normal text-muted-foreground"
                  variant="ghost"
                  onClick={() => onRegenerateResponse(answer.id)}
                >
                  <IconRefresh />
                  <p>Regenerate</p>
                </Button>
              )} */}
            </div>
          )}
        </div>

        {/* Related questions */}
        {showRelatedQuestion &&
          !isLoading &&
          answer.threadRelevantQuestions &&
          answer.threadRelevantQuestions.length > 0 && (
            <div>
              <div className="flex items-center gap-x-1.5">
                <IconLayers />
                <p className="text-sm font-bold leading-none">Suggestions</p>
              </div>
              <div className="mt-2 flex flex-col gap-y-3">
                {answer.threadRelevantQuestions?.map((related, index) => (
                  <div
                    key={index}
                    className="flex cursor-pointer items-center justify-between rounded-lg border p-4 py-3 transition-opacity hover:opacity-70"
                    onClick={onSubmitSearch.bind(null, related)}
                  >
                    <p className="w-full overflow-hidden text-ellipsis text-sm">
                      {related}
                    </p>
                    <IconPlus />
                  </div>
                ))}
              </div>
            </div>
          )}
      </div>
    </AnswerBlockContext.Provider>
  )
}

// Remove HTML and Markdown format
const normalizedText = (input: string) => {
  const sanitizedHtml = DOMPurify.sanitize(input, {
    ALLOWED_TAGS: [],
    ALLOWED_ATTR: []
  })
  const parsed = marked.parse(sanitizedHtml) as string
  const decoded = he.decode(parsed)
  const plainText = decoded.replace(/<\/?[^>]+(>|$)/g, '')
  return plainText
}

function SourceCard({
  conversationId,
  source,
  showMore,
  showDevTooltip
}: {
  conversationId: string
  source: AttachmentDocItem
  showMore: boolean
  showDevTooltip?: boolean
}) {
  const { setDevPanelOpen, setConversationIdForDev } = useContext(SearchContext)
  const { hostname } = new URL(source.link)
  const [devTooltipOpen, setDevTooltipOpen] = useState(false)

  const onOpenChange = (v: boolean) => {
    if (!showDevTooltip) return
    setDevTooltipOpen(v)
  }

  const onTootipClick: MouseEventHandler<HTMLDivElement> = e => {
    e.stopPropagation()
    setConversationIdForDev(conversationId)
    setDevPanelOpen(true)
  }

  return (
    <Tooltip
      open={devTooltipOpen}
      onOpenChange={onOpenChange}
      delayDuration={0}
    >
      <TooltipTrigger asChild>
        <div
          className="flex cursor-pointer flex-col justify-between rounded-lg border bg-card p-3 hover:bg-card/60"
          style={{
            height: showMore
              ? `${SOURCE_CARD_STYLE.expand}rem`
              : `${SOURCE_CARD_STYLE.compress}rem`,
            transition: 'all 0.25s ease-out'
          }}
          onClick={() => window.open(source.link)}
        >
          <div className="flex flex-1 flex-col justify-between gap-y-1">
            <div className="flex flex-col gap-y-0.5">
              <p className="line-clamp-1 w-full overflow-hidden text-ellipsis break-all text-xs font-semibold">
                {source.title}
              </p>
              <p
                className={cn(
                  ' w-full overflow-hidden text-ellipsis break-all text-xs text-muted-foreground',
                  {
                    'line-clamp-2': showMore,
                    'line-clamp-1': !showMore
                  }
                )}
              >
                {normalizedText(source.content)}
              </p>
            </div>
            <div className="flex items-center text-xs text-muted-foreground">
              <div className="flex w-full flex-1 items-center">
                <SiteFavicon hostname={hostname} />
                <p className="ml-1 overflow-hidden text-ellipsis">
                  {hostname.replace('www.', '').split('/')[0]}
                </p>
              </div>
            </div>
          </div>
        </div>
      </TooltipTrigger>
      <TooltipContent
        align="start"
        className="cursor-pointer p-2"
        onClick={onTootipClick}
      >
        <p>Score: {source?.extra?.score ?? '-'}</p>
      </TooltipContent>
    </Tooltip>
  )
}

type RelevantDocItem = {
  type: 'doc'
  data: AttachmentDocItem
}

type RelevantCodeItem = {
  type: 'code'
  data: AttachmentCodeItem
}

type MessageAttachments = Array<RelevantDocItem | RelevantCodeItem>

function MessageMarkdown({
  message,
  headline = false,
  relevantDocuments,
  relevantCode
}: {
  message: string
  headline?: boolean
  relevantDocuments?: Array<AttachmentDocItem>
  relevantCode?: Array<AttachmentCodeItem>
}) {
  const messageAttachments: MessageAttachments = useMemo(() => {
    const docs: MessageAttachments =
      relevantDocuments?.map(item => ({
        type: 'doc',
        data: item
      })) ?? []
    const code: MessageAttachments =
      relevantCode?.map(item => ({
        type: 'code',
        data: item
      })) ?? []
    return compact([...docs, ...code])
  }, [relevantDocuments, relevantCode])

  const renderTextWithCitation = (nodeStr: string, index: number) => {
    const citationMatchRegex = /\[\[?citation:\s*\d+\]?\]/g
    const textList = nodeStr.split(citationMatchRegex)
    const citationList = nodeStr.match(citationMatchRegex)
    return (
      <span key={index}>
        {textList.map((text, index) => {
          const citation = citationList?.[index]
          const citationNumberMatch = citation?.match(/\d+/)
          const citationIndex = citationNumberMatch
            ? parseInt(citationNumberMatch[0], 10)
            : null
          const citationSource = !isNil(citationIndex)
            ? messageAttachments?.[citationIndex - 1]
            : undefined
          const citationType = citationSource?.type
          const showcitation = citationSource && !isNil(citationIndex)

          return (
            <span key={index}>
              {text && <span>{text}</span>}
              {showcitation && (
                <>
                  {citationType === 'doc' ? (
                    <RelevantDocumentBadge
                      relevantDocument={citationSource.data}
                      citationIndex={citationIndex}
                    />
                  ) : citationType === 'code' ? (
                    <RelevantCodeBadge
                      relevantCode={citationSource.data}
                      citationIndex={citationIndex}
                    />
                  ) : null}
                </>
              )}
            </span>
          )
        })}
      </span>
    )
  }

  return (
    <MemoizedReactMarkdown
      className="prose max-w-none break-words dark:prose-invert prose-p:leading-relaxed prose-pre:mt-1 prose-pre:p-0"
      remarkPlugins={[remarkGfm, remarkMath]}
      components={{
        p({ children }) {
          if (headline) {
            return (
              <h3 className="break-anywhere cursor-text scroll-m-20 text-xl font-semibold tracking-tight">
                {children}
              </h3>
            )
          }

          if (children.length) {
            return (
              <div className="mb-2 inline-block leading-relaxed last:mb-0">
                {children.map((childrenItem, index) => {
                  if (typeof childrenItem === 'string') {
                    return renderTextWithCitation(childrenItem, index)
                  }

                  return <span key={index}>{childrenItem}</span>
                })}
              </div>
            )
          }

          return <p className="mb-2 last:mb-0">{children}</p>
        },
        li({ children }) {
          if (children && children.length) {
            return (
              <li>
                {children.map((childrenItem, index) => {
                  if (typeof childrenItem === 'string') {
                    return renderTextWithCitation(childrenItem, index)
                  }

                  return <span key={index}>{childrenItem}</span>
                })}
              </li>
            )
          }
          return <li>{children}</li>
        },
        code({ node, inline, className, children, ...props }) {
          if (children.length) {
            if (children[0] == '▍') {
              return (
                <span className="mt-1 animate-pulse cursor-default">▍</span>
              )
            }

            children[0] = (children[0] as string).replace('`▍`', '▍')
          }

          const match = /language-(\w+)/.exec(className || '')

          if (inline) {
            return (
              <code className={className} {...props}>
                {children}
              </code>
            )
          }

          return (
            <CodeBlock
              key={Math.random()}
              language={(match && match[1]) || ''}
              value={String(children).replace(/\n$/, '')}
              {...props}
            />
          )
        }
      }}
    >
      {message}
    </MemoizedReactMarkdown>
  )
}

function RelevantDocumentBadge({
  relevantDocument,
  citationIndex
}: {
  relevantDocument: MessageAttachmentDoc
  citationIndex: number
}) {
  const sourceUrl = relevantDocument ? new URL(relevantDocument.link) : null

  return (
    <HoverCard>
      <HoverCardTrigger>
        <span
          className="relative -top-2 mr-0.5 inline-block h-4 w-4 cursor-pointer rounded-full bg-muted text-center text-xs"
          onClick={() => window.open(relevantDocument.link)}
        >
          {citationIndex}
        </span>
      </HoverCardTrigger>
      <HoverCardContent className="w-96 text-sm">
        <div className="flex w-full flex-col gap-y-1">
          <div className="m-0 flex items-center space-x-1 text-xs leading-none text-muted-foreground">
            <SiteFavicon
              hostname={sourceUrl!.hostname}
              className="m-0 mr-1 leading-none"
            />
            <p className="m-0 leading-none">{sourceUrl!.hostname}</p>
          </div>
          <p
            className="m-0 cursor-pointer font-bold leading-none transition-opacity hover:opacity-70"
            onClick={() => window.open(relevantDocument.link)}
          >
            {relevantDocument.title}
          </p>
          <p className="m-0 line-clamp-4 leading-none">
            {normalizedText(relevantDocument.content)}
          </p>
        </div>
      </HoverCardContent>
    </HoverCard>
  )
}

function RelevantCodeBadge({
  relevantCode,
  citationIndex
}: {
  relevantCode: MessageAttachmentCode
  citationIndex: number
}) {
  const {
    onCodeCitationClick,
    onCodeCitationMouseEnter,
    onCodeCitationMouseLeave
  } = useContext(AnswerBlockContext)

  return (
    <span
      className="relative -top-2 mr-0.5 inline-block h-4 w-4 cursor-pointer rounded-full bg-muted text-center text-xs"
      onClick={() => {
        onCodeCitationClick?.(relevantCode)
      }}
      onMouseEnter={() => {
        onCodeCitationMouseEnter?.(citationIndex)
      }}
      onMouseLeave={() => {
        onCodeCitationMouseLeave?.(citationIndex)
      }}
    >
      {citationIndex}
    </span>
  )
}

function SiteFavicon({
  hostname,
  className
}: {
  hostname: string
  className?: string
}) {
  const [isLoaded, setIsLoaded] = useState(false)

  const handleImageLoad = () => {
    setIsLoaded(true)
  }

  return (
    <div className="relative h-3.5 w-3.5">
      <Image
        src={defaultFavicon}
        alt={hostname}
        width={14}
        height={14}
        className={cn(
          'absolute left-0 top-0 z-0 h-3.5 w-3.5 rounded-full leading-none',
          className
        )}
      />
      <Image
        src={`https://s2.googleusercontent.com/s2/favicons?sz=128&domain_url=${hostname}`}
        alt={hostname}
        width={14}
        height={14}
        className={cn(
          'relative z-10 h-3.5 w-3.5 rounded-full bg-card leading-none',
          className,
          {
            'opacity-0': !isLoaded
          }
        )}
        onLoad={handleImageLoad}
      />
    </div>
  )
}

function ErrorMessageBlock({ error = 'Fail to fetch' }: { error?: string }) {
  const errorMessage = useMemo(() => {
    let jsonString = JSON.stringify(
      {
        error: true,
        message: error
      },
      null,
      2
    )
    const markdownJson = '```\n' + jsonString + '\n```'
    return markdownJson
  }, [error])
  return (
    <MemoizedReactMarkdown
      className="prose-full-width prose break-words text-sm dark:prose-invert prose-p:leading-relaxed prose-pre:mt-1 prose-pre:p-0"
      remarkPlugins={[remarkGfm, remarkMath]}
      components={{
        code({ node, inline, className, children, ...props }) {
          return (
            <div {...props} className={cn(className, 'bg-zinc-950 p-2')}>
              {children}
            </div>
          )
        }
      }}
    >
      {errorMessage}
    </MemoizedReactMarkdown>
  )
}
