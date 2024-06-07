'use client'

import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState
} from 'react'
import Image from 'next/image'
import Link from 'next/link'
import logoDarkUrl from '@/assets/logo-dark.png'
import logoUrl from '@/assets/logo.png'
import tabbyUrl from '@/assets/tabby.png'
import { Message } from 'ai'
import { nanoid } from 'nanoid'
import { useTheme } from 'next-themes'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'

import { SESSION_STORAGE_KEY } from '@/lib/constants'
import { useEnableSearch } from '@/lib/experiment-flags'
import { useIsChatEnabled } from '@/lib/hooks/use-server-info'
import { useTabbyAnswer } from '@/lib/hooks/use-tabby-answer'
import fetcher from '@/lib/tabby/fetcher'
import { AnswerRequest } from '@/lib/types'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { CodeBlock } from '@/components/ui/codeblock'
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger
} from '@/components/ui/hover-card'
import {
  IconBlocks,
  IconChevronRight,
  IconLayers,
  IconPlus,
  IconRefresh,
  IconSparkles,
  IconStop
} from '@/components/ui/icons'
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

interface Source {
  title: string
  link: string
  snippet: string
}

type ConversationMessage = Message & {
  relevant_documents?: {
    title: string
    link: string
    snippet: string
  }[]
  relevant_questions?: string[]
  isLoading?: boolean
  error?: string
}

type SearchContextValue = {
  isLoading: boolean
  onRegenerateResponse: (id: string) => void
  onSubmitSearch: (question: string) => void
}

export const SearchContext = createContext<SearchContextValue>(
  {} as SearchContextValue
)

const tabbyFetcher = ((url: string, init?: RequestInit) => {
  return fetcher(url, {
    ...init,
    responseFormatter(response) {
      return response
    },
    errorHandler(response) {
      throw new Error(response ? String(response.status) : 'Fail to fetch')
    }
  })
}) as typeof fetch

const SOURCE_CARD_STYLE = {
  compress: 5,
  expand: 7
}

export function Search() {
  const isChatEnabled = useIsChatEnabled()
  const [searchFlag] = useEnableSearch()
  const [isShowDemoBanner] = useShowDemoBanner()
  const { theme } = useTheme()
  const [conversation, setConversation] = useState<ConversationMessage[]>([])
  const [showStop, setShowStop] = useState(false)
  const [container, setContainer] = useState<HTMLDivElement | null>(null)
  const [title, setTitle] = useState('')
  const [currentLoadindId, setCurrentLoadingId] = useState<string>('')
  const contentContainerRef = useRef<HTMLDivElement>(null)

  const { triggerRequest, isLoading, error, answer, stop } = useTabbyAnswer({
    fetcher: tabbyFetcher
  })

  useEffect(() => {
    const initialQuestion = sessionStorage.getItem(
      SESSION_STORAGE_KEY.SEARCH_INITIAL_MSG
    )
    if (initialQuestion) {
      onSubmitSearch(initialQuestion)
      sessionStorage.removeItem(SESSION_STORAGE_KEY.SEARCH_INITIAL_MSG)
    }
  }, [])

  useEffect(() => {
    if (title) document.title = title
  }, [title])

  useEffect(() => {
    setContainer(
      contentContainerRef?.current?.children[1] as HTMLDivElement | null
    )
  }, [contentContainerRef?.current])

  // Handling the stream response from useTabbyAnswer
  useEffect(() => {
    if (!answer) return
    const newConversation = [...conversation]
    const currentAnswer = newConversation.find(
      item => item.id === currentLoadindId
    )
    if (!currentAnswer) return
    currentAnswer.content = answer.answer_delta || ''
    currentAnswer.relevant_documents = answer.relevant_documents
    currentAnswer.relevant_questions = answer.relevant_questions
    currentAnswer.isLoading = isLoading
    setConversation(newConversation)
  }, [isLoading, answer])

  // Handling the error response from useTabbyAnswer
  useEffect(() => {
    if (error) {
      const newConversation = [...conversation]
      const currentAnswer = newConversation.find(
        item => item.id === currentLoadindId
      )
      if (currentAnswer) {
        currentAnswer.error =
          error.message === '401' ? 'Unauthorized' : 'Fail to fetch'
        currentAnswer.isLoading = false
      }
    }
  }, [error])

  // Delay showing the stop button
  useEffect(() => {
    if (isLoading && !showStop) {
      setTimeout(() => {
        setShowStop(true)

        // Scroll to the bottom
        if (container) {
          const isLastAnswerLoading =
            currentLoadindId === conversation[conversation.length - 1].id
          if (isLastAnswerLoading) {
            container.scrollTo({
              top: container.scrollHeight,
              behavior: 'smooth'
            })
          }
        }
      }, 1500)
    }

    if (!isLoading && showStop) {
      setShowStop(false)
    }
  }, [isLoading])

  const onSubmitSearch = (question: string) => {
    // FIXME: code query? extra from user's input?
    const previousMessages = conversation.map(message => ({
      role: message.role,
      id: message.id,
      content: message.content
    }))
    const previousUserId = previousMessages.length > 0 && previousMessages[0].id
    const newAssistantId = nanoid()
    const newUserMessage: ConversationMessage = {
      id: previousUserId || nanoid(),
      role: 'user',
      content: question
    }
    const newAssistantMessage: ConversationMessage = {
      id: newAssistantId,
      role: 'assistant',
      content: '',
      isLoading: true
    }

    const answerRequest: AnswerRequest = {
      messages: [...previousMessages, newUserMessage],
      doc_query: true,
      generate_relevant_questions: true
    }

    setCurrentLoadingId(newAssistantId)
    setConversation(
      [...conversation].concat([newUserMessage, newAssistantMessage])
    )
    triggerRequest(answerRequest)

    // Update HTML page title
    if (!title) setTitle(question)
  }

  const onRegenerateResponse = (id: string) => {
    const targetAnswerIdx = conversation.findIndex(item => item.id === id)
    if (targetAnswerIdx < 1) return
    const targetQuestionIdx = targetAnswerIdx - 1
    const targetQuestion = conversation[targetQuestionIdx]

    const previousMessages = conversation
      .slice(0, targetQuestionIdx)
      .map(message => ({
        role: message.role,
        id: message.id,
        content: message.content
      }))
    const newUserMessage = {
      role: 'user',
      id: targetQuestion.id,
      content: targetQuestion.content
    }
    const answerRequest: AnswerRequest = {
      messages: [...previousMessages, newUserMessage],
      doc_query: true,
      generate_relevant_questions: true
    }

    const newConversation = [...conversation]
    let newTargetAnswer = newConversation[targetAnswerIdx]
    newTargetAnswer.content = ''
    newTargetAnswer.error = undefined
    newTargetAnswer.isLoading = true

    setCurrentLoadingId(newTargetAnswer.id)
    setConversation(newConversation)
    triggerRequest(answerRequest)
  }

  if (!searchFlag.value || !isChatEnabled) {
    return <></>
  }

  const noConversation = conversation.length === 0
  const style = isShowDemoBanner
    ? { height: `calc(100vh - ${BANNER_HEIGHT})` }
    : { height: '100vh' }
  return (
    <SearchContext.Provider
      value={{
        isLoading: isLoading,
        onRegenerateResponse: onRegenerateResponse,
        onSubmitSearch: onSubmitSearch
      }}
    >
      <div className="flex flex-col transition-all" style={style}>
        <div className="flex w-full items-center justify-between border-b px-10 py-3">
          <Link href="/">
            <Image
              src={theme === 'dark' ? logoDarkUrl : logoUrl}
              alt="logo"
              width={80}
            />
          </Link>

          <div className="flex items-center justify-center gap-4">
            <ClientOnly>
              <ThemeToggle />
            </ClientOnly>
            <UserPanel>
              <UserAvatar className="h-8 w-8 border" />
            </UserPanel>
          </div>
        </div>
        <ScrollArea className="flex-1" ref={contentContainerRef}>
          <div className="mx-auto px-10 pb-24 lg:max-w-4xl lg:px-0">
            <div className="flex flex-col">
              {conversation.map((item, idx) => {
                if (item.role === 'user') {
                  return (
                    <div key={item.id + idx}>
                      {idx !== 0 && <Separator />}
                      <div className="pb-2 pt-8">
                        <MessageMarkdown message={item.content} headline />
                      </div>
                    </div>
                  )
                }
                if (item.role === 'assistant') {
                  return (
                    <div key={item.id + idx} className="pb-8 pt-2">
                      <AnswerBlock
                        question="todo"
                        answer={item}
                        showRelatedQuestion={idx === conversation.length - 1}
                      />
                    </div>
                  )
                }
                return <></>
              })}
            </div>
          </div>
        </ScrollArea>

        {container && (
          <ButtonScrollToBottom
            className="!fixed !bottom-[7rem] !right-10 !top-auto lg:!bottom-[3.8rem]"
            container={container}
            offset={100}
          />
        )}

        <div
          className={cn(
            'fixed left-0 flex w-full flex-col items-center transition-all',
            {
              'bottom-1/2 -mt-48': noConversation,
              'bottom-5 min-h-[5rem]': !noConversation
            }
          )}
        >
          {noConversation && (
            <>
              <Image src={tabbyUrl} alt="logo" width={42} />
              <h4 className="mb-6 scroll-m-20 text-xl font-semibold tracking-tight text-secondary-foreground">
                The Private Search Assistant
              </h4>
            </>
          )}
          {!isLoading && (
            <div className="relative z-20 flex justify-center self-stretch px-10">
              <TextAreaSearch
                className={cn({
                  'lg:max-w-2xl': noConversation,
                  'lg:max-w-4xl': !noConversation
                })}
                onSearch={onSubmitSearch}
                placeholder={
                  (!noConversation && 'Ask a follow up question') || undefined
                }
              />
            </div>
          )}
          <Button
            className={cn(
              'absolute top-2 z-0 flex items-center gap-x-2 px-8 py-4',
              {
                'opacity-0 pointer-events-none': !showStop,
                'opacity-100': showStop
              }
            )}
            style={{
              transition: 'opacity 0.55s ease-out'
            }}
            variant="destructive"
            onClick={stop}
          >
            <IconStop />
            <p>Stop</p>
          </Button>
        </div>
      </div>
    </SearchContext.Provider>
  )
}

function AnswerBlock({
  question,
  answer,
  showRelatedQuestion
}: {
  question: string
  answer: ConversationMessage
  showRelatedQuestion: boolean
}) {
  const { onRegenerateResponse, onSubmitSearch, isLoading } =
    useContext(SearchContext)
  const [showMore, setShowMore] = useState(false)

  const getCopyContent = (answer: ConversationMessage) => {
    if (!answer.relevant_documents) return answer.content

    const citationMatchRegex = /\[\[?citation:\s*\d+\]?\]/g
    const content = answer.content
      .replace(citationMatchRegex, (match, p1) => {
        const citationNumberMatch = match?.match(/\d+/)
        return `[${citationNumberMatch}]`
      })
      .trim()
    const citations = answer.relevant_documents
      .map((relevent, idx) => `[${idx + 1}] ${relevent.link}`)
      .join('\n')
    return `${content}\n\nCitations:\n${citations}`
  }

  const totalHeightInRem = answer.relevant_documents
    ? Math.ceil(answer.relevant_documents.length / 3) *
        SOURCE_CARD_STYLE.expand +
      0.5 * Math.floor(answer.relevant_documents.length / 3)
    : 0
  return (
    <div className="flex flex-col gap-y-5">
      {/* Relevant documents */}
      {answer.relevant_documents && answer.relevant_documents.length > 0 && (
        <div>
          <div className="mb-1 flex items-center gap-x-2">
            <IconBlocks className="relative" style={{ top: '-0.04rem' }} />
            <p className="text-sm font-bold leading-normal">Source</p>
          </div>
          <div
            className="gap-sm grid grid-cols-3 gap-2 overflow-hidden"
            style={{
              transition: 'height 0.25s ease-out',
              height: showMore
                ? `${totalHeightInRem}rem`
                : `${SOURCE_CARD_STYLE.compress}rem`
            }}
          >
            {answer.relevant_documents.map((source, index) => (
              <SourceCard
                key={source.link}
                source={source}
                showMore={showMore}
                index={index + 1}
              />
            ))}
          </div>
          <Button
            variant="ghost"
            className="-ml-1.5 mt-1 flex items-center gap-x-1 px-1 py-2 text-sm font-normal text-muted-foreground"
            onClick={() => setShowMore(!showMore)}
          >
            <IconChevronRight
              className={cn({
                '-rotate-90': showMore,
                'rotate-90': !showMore
              })}
            />
            <p>{showMore ? 'Show less' : 'Show more'}</p>
          </Button>
        </div>
      )}

      {/* Answer content */}
      <div>
        <div className="flex items-center gap-x-1.5">
          <IconSparkles
            className={cn({
              'sparkle-animation': answer.isLoading
            })}
          />
          <p className="text-sm font-bold leading-none">Answer</p>
        </div>
        {answer.isLoading && !answer.content && (
          <Skeleton className="mt-1 h-40 w-full" />
        )}

        <MessageMarkdown
          message={answer.content}
          sources={answer.relevant_documents}
        />
        {answer.error && <ErrorMessageBlock error={answer.error} />}

        {!answer.isLoading && (
          <div className="mt-3 flex items-center gap-x-3 text-sm">
            <CopyButton
              className="-ml-1.5 gap-x-1 px-1 font-normal text-muted-foreground"
              value={getCopyContent(answer)}
              text="Copy"
            />
            {!isLoading && (
              <Button
                className="flex items-center gap-x-1 px-1 font-normal text-muted-foreground"
                variant="ghost"
                onClick={onRegenerateResponse.bind(null, answer.id)}
              >
                <IconRefresh />
                <p>Regenerate</p>
              </Button>
            )}
          </div>
        )}
      </div>

      {/* Related questions */}
      {showRelatedQuestion &&
        !answer.isLoading &&
        answer.relevant_questions &&
        answer.relevant_questions.length > 0 && (
          <div>
            <div className="flex items-center gap-x-1.5">
              <IconLayers />
              <p className="text-sm font-bold leading-none">Related</p>
            </div>
            <div className="mt-3 flex flex-col gap-y-3">
              {answer.relevant_questions?.map((related, index) => (
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
  )
}

function SourceCard({
  source,
  index,
  showMore
}: {
  source: Source
  index: number
  showMore: boolean
}) {
  const { hostname } = new URL(source.link)
  return (
    <div
      className="flex cursor-pointer flex-col justify-between gap-y-1 rounded-lg border bg-card px-4 py-2 transition-all hover:bg-card/60"
      style={{
        height: showMore
          ? `${SOURCE_CARD_STYLE.expand}rem`
          : `${SOURCE_CARD_STYLE.compress}rem`
      }}
      onClick={() => window.open(source.link)}
    >
      <p className="line-clamp-2 w-full overflow-hidden text-ellipsis break-all text-xs font-semibold">
        {source.title}
      </p>
      {showMore && (
        <p className="line-clamp-2 w-full overflow-hidden text-ellipsis break-all text-xs text-muted-foreground">
          {source.snippet}
        </p>
      )}
      <div className="flex items-center">
        <SiteFavicon hostname={hostname} className="mr-1" />

        <div className="flex items-center gap-x-0.5 text-xs text-muted-foreground">
          <p className="flex-1 overflow-hidden text-ellipsis">
            {hostname.replace('www.', '').split('.')[0]}
          </p>
          <span className="relative -top-1.5 text-xl leading-none">.</span>
          <p>{index}</p>
        </div>
      </div>
    </div>
  )
}

function MessageMarkdown({
  message,
  headline = false,
  sources
}: {
  message: string
  headline?: boolean
  sources?: Source[]
}) {
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
                    const citationMatchRegex = /\[\[?citation:\s*\d+\]?\]/g
                    const textList = childrenItem.split(citationMatchRegex)
                    const citationList = childrenItem.match(citationMatchRegex)
                    return (
                      <span key={index}>
                        {textList.map((text, index) => {
                          const citation = citationList?.[index]
                          const citationNumberMatch = citation?.match(/\d+/)
                          const citationIndex = citationNumberMatch
                            ? parseInt(citationNumberMatch[0], 10)
                            : null
                          const source =
                            citationIndex !== null
                              ? sources?.[citationIndex - 1]
                              : null
                          const sourceUrl = source ? new URL(source.link) : null
                          return (
                            <span key={index}>
                              {text && <span>{text}</span>}
                              {source && (
                                <HoverCard>
                                  <HoverCardTrigger>
                                    <span
                                      className="relative -top-2 inline-block h-4 w-4 cursor-pointer rounded-full bg-muted text-center text-xs"
                                      onClick={() => window.open(source.link)}
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
                                        <p className="m-0 leading-none">
                                          {sourceUrl!.hostname}
                                        </p>
                                      </div>
                                      <p
                                        className="m-0 cursor-pointer font-bold leading-none transition-opacity hover:opacity-70"
                                        onClick={() => window.open(source.link)}
                                      >
                                        {source.title}
                                      </p>
                                      <p className="m-0 leading-none">
                                        {source.snippet}
                                      </p>
                                    </div>
                                  </HoverCardContent>
                                </HoverCard>
                              )}
                            </span>
                          )
                        })}
                      </span>
                    )
                  }

                  return <span key={index}>{childrenItem}</span>
                })}
              </div>
            )
          }

          return <p className="mb-2 last:mb-0">{children}</p>
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

function SiteFavicon({
  hostname,
  className
}: {
  hostname: string
  className?: string
}) {
  return (
    <img
      src={`https://s2.googleusercontent.com/s2/favicons?sz=128&domain_url=${hostname}`}
      alt={hostname}
      className={cn('h-3.5 w-3.5 rounded-full leading-none', className)}
    />
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
