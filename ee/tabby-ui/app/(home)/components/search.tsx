'use client'

import {
  createContext,
  ForwardedRef,
  forwardRef,
  useContext,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState
} from 'react'
import { Message } from 'ai'
import { nanoid } from 'nanoid'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'

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
  IconSpinner,
  IconStop
} from '@/components/ui/icons'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { Skeleton } from '@/components/ui/skeleton'
import { ButtonScrollToBottom } from '@/components/button-scroll-to-bottom'
import { CopyButton } from '@/components/copy-button'
import { MemoizedReactMarkdown } from '@/components/markdown'
import TextAreaSearch from '@/components/textarea-search'

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

type SubmitSearchOpts = {
  isNew?: boolean
}

type SearchContextValue = {
  isLoading: boolean
  onRegenerateResponse: (id: string) => void
  onSubmitSearch: (question: string, opts?: SubmitSearchOpts) => void
}

export interface SearchRef {
  onSubmitSearch: (question: string, opts?: SubmitSearchOpts) => void
  stop: () => void
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

export function SearchRenderer({}, ref: ForwardedRef<SearchRef>) {
  const isChatEnabled = useIsChatEnabled()
  const [searchFlag] = useEnableSearch()
  const [conversation, setConversation] = useState<ConversationMessage[]>([])
  const [showStop, setShowStop] = useState(true)
  const [container, setContainer] = useState<HTMLDivElement | null>(null)
  const [title, setTitle] = useState('')
  const [currentLoadindId, setCurrentLoadingId] = useState<string>('')
  const contentContainerRef = useRef<HTMLDivElement>(null)
  const [showSearchInput, setShowSearchInput] = useState(false)

  const { triggerRequest, isLoading, error, answer, stop } = useTabbyAnswer({
    fetcher: tabbyFetcher
  })

  useImperativeHandle(
    ref,
    () => {
      return {
        onSubmitSearch,
        stop
      }
    },
    []
  )

  useEffect(() => {
    if (title) document.title = title
  }, [title])

  useEffect(() => {
    setTimeout(() => {
      setShowSearchInput(true)
    }, 500)
  }, [])

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
    if (isLoading) {
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

    if (!isLoading) {
      setShowStop(false)
    }
  }, [isLoading])

  const onSubmitSearch = (question: string, opts?: SubmitSearchOpts) => {
    // FIXME: code query? extra from user's input?
    const previousMessages = opts?.isNew
      ? []
      : conversation.map(message => ({
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
    const previousConversation = opts?.isNew ? [] : [...conversation]
    setConversation(
      previousConversation.concat([newUserMessage, newAssistantMessage])
    )
    triggerRequest(answerRequest)

    // Update HTML page title
    if (!title || opts?.isNew) setTitle(question)
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
    newTargetAnswer.relevant_documents = undefined
    newTargetAnswer.error = undefined
    newTargetAnswer.isLoading = true

    setCurrentLoadingId(newTargetAnswer.id)
    setConversation(newConversation)
    triggerRequest(answerRequest)
  }

  if (!searchFlag.value || !isChatEnabled) {
    return <></>
  }

  return (
    <SearchContext.Provider
      value={{
        isLoading: isLoading,
        onRegenerateResponse: onRegenerateResponse,
        onSubmitSearch: onSubmitSearch
      }}
    >
      <>
        <ScrollArea className="h-full" ref={contentContainerRef}>
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
            'fixed bottom-5 left-0 flex min-h-[5rem] w-full flex-col items-center gap-y-2',
            {
              'opacity-100 translate-y-0': showSearchInput,
              'opacity-0 translate-y-10': !showSearchInput
            }
          )}
          style={{ transition: 'all 0.35s ease-out' }}
        >
          <Button
            className={cn('bg-background', {
              'opacity-0 pointer-events-none': !showStop,
              'opacity-100': showStop
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
          <div className="relative z-20 flex justify-center self-stretch px-10">
            <TextAreaSearch
              onSearch={onSubmitSearch}
              className="lg:max-w-4xl"
              placeholder="Ask a follow up question"
              isLoading={isLoading}
            />
          </div>
        </div>
      </>
    </SearchContext.Provider>
  )
}

export const Search = forwardRef<SearchRef>(SearchRenderer)

function AnswerBlock({
  answer,
  showRelatedQuestion
}: {
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

  const IconAnswer = answer.isLoading ? IconSpinner : IconSparkles

  const totalHeightInRem = answer.relevant_documents
    ? Math.ceil(answer.relevant_documents.length / 4) *
        SOURCE_CARD_STYLE.expand +
      0.5 * Math.floor(answer.relevant_documents.length / 4)
    : 0
  return (
    <div className="flex flex-col gap-y-5">
      {/* Relevant documents */}
      {answer.relevant_documents && answer.relevant_documents.length > 0 && (
        <div>
          <div className="mb-1 flex items-center gap-x-2">
            <IconBlocks className="relative" style={{ top: '-0.04rem' }} />
            <p className="text-sm font-bold leading-normal">Sources</p>
          </div>
          <div
            className="gap-sm grid grid-cols-3 gap-2 overflow-hidden md:grid-cols-4"
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
          <IconAnswer
            className={cn({
              'animate-spinner': answer.isLoading
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
              <p className="text-sm font-bold leading-none">Suggestions</p>
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
      className="flex cursor-pointer flex-col justify-between gap-y-1 rounded-lg border bg-card px-3 py-2 transition-all hover:bg-card/60"
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
      <div className="flex items-center text-xs text-muted-foreground">
        <div className="flex flex-1 items-center">
          <SiteFavicon hostname={hostname} />
          <p className="ml-1 overflow-hidden text-ellipsis">
            {hostname.replace('www.', '').split('.')[0]}
          </p>
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
                                      className="relative -top-2 mr-0.5 inline-block h-4 w-4 cursor-pointer rounded-full bg-muted text-center text-xs"
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
