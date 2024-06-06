'use client'

import { createContext, useContext, useEffect, useRef, useState } from 'react'
import Image from 'next/image'
import logoUrl from '@/assets/tabby.png'
import { Message } from 'ai'
import { nanoid } from 'nanoid'
import TextareaAutosize from 'react-textarea-autosize'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'

import { useEnableSearch } from '@/lib/experiment-flags'
import fetcher from '@/lib/tabby/fetcher'
import { AnswerRequest } from '@/lib/types'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { CodeBlock } from '@/components/ui/codeblock'
import {
  IconArrowRight,
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
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger
} from '@/components/ui/sheet'
import { Skeleton } from '@/components/ui/skeleton'
import { ButtonScrollToBottom } from '@/components/button-scroll-to-bottom'
// FIXME: move to lib/hooks
import { useTabbyAnswer } from '@/components/chat/use-tabby-answer'
import { CopyButton } from '@/components/copy-button'
import { MemoizedReactMarkdown } from '@/components/markdown'

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

export function Search() {
  const [searchFlag] = useEnableSearch()
  const [conversation, setConversation] = useState<ConversationMessage[]>([])

  const [container, setContainer] = useState<HTMLDivElement | null>(null)
  const [title, setTitle] = useState('')
  const [currentLoadindId, setCurrentLoadingId] = useState<string>('')
  const contentContainerRef = useRef<HTMLDivElement>(null)

  // FIXME: error
  const { triggerRequest, isLoading, error, answer, stop } = useTabbyAnswer({
    fetcher: tabbyFetcher
  })

  useEffect(() => {
    if (title) return
    document.title = title
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
    let currentAnswer = newConversation.find(
      item => item.id === currentLoadindId
    )
    if (!currentAnswer) return
    currentAnswer.content = answer.answer_delta || ''
    currentAnswer.relevant_documents = answer.relevant_documents
    currentAnswer.relevant_questions = answer.relevant_questions
    currentAnswer.isLoading = isLoading
    setConversation(newConversation)
  }, [isLoading, answer])

  const onSubmitSearch = (question: string) => {
    // FIXME: code query? extra from user's input?
    // FIXME: after search, the search button should spinning for a while
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

    // Scroll to the bottom
    if (container) {
      setTimeout(() => {
        container.scrollTo({
          top: container.scrollHeight,
          behavior: 'smooth'
        })
      }, 2000)
    }

    // Update HTML page title
    setTitle(question)
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
    newTargetAnswer.isLoading = true

    setCurrentLoadingId(newTargetAnswer.id)
    setConversation(newConversation)
    triggerRequest(answerRequest)
  }

  if (!searchFlag.value) {
    return <></>
  }

  const noConversation = conversation.length === 0
  const currentAnswerHasContent = Boolean(
    conversation[conversation.length - 1]?.content
  )
  // FIXME: the height considering demo banner
  return (
    <SearchContext.Provider
      value={{
        isLoading: isLoading,
        onRegenerateResponse: onRegenerateResponse,
        onSubmitSearch: onSubmitSearch
      }}
    >
      <div className="flex h-screen flex-col">
        <ScrollArea className="flex-1" ref={contentContainerRef}>
          <div className="mx-auto px-0 pb-20 md:w-[48rem] md:px-6">
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

        {/* FIXME: adjust position in small width */}
        {container && (
          <ButtonScrollToBottom
            className="!fixed !bottom-9 !right-10 !top-auto"
            container={container}
            offset={100}
          />
        )}

        <div
          className={cn(
            'fixed left-1/2 flex h-24 flex-col items-center transition-all md:-ml-[24rem] md:w-[48rem] md:p-6',
            {
              'bottom-2/3': noConversation,
              'bottom-0': !noConversation
            }
          )}
        >
          {noConversation && (
            <>
              <Image
                src={logoUrl}
                alt="logo"
                width={42}
                className="dark:hidden"
              />
              <h4 className="mb-6 scroll-m-20 text-xl font-semibold tracking-tight text-secondary-foreground">
                Your private search engine (TODO)
              </h4>
            </>
          )}
          {!isLoading && (
            <div className="relative z-20 w-full">
              <SearchArea />
            </div>
          )}
          <Button
            className={cn(
              'absolute top-8 z-0 flex items-center gap-x-2 px-8 py-4',
              {
                'opacity-0 pointer-events-none':
                  !isLoading || !currentAnswerHasContent,
                'opacity-100': isLoading && currentAnswerHasContent
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

function SearchArea() {
  const { onSubmitSearch } = useContext(SearchContext)
  const [isShow, setIsShow] = useState(false)
  const [isFocus, setIsFocus] = useState(false)
  const [value, setValue] = useState('how to add function in python')

  useEffect(() => {
    // Ensure the textarea height remains consistent during rendering
    setIsShow(true)
  }, [])

  const onSearchKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) return e.preventDefault()
  }

  const onSearchKeyUp = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      return search()
    }
  }

  const search = () => {
    if (!value) return
    onSubmitSearch(value)
    setValue('')
  }

  return (
    <div
      className={cn(
        'flex w-full items-center rounded-lg border border-muted-foreground bg-background transition-all hover:border-muted-foreground/60',
        {
          '!border-primary': isFocus
        }
      )}
    >
      <TextareaAutosize
        className={cn(
          'flex-1 resize-none rounded-lg !border-none bg-transparent px-4 py-3 !shadow-none !outline-none !ring-0 !ring-offset-0',
          {
            '!h-[48px]': !isShow
          }
        )}
        placeholder="Ask anything"
        maxRows={5}
        onKeyDown={onSearchKeyDown}
        onKeyUp={onSearchKeyUp}
        onFocus={() => setIsFocus(true)}
        onBlur={() => setIsFocus(false)}
        onChange={e => setValue(e.target.value)}
        value={value}
      />
      <div
        className={cn(
          'mr-3 flex items-center rounded-lg bg-muted p-1 text-muted-foreground transition-all',
          {
            '!bg-primary !text-primary-foreground': value.length > 0
          }
        )}
        onClick={search}
      >
        <IconArrowRight className="h-3.5 w-3.5" />
      </div>
    </div>
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
  return (
    <div className="flex flex-col gap-y-5">
      {/* Relevant documents */}
      {answer.relevant_documents && answer.relevant_documents.length > 0 && (
        <div>
          <div className="mb-1 flex items-center gap-x-2">
            <IconBlocks className="relative" style={{ top: '-0.04rem' }} />
            <p className="text-sm font-bold leading-normal">Source</p>
          </div>
          <div className="gap-sm grid grid-cols-4 gap-x-2">
            {answer.relevant_documents.slice(0, 3).map((source, index) => (
              <SourceCard key={source.link} source={source} index={index + 1} />
            ))}
            {answer.relevant_documents &&
              answer.relevant_documents.length > 3 && (
                <Sheet>
                  <SheetTrigger>
                    <div className="flex h-full cursor-pointer flex-col items-start justify-between gap-y-1 rounded-lg border bg-card px-4 py-2 transition-all hover:bg-card/60">
                      <div className="flex items-center gap-x-0.5 text-xs">
                        <p>Check more</p>
                        <IconChevronRight />
                      </div>
                      <div className="flex h-5 items-center">
                        {answer.relevant_documents
                          .slice(3, 6)
                          .map((source, idx) => {
                            const { hostname } = new URL(source.link)
                            return (
                              <img
                                key={hostname + idx}
                                src={`https://s2.googleusercontent.com/s2/favicons?sz=128&domain_url=${hostname}`}
                                alt={hostname}
                                className="mr-1 h-3.5 w-3.5 rounded-full"
                              />
                            )
                          })}
                      </div>
                    </div>
                  </SheetTrigger>
                  <SheetContent className="flex !max-w-3xl flex-col">
                    <SheetHeader>
                      <SheetTitle>{question}</SheetTitle>
                      <SheetDescription>
                        {answer.relevant_documents.length} resources
                      </SheetDescription>
                    </SheetHeader>
                    <ScrollArea className="flex-1">
                      <div className="mt-2 flex flex-col gap-y-8">
                        {answer.relevant_documents.map((source, index) => (
                          <SourceBlock
                            source={source}
                            index={index + 1}
                            key={index}
                          />
                        ))}
                      </div>
                    </ScrollArea>
                  </SheetContent>
                </Sheet>
              )}
          </div>
        </div>
      )}

      {/* Answer content */}
      <div>
        <div className="flex items-center gap-x-1.5">
          <IconSparkles
            className={cn({
              'spark-animation': answer.isLoading
            })}
          />
          <p className="text-sm font-bold leading-none">Answer</p>
        </div>
        {answer.isLoading && !answer.content && (
          <Skeleton className="h-40 w-full" />
        )}

        <MessageMarkdown message={answer.content} />

        {!answer.isLoading && (
          <div className="mt-3 flex items-center gap-x-3 text-sm">
            <CopyButton
              className="-ml-2.5 gap-x-1 px-2 font-normal text-muted-foreground"
              value={answer.content}
              text="Copy"
            />
            {!isLoading && (
              <Button
                className="flex items-center gap-x-1 px-2 font-normal text-muted-foreground"
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
            <div className="mt-9 flex items-center gap-x-1.5">
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

function SourceBlock({ source, index }: { source: Source; index: number }) {
  return (
    <div className="flex gap-x-1.5">
      <p className="text-sm">{index}.</p>
      <div
        className="flex-1 cursor-pointer transition-opacity hover:opacity-70"
        onClick={() => window.open(source.link)}
      >
        <p className="mb-0.5 text-sm font-bold">{source.title}</p>
        <p className="text-sm">{source.snippet}</p>
      </div>
    </div>
  )
}

function SourceCard({ source, index }: { source: Source; index: number }) {
  const { hostname } = new URL(source.link)
  return (
    <div
      className="flex cursor-pointer flex-col justify-between gap-y-1 rounded-lg border bg-card px-4 py-2 transition-all hover:bg-card/60"
      onClick={() => window.open(source.link)}
    >
      <p className="line-clamp-2 w-full overflow-hidden text-ellipsis break-all text-xs font-semibold">
        {source.title}
      </p>
      <div className="flex items-center">
        <img
          src={`https://s2.googleusercontent.com/s2/favicons?sz=128&domain_url=${hostname}`}
          alt={hostname}
          className="mr-1 h-3.5 w-3.5 rounded-full"
        />

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
  headline = false
}: {
  message: string
  headline?: boolean
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
