'use client'

import { useEffect, useRef, useState } from 'react'
import Image from 'next/image'
import logoUrl from '@/assets/tabby.png'
import { Message } from 'ai'
import { nanoid } from 'nanoid'
import TextareaAutosize from 'react-textarea-autosize'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'

import fetcher from '@/lib/tabby/fetcher'
import { AnswerRequest } from '@/lib/types'
import { cn } from '@/lib/utils'
import { CodeBlock } from '@/components/ui/codeblock'
import {
  IconArrowRight,
  IconBlocks,
  IconCopy,
  IconLayers,
  IconPlus,
  IconRefresh,
  IconSparkles
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

export default function Search() {
  const [conversation, setConversation] = useState<ConversationMessage[]>([])
  // const [conversation, setConversation] = useState<ConversationMessage[]>([{
  //   id: nanoid(), // FIXME
  //   role: 'user',
  //   content: 'add function'
  // }, {
  //   id: nanoid(), // FIXME
  //   role: 'assistant',
  //   content: "",
  //   isLoading: true
  // }])
  const contentContainerRef = useRef<HTMLDivElement>(null)
  const [container, setContainer] = useState<HTMLDivElement | null>(null)
  const [title, setTitle] = useState('')

  // FIXME: error and stop
  const { triggerRequest, isLoading, error, answer, stop } = useTabbyAnswer({
    fetcher: tabbyFetcher
  })

  useEffect(() => {
    document.title = title
  }, [title])

  useEffect(() => {
    setContainer(
      contentContainerRef?.current?.children[1] as HTMLDivElement | null
    )
  }, [])

  // Handling the stream response from useTabbyAnswer
  useEffect(() => {
    if (!answer) return
    const newConversation = [...conversation]
    let currentAnswer = newConversation[newConversation.length - 1]
    currentAnswer.content = answer.answer_delta
    currentAnswer.relevant_documents = answer.relevant_documents
    currentAnswer.relevant_questions = answer.relevant_questions
    currentAnswer.isLoading = isLoading
    setConversation(newConversation)
  }, [isLoading, answer])

  const onSubmitSearch = (question: string) => {
    // FIXME: code query? extra from user's input?
    const newUserMessage: ConversationMessage = {
      id: nanoid(), // FIXME
      role: 'user',
      content: question
    }
    const newAssistantMessage: ConversationMessage = {
      id: nanoid(), // FIXME
      role: 'assistant',
      content: '',
      isLoading: true
    }
    setConversation(
      [...conversation].concat([newUserMessage, newAssistantMessage])
    )

    const previousMessages = conversation.map(message => ({
      role: message.role,
      id: message.id,
      content: message.content
    }))
    const answerRequest: AnswerRequest = {
      messages: [
        ...previousMessages,
        {
          role: 'user',
          id: nanoid(),
          content: question
        }
      ],
      doc_query: true,
      generate_relevant_questions: true
    }
    setTitle(question)
    triggerRequest(answerRequest)
  }

  const noConversation = conversation.length === 0
  // FIXME: the height considering demo banner
  return (
    <div className="flex h-screen flex-col">
      <ScrollArea className="flex-1" ref={contentContainerRef}>
        <div className="mx-auto px-0 md:w-[48rem] md:px-6">
          <div className="flex flex-col pb-20">
            {conversation.map((item, idx) => {
              if (item.role === 'user') {
                return (
                  <div key={item.id}>
                    {idx !== 0 && <Separator />}
                    <div className="pb-2 pt-8">
                      <MessageMarkdown message={item.content} headline />
                    </div>
                  </div>
                )
              }
              if (item.role === 'assistant') {
                return (
                  <div key={item.id} className="pb-8 pt-2">
                    <AnswerBlock question="todo" answer={item} />
                  </div>
                )
              }
              return <></>
            })}
          </div>
        </div>
      </ScrollArea>

      {/* FIXME: support offset, currently the button wont disapper in the bottom */}
      {container && (
        <ButtonScrollToBottom
          className="!fixed !bottom-9 !right-10 !top-auto"
          container={container}
        />
      )}

      <div
        className={cn(
          'fixed left-1/2 flex flex-col items-center transition-all md:-ml-[24rem] md:w-[48rem] md:p-6',
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
        <SearchArea onSubmitSearch={onSubmitSearch} />
      </div>
    </div>
  )
}

function SearchArea({
  onSubmitSearch
}: {
  onSubmitSearch: (question: string) => void
}) {
  // FIXME: the textarea has unexpected flash when it's mounted, maybe it can be fixed after adding loader
  const [isShow, setIsShow] = useState(false)
  const [isFocus, setIsFocus] = useState(false)
  const [value, setValue] = useState('')

  useEffect(() => {
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
        className="flex-1 resize-none rounded-lg !border-none bg-transparent px-4 py-3 !shadow-none !outline-none !ring-0 !ring-offset-0"
        placeholder="Ask anything"
        maxRows={15}
        onKeyDown={onSearchKeyDown}
        onKeyUp={onSearchKeyUp}
        onFocus={() => setIsFocus(true)}
        onBlur={() => setIsFocus(false)}
        onChange={e => setValue(e.target.value)}
        value={value}
      />
      <div
        className={cn(
          'mr-3 flex items-center rounded-lg bg-muted p-1 text-muted-foreground transition-all  ',
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
  answer
}: {
  question: string
  answer: ConversationMessage
}) {
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
            {answer.relevant_documents.map((source, index) => (
              <SourceCard key={source.link} source={source} index={index + 1} />
            ))}
            {answer.relevant_documents &&
              answer.relevant_documents.length > 3 && (
                <Sheet>
                  <SheetTrigger>
                    <div className="flex h-full cursor-pointer flex-col justify-between gap-y-1 rounded-lg border bg-card px-4 py-2 hover:bg-card/60">
                      <div className="flex flex-1 gap-x-1 py-1">
                        <img
                          src={`https://s2.googleusercontent.com/s2/favicons?sz=128&domain_url=github.com`}
                          alt="github.com"
                          className="mr-1 h-3.5 w-3.5 rounded-full"
                        />
                        <img
                          src={`https://s2.googleusercontent.com/s2/favicons?sz=128&domain_url=github.com`}
                          alt="github.com"
                          className="mr-1 h-3.5 w-3.5 rounded-full"
                        />
                      </div>

                      <p className="flex items-center gap-x-0.5 text-xs text-muted-foreground">
                        Check mroe
                      </p>
                    </div>
                  </SheetTrigger>
                  <SheetContent className="!max-w-3xl">
                    <SheetHeader>
                      <SheetTitle>
                        {question} (Style here need to be polished)
                      </SheetTitle>
                      <SheetDescription>
                        {answer.relevant_documents.length} resources
                      </SheetDescription>
                    </SheetHeader>
                    {/* FIXME: pagination or scrolling */}
                    <div className="mt-2 flex flex-col gap-y-8">
                      {answer.relevant_documents.map((source, index) => (
                        <SourceBlock
                          source={source}
                          index={index + 1}
                          key={index}
                        />
                      ))}
                    </div>
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
          <div className="flex flex-col gap-y-1">
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-32 w-full" />
          </div>
        )}

        <MessageMarkdown message={answer.content} />

        {!answer.isLoading && (
          <div className="mt-3 flex items-center gap-x-3 text-sm">
            <div className="flex cursor-pointer items-center gap-x-0.5 text-muted-foreground transition-all hover:text-primary">
              <IconCopy />
              <p>Copy</p>
            </div>
            <div className="flex cursor-pointer items-center gap-x-0.5 text-muted-foreground transition-all hover:text-primary">
              <IconRefresh />
              <p>Regenerate</p>
            </div>
          </div>
        )}
      </div>

      {/* Related questions */}
      {answer.relevant_questions && answer.relevant_questions.length > 0 && (
        <div>
          <div className="mt-9 flex items-center gap-x-1.5">
            <IconLayers />
            <p className="text-sm font-bold leading-none">Related</p>
          </div>
          <div className="mt-3 flex flex-col gap-y-3">
            {answer.relevant_questions?.map((related, index) => (
              <RealtedCard key={index} related={related} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function SourceBlock({ source, index }: { source: Source; index: number }) {
  return (
    <div className="flex gap-x-1">
      <p className="text-sm">{index}.</p>
      <div className="flex-1">
        <p className="text-sm">{source.title}</p>
        <p>{source.snippet}</p>
      </div>
    </div>
  )
}

function SourceCard({ source, index }: { source: Source; index: number }) {
  const { hostname } = new URL(source.link)
  return (
    <div className="flex cursor-pointer flex-col justify-between gap-y-1 rounded-lg border bg-card px-4 py-2 transition-all hover:bg-card/60">
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
            {hostname.split('.')[0]}
          </p>
          <span className="relative -top-1.5 text-xl leading-none">.</span>
          <p>{index}</p>
        </div>
      </div>
    </div>
  )
}

function RealtedCard({ related }: { related: string }) {
  return (
    <div className="flex cursor-pointer items-center justify-between rounded-lg border p-4 py-3 transition-all hover:text-primary">
      <p className="w-full overflow-hidden text-ellipsis text-sm">{related}</p>
      <IconPlus />
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
  // FIXME: onCopyContent
  // const { onCopyContent } = React.useContext(ChatContext)
  return (
    <MemoizedReactMarkdown
      className="prose max-w-none break-words dark:prose-invert prose-p:leading-relaxed prose-pre:mt-1 prose-pre:p-0"
      remarkPlugins={[remarkGfm, remarkMath]}
      components={{
        p({ children }) {
          if (headline) {
            return (
              <h3 className="break-anywhere cursor-text scroll-m-20 text-xl font-semibold tracking-tight sm:w-9/12 sm:text-2xl">
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
              // onCopyContent={onCopyContent}
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
