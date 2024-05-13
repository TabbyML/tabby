'use client'

import { useEffect, useRef, useState } from 'react'
import TextareaAutosize from 'react-textarea-autosize'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'

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
import { ButtonScrollToBottom } from '@/components/button-scroll-to-bottom'
import { Header } from '@/components/header'
import { MemoizedReactMarkdown } from '@/components/markdown'

interface Source {
  url: string
  title: string
  content: string
}

interface Related {
  title: string
}

const mockData = [
  {
    question: 'how to add function in python',
    answer:
      "Hello World in Python. This is a basic example of a Python program.\nGetting Started. In this section, we'll walk through writing a simple Python program. This program will print out 'Hello, world!'.\nPrerequisites. This program requires Python 3. You can download Python 3 from the official website: https://www.python.org/downloads/\nCode. Here is the Python code:\n```python\ndef say_hello():\n    print('Hello, world!')\n\nsay_hello()\n```\nThis code defines a function, `say_hello`, that prints out the string 'Hello, world!'. Then, it calls this function.\nRunning the Code. To run the code, save it as a .py file and run it from the command line with Python 3.\nConclusion. Congratulations, you've just written and run your first Python program! The print function is one of the most basic functions in Python, but it's also one of the most useful. You can use it to display information, debug your code, and more. Happy coding!",
    sources: [
      {
        url: 'https://github.com/TabbyML/tabby/blob/main/clients/vscode/src/TabbyCompletionProvider.ts#L45-L49',
        title: 'tabby/clients/vscode/src/TabbyCompletionProvider.ts',
        content:
          "```typescript\nworkspace.onDidChangeConfiguration((event) => {\nif (event.affectsConfiguration('tabby') || event.affectsConfiguration('editor.inlineSuggest')) {\nthis.updateConfiguration();\n}\n});```"
      },
      {
        url: 'https://github.com/TabbyML/tabby/issues/2083',
        title: 'Run chat in command line',
        content:
          'I wish to chat to tabby like in your demo page.\nBut from a bash command line.\nAt the moment I just used the docker container.. so I have /opt/tabby/bin/tabby and tabby-cpu'
      },
      {
        url: 'https://gitlab.com/gitlab-org/gitlab/-/blob/master/ee/db/embedding/structure.sql',
        title: 'ee/db/embedding/structure.sql',
        content:
          "```typescript\nworkspace.onDidChangeConfiguration((event) => {\nif (event.affectsConfiguration('tabby') || event.affectsConfiguration('editor.inlineSuggest')) {\nthis.updateConfiguration();\n}\n});```"
      }
    ],
    related: [
      {
        title:
          'What are the key differences between safetensor and GGUF formats?'
      },
      {
        title:
          'Can you provide more detail on the tools and dependencies needed for converting a safetensor model to GGUF?'
      },
      {
        title:
          'What are the benefits of converting a safetensor model to GGUF format?'
      }
    ]
  },
  {
    question: 'write a triangle with CSS',
    answer:
      "The 'hubspot-api' library is not a preloaded package in HubSpot. Instead, you should use the '@hubspot/api-client' package. Here is an example of how to include it:\n\n```javascript\nconst hubspot = require('@hubspot/api-client');\nconst hubspotClient = new hubspot.Client({ apiKey: YOUR_API_KEY });\n```\n\nYou can find more information about the package in the [NPM reference](https://www.npmjs.com/package/@hubspot/api-client) and the [HubSpot Serverless Reference](https://developers.hubspot.com/docs/cms/data/serverless-functions/reference#preloaded-packages).",
    sources: [
      {
        url: 'https://github.com/TabbyML/tabby/blob/main/clients/vscode/src/TabbyCompletionProvider.ts#L45-L49',
        title: 'tabby/clients/vscode/src/TabbyCompletionProvider.ts',
        content:
          "```typescript\nworkspace.onDidChangeConfiguration((event) => {\nif (event.affectsConfiguration('tabby') || event.affectsConfiguration('editor.inlineSuggest')) {\nthis.updateConfiguration();\n}\n});```"
      },
      {
        url: 'https://github.com/TabbyML/tabby/issues/2083',
        title: 'Run chat in command line',
        content:
          'I wish to chat to tabby like in your demo page.\nBut from a bash command line.\nAt the moment I just used the docker container.. so I have /opt/tabby/bin/tabby and tabby-cpu'
      },
      {
        url: 'https://gitlab.com/gitlab-org/gitlab/-/blob/master/ee/db/embedding/structure.sql',
        title: 'ee/db/embedding/structure.sql',
        content:
          "```typescript\nworkspace.onDidChangeConfiguration((event) => {\nif (event.affectsConfiguration('tabby') || event.affectsConfiguration('editor.inlineSuggest')) {\nthis.updateConfiguration();\n}\n});```"
      }
    ],
    related: [
      {
        title:
          'What are the key differences between safetensor and GGUF formats?'
      },
      {
        title:
          'Can you provide more detail on the tools and dependencies needed for converting a safetensor model to GGUF?'
      },
      {
        title:
          'What are the benefits of converting a safetensor model to GGUF format?'
      }
    ]
  }
]

export default function Search() {
  const contentContainerRef = useRef<HTMLDivElement>(null)
  const [container, setContainer] = useState<HTMLDivElement | null>(null)

  useEffect(() => {
    setContainer(
      contentContainerRef?.current?.children[1] as HTMLDivElement | null
    )
  }, [])

  // FIXME: the height considering demo banner
  return (
    <div className="flex h-screen flex-col">
      <Header />
      <ScrollArea className="flex-1" ref={contentContainerRef}>
        <div className="mx-auto px-0 md:w-[48rem] md:px-6">
          <div className="pb-20">
            {mockData.map((data, idx) => (
              <>
                {idx !== 0 && <Separator />}
                <QuestionAnswerPair
                  key={idx}
                  question={data.question}
                  answer={data.answer}
                  sources={data.sources}
                  related={data.related}
                />
              </>
            ))}
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
      <SearchArea />
    </div>
  )
}

function SearchArea() {
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
      return alert('search')
    }
  }

  return (
    <div
      className={cn(
        'fixed bottom-6 left-1/2 transition-all md:-ml-[24rem] md:w-[48rem] md:px-6',
        {
          'opacity-0': !isShow,
          'opacity-100': isShow
        }
      )}
    >
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
          placeholder="Ask a followup question"
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
        >
          <IconArrowRight className="h-3.5 w-3.5" />
        </div>
      </div>
    </div>
  )
}

function QuestionAnswerPair({
  question,
  answer,
  sources,
  related
}: {
  question: string
  answer: string
  sources: Source[]
  related: Related[]
}) {
  return (
    <div className="py-12">
      <h3 className="mb-4 text-2xl font-semibold tracking-tight first:mt-0">
        {question}
      </h3>

      <div className="mb-1 flex items-center gap-x-2">
        <IconBlocks className="relative" style={{ top: '-0.04rem' }} />
        <p className="text-sm font-bold leading-normal">Source</p>
      </div>
      <div className="gap-sm grid grid-cols-4 gap-x-2">
        {sources.map((source, index) => (
          <SourceCard key={source.url} source={source} index={index + 1} />
        ))}
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
              <SheetDescription>{sources.length} resources</SheetDescription>
            </SheetHeader>
            {/* FIXME: pagination or scrolling */}
            <div className="mt-2 flex flex-col gap-y-8">
              {sources.map((source, index) => (
                <SourceBlock source={source} index={index + 1} key={index} />
              ))}
            </div>
          </SheetContent>
        </Sheet>
      </div>

      <div className="mt-9 flex items-center gap-x-1.5">
        <IconSparkles />
        <p className="text-sm font-bold leading-none">Answer</p>
      </div>
      <MemoizedReactMarkdown
        className="prose-full-width prose break-words dark:prose-invert prose-p:leading-relaxed prose-pre:mt-1 prose-pre:p-0"
        remarkPlugins={[remarkGfm, remarkMath]}
        components={{
          p({ children }) {
            return <p className="mb-2 last:mb-0 ">{children}</p>
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
        {answer}
      </MemoizedReactMarkdown>
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

      <div className="mt-9 flex items-center gap-x-1.5">
        <IconLayers />
        <p className="text-sm font-bold leading-none">Related</p>
      </div>
      <div className="mt-3 flex flex-col gap-y-3">
        {related.map((related, index) => (
          <RealtedCard key={index} related={related} />
        ))}
      </div>
    </div>
  )
}

function SourceBlock({ source, index }: { source: Source; index: number }) {
  return (
    <div className="flex gap-x-1">
      <p className="text-sm">{index}.</p>
      <div className="flex-1">
        <p className="text-sm">{source.title}</p>
        <MemoizedReactMarkdown
          className="prose break-words dark:prose-invert prose-p:leading-relaxed prose-pre:mt-1 prose-pre:p-0"
          remarkPlugins={[remarkGfm, remarkMath]}
          components={{
            p({ children }) {
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
          {source.content}
        </MemoizedReactMarkdown>
      </div>
    </div>
  )
}

function SourceCard({ source, index }: { source: Source; index: number }) {
  const { hostname } = new URL(source.url)
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

function RealtedCard({ related }: { related: Related }) {
  return (
    <div className="flex cursor-pointer items-center justify-between rounded-lg border p-4 py-3 transition-all hover:text-primary">
      <p className="w-full overflow-hidden text-ellipsis text-sm">
        {related.title}
      </p>
      <IconPlus />
    </div>
  )
}
