import { createContext, ReactNode, useContext, useMemo, useState } from 'react'
import Image from 'next/image'
import defaultFavicon from '@/assets/default-favicon.png'
import DOMPurify from 'dompurify'
import he from 'he'
import { compact, isNil } from 'lodash-es'
import { marked } from 'marked'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'

import {
  ContextInfo,
  Maybe,
  MessageAttachmentCode,
  MessageAttachmentDoc
} from '@/lib/gql/generates/graphql'
import { AttachmentCodeItem, AttachmentDocItem } from '@/lib/types'
import { cn } from '@/lib/utils'
import { CodeBlock, CodeBlockProps } from '@/components/ui/codeblock'
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger
} from '@/components/ui/hover-card'
import { MemoizedReactMarkdown } from '@/components/markdown'

import './style.css'

import {
  MARKDOWN_CITATION_REGEX,
  MARKDOWN_SOURCE_REGEX
} from '@/lib/constants/regex'

import { Mention } from '../mention-tag'
import { Skeleton } from '../ui/skeleton'

type RelevantDocItem = {
  type: 'doc'
  data: AttachmentDocItem
}

type RelevantCodeItem = {
  type: 'code'
  data: AttachmentCodeItem
  isClient?: boolean
}

type MessageAttachments = Array<RelevantDocItem | RelevantCodeItem>

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

export interface MessageMarkdownProps {
  message: string
  headline?: boolean
  attachmentDocs?: Maybe<Array<AttachmentDocItem>>
  attachmentCode?: Maybe<Array<AttachmentCodeItem>>
  onCopyContent?: ((value: string) => void) | undefined
  onApplyInEditor?: ((value: string) => void) | undefined
  onCodeCitationClick?: (code: MessageAttachmentCode) => void
  onCodeCitationMouseEnter?: (index: number) => void
  onCodeCitationMouseLeave?: (index: number) => void
  contextInfo?: ContextInfo
  fetchingContextInfo?: boolean
  className?: string
  // wrapLongLines for code block
  canWrapLongLines?: boolean
}

type MessageMarkdownContextValue = {
  onCopyContent?: ((value: string) => void) | undefined
  onApplyInEditor?: ((value: string) => void) | undefined
  onCodeCitationClick?: (code: MessageAttachmentCode) => void
  onCodeCitationMouseEnter?: (index: number) => void
  onCodeCitationMouseLeave?: (index: number) => void
  contextInfo: ContextInfo | undefined
  fetchingContextInfo: boolean
  canWrapLongLines: boolean
}

const MessageMarkdownContext = createContext<MessageMarkdownContextValue>(
  {} as MessageMarkdownContextValue
)

export function MessageMarkdown({
  message,
  headline = false,
  attachmentDocs,
  attachmentCode,
  onApplyInEditor,
  onCopyContent,
  contextInfo,
  fetchingContextInfo,
  className,
  canWrapLongLines,
  ...rest
}: MessageMarkdownProps) {
  const messageAttachments: MessageAttachments = useMemo(() => {
    const docs: MessageAttachments =
      attachmentDocs?.map(item => ({
        type: 'doc',
        data: item
      })) ?? []
    const code: MessageAttachments =
      attachmentCode?.map(item => ({
        type: 'code',
        data: item
      })) ?? []
    return compact([...docs, ...code])
  }, [attachmentDocs, attachmentCode])

  const processMessagePlaceholder = (text: string) => {
    const elements: React.ReactNode[] = []
    let lastIndex = 0
    let match

    const addTextNode = (text: string) => {
      if (text) {
        elements.push(text)
      }
    }

    const processMatches = (
      regex: RegExp,
      Component: (...arg: any) => ReactNode,
      getProps: Function
    ) => {
      while ((match = regex.exec(text)) !== null) {
        addTextNode(text.slice(lastIndex, match.index))
        elements.push(<Component key={match.index} {...getProps(match)} />)
        lastIndex = match.index + match[0].length
      }
    }

    processMatches(MARKDOWN_CITATION_REGEX, CitationTag, (match: string) => {
      const citationIndex = parseInt(match[1], 10)
      const citationSource = !isNil(citationIndex)
        ? messageAttachments?.[citationIndex - 1]
        : undefined
      const citationType = citationSource?.type
      const showcitation = citationSource && !isNil(citationIndex)
      return {
        citationIndex,
        showcitation,
        citationType,
        citationSource
      }
    })
    processMatches(MARKDOWN_SOURCE_REGEX, SourceTag, (match: string) => {
      const sourceId = match[1]
      const className = headline ? 'text-[1rem] font-semibold' : undefined
      return { sourceId, className }
    })

    addTextNode(text.slice(lastIndex))

    return elements
  }

  return (
    <MessageMarkdownContext.Provider
      value={{
        onCopyContent,
        onApplyInEditor,
        onCodeCitationClick: rest.onCodeCitationClick,
        onCodeCitationMouseEnter: rest.onCodeCitationMouseEnter,
        onCodeCitationMouseLeave: rest.onCodeCitationMouseLeave,
        contextInfo,
        fetchingContextInfo: !!fetchingContextInfo,
        canWrapLongLines: !!canWrapLongLines
      }}
    >
      <MemoizedReactMarkdown
        className={cn(
          'message-markdown prose max-w-none break-words dark:prose-invert prose-p:leading-relaxed prose-pre:mt-1 prose-pre:p-0',
          className
        )}
        remarkPlugins={[remarkGfm, remarkMath]}
        components={{
          p({ children }) {
            return (
              <p className="mb-2 last:mb-0">
                {children.map((child, index) =>
                  typeof child === 'string' ? (
                    processMessagePlaceholder(child)
                  ) : (
                    <span key={index}>{child}</span>
                  )
                )}
              </p>
            )
          },
          li({ children }) {
            if (children && children.length) {
              return (
                <li>
                  {children.map((childrenItem, index) => {
                    if (typeof childrenItem === 'string') {
                      return processMessagePlaceholder(childrenItem)
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
              <CodeBlockWrapper
                key={Math.random()}
                language={(match && match[1]) || ''}
                value={String(children).replace(/\n$/, '')}
                onApplyInEditor={onApplyInEditor}
                onCopyContent={onCopyContent}
                canWrapLongLines={canWrapLongLines}
                {...props}
              />
            )
          }
        }}
      >
        {message}
      </MemoizedReactMarkdown>
    </MessageMarkdownContext.Provider>
  )
}

export function ErrorMessageBlock({
  error = 'Failed to fetch'
}: {
  error?: string
}) {
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

function CodeBlockWrapper(props: CodeBlockProps) {
  const { canWrapLongLines } = useContext(MessageMarkdownContext)

  return <CodeBlock {...props} canWrapLongLines={canWrapLongLines} />
}

function CitationTag({
  citationIndex,
  showcitation,
  citationType,
  citationSource
}: any) {
  return (
    <div className="inline">
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
    </div>
  )
}

function SourceTag({
  sourceId,
  className
}: {
  sourceId: string | undefined
  className?: string
}) {
  const { contextInfo, fetchingContextInfo } = useContext(
    MessageMarkdownContext
  )

  if (!sourceId) return null
  const source = contextInfo?.sources?.find(o => o.sourceId === sourceId)
  if (!source) return null

  return (
    <span className="node-mention">
      <span>
        {fetchingContextInfo ? (
          <Skeleton className="w-16" />
        ) : (
          <Mention
            id={source.sourceId}
            label={source.sourceName}
            kind={source.sourceKind}
            className={className}
          />
        )}
      </span>
    </span>
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
          className="relative -top-2 mr-0.5 inline-block h-4 w-4 cursor-pointer rounded-full bg-muted text-center text-xs font-medium"
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
  } = useContext(MessageMarkdownContext)

  return (
    <span
      className="relative -top-2 mr-0.5 inline-block h-4 w-4 cursor-pointer rounded-full bg-muted text-center text-xs font-medium"
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

export function SiteFavicon({
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
