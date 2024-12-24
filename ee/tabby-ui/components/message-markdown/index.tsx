import { ReactNode, useContext, useMemo, useState } from 'react'
import { compact, isNil } from 'lodash-es'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'

import {
  ContextInfo,
  Maybe,
  MessageAttachmentClientCode
} from '@/lib/gql/generates/graphql'
import { AttachmentCodeItem, AttachmentDocItem, FileContext } from '@/lib/types'
import { cn } from '@/lib/utils'
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger
} from '@/components/ui/hover-card'
import { MemoizedReactMarkdown } from '@/components/markdown'

import './style.css'

import {
  FileLocation,
  Filepath,
  LookupSymbolHint,
  SymbolInfo
} from 'tabby-chat-panel/index'

import {
  MARKDOWN_CITATION_REGEX,
  MARKDOWN_SOURCE_REGEX
} from '@/lib/constants/regex'

import { Mention } from '../mention-tag'
import { Skeleton } from '../ui/skeleton'
import { CodeElement } from './code'
import { DocDetailView } from './doc-detail-view'
import { MessageMarkdownContext } from './markdown-context'

type RelevantDocItem = {
  type: 'doc'
  data: AttachmentDocItem
}

type RelevantCodeItem = {
  type: 'code'
  data: AttachmentCodeItem | MessageAttachmentClientCode
  isClient?: boolean
}

type MessageAttachments = Array<RelevantDocItem | RelevantCodeItem>

export interface MessageMarkdownProps {
  message: string
  headline?: boolean
  attachmentDocs?: Maybe<Array<AttachmentDocItem>>
  attachmentCode?: Maybe<Array<AttachmentCodeItem>>
  attachmentClientCode?: Maybe<Array<MessageAttachmentClientCode>>
  onCopyContent?: ((value: string) => void) | undefined
  onApplyInEditor?: (
    content: string,
    opts?: { languageId: string; smart: boolean }
  ) => void
  onLookupSymbol?: (
    symbol: string,
    hints?: LookupSymbolHint[] | undefined
  ) => Promise<SymbolInfo | undefined>
  openInEditor?: (target: FileLocation) => void
  onCodeCitationClick?: (code: AttachmentCodeItem) => void
  onCodeCitationMouseEnter?: (index: number) => void
  onCodeCitationMouseLeave?: (index: number) => void
  contextInfo?: ContextInfo
  fetchingContextInfo?: boolean
  className?: string
  // wrapLongLines for code block
  canWrapLongLines?: boolean
  supportsOnApplyInEditorV2: boolean
  activeSelection?: FileContext
}

export function MessageMarkdown({
  message,
  headline = false,
  attachmentDocs,
  attachmentClientCode,
  attachmentCode,
  onApplyInEditor,
  onCopyContent,
  contextInfo,
  fetchingContextInfo,
  className,
  canWrapLongLines,
  onLookupSymbol,
  openInEditor,
  supportsOnApplyInEditorV2,
  activeSelection,
  ...rest
}: MessageMarkdownProps) {
  const [symbolPositionMap, setSymbolLocationMap] = useState<
    Map<string, SymbolInfo | undefined>
  >(new Map())
  const messageAttachments: MessageAttachments = useMemo(() => {
    const docs: MessageAttachments =
      attachmentDocs?.map(item => ({
        type: 'doc',
        data: item
      })) ?? []

    const clientCode: MessageAttachments =
      attachmentClientCode?.map(item => ({
        type: 'code',
        data: item
      })) ?? []

    const code: MessageAttachments =
      attachmentCode?.map(item => ({
        type: 'code',
        data: item
      })) ?? []
    return compact([...docs, ...clientCode, ...code])
  }, [attachmentDocs, attachmentClientCode, attachmentCode])

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

  const lookupSymbol = async (keyword: string) => {
    if (!onLookupSymbol) return
    if (symbolPositionMap.has(keyword)) return

    setSymbolLocationMap(map => new Map(map.set(keyword, undefined)))
    const hints: LookupSymbolHint[] = []
    if (activeSelection && activeSelection?.range) {
      // FIXME(@icycodes): this is intended to convert the filepath to Filepath type
      // We should remove this after FileContext.filepath use type Filepath instead of string
      let filepath: Filepath
      if (
        activeSelection.git_url.length > 1 &&
        !activeSelection.filepath.includes(':')
      ) {
        filepath = {
          kind: 'git',
          filepath: activeSelection.filepath,
          gitUrl: activeSelection.git_url
        }
      } else {
        filepath = {
          kind: 'uri',
          uri: activeSelection.filepath
        }
      }
      hints.push({
        filepath,
        location: {
          start: activeSelection.range.start,
          end: activeSelection.range.end
        }
      })
    }
    const symbolInfo = await onLookupSymbol(keyword, hints)
    setSymbolLocationMap(map => new Map(map.set(keyword, symbolInfo)))
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
        canWrapLongLines: !!canWrapLongLines,
        supportsOnApplyInEditorV2,
        activeSelection,
        symbolPositionMap,
        lookupSymbol: onLookupSymbol ? lookupSymbol : undefined,
        openInEditor
      }}
    >
      <MemoizedReactMarkdown
        className={cn(
          'message-markdown prose max-w-none break-words dark:prose-invert prose-p:leading-relaxed prose-pre:mt-1 prose-pre:p-0',
          {
            'cursor-default': !!onApplyInEditor
          },
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
            return (
              <CodeElement
                node={node}
                inline={inline}
                className={className}
                {...props}
              >
                {children}
              </CodeElement>
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
  relevantDocument: AttachmentDocItem
  citationIndex: number
}) {
  return (
    <HoverCard openDelay={100} closeDelay={100}>
      <HoverCardTrigger>
        <span
          className="relative -top-2 mr-0.5 inline-block h-4 w-4 cursor-pointer rounded-full bg-muted text-center text-xs font-medium"
          onClick={() => window.open(relevantDocument.link)}
        >
          {citationIndex}
        </span>
      </HoverCardTrigger>
      <HoverCardContent className="w-96 bg-background text-sm text-foreground dark:border-muted-foreground/60">
        <DocDetailView relevantDocument={relevantDocument} />
      </HoverCardContent>
    </HoverCard>
  )
}

function RelevantCodeBadge({
  relevantCode,
  citationIndex
}: {
  relevantCode: AttachmentCodeItem
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
