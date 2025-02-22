import { ReactNode, useContext, useMemo, useState } from 'react'
import { compact, isNil } from 'lodash-es'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'

import {
  ContextInfo,
  Maybe,
  MessageAttachmentClientCode
} from '@/lib/gql/generates/graphql'
import {
  AttachmentCodeItem,
  AttachmentDocItem,
  FileContext,
  RelevantCodeContext
} from '@/lib/types'
import {
  cn,
  convertFilepath,
  encodeMentionPlaceHolder,
  getRangeFromAttachmentCode,
  resolveFileNameForDisplay
} from '@/lib/utils'
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger
} from '@/components/ui/hover-card'
import { MemoizedReactMarkdown } from '@/components/markdown'

import './style.css'

import { SquareFunctionIcon } from 'lucide-react'
import {
  FileLocation,
  Filepath,
  ListSymbolItem,
  LookupSymbolHint,
  SymbolInfo
} from 'tabby-chat-panel/index'

import {
  MARKDOWN_CITATION_REGEX,
  MARKDOWN_FILE_REGEX,
  MARKDOWN_SOURCE_REGEX,
  MARKDOWN_SYMBOL_REGEX
} from '@/lib/constants/regex'

import { Mention } from '../mention-tag'
import { IconFile } from '../ui/icons'
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
    processMatches(MARKDOWN_FILE_REGEX, FileTag, (match: string) => {
      const encodedFilepath = match[1]
      try {
        return {
          encodedFilepath,
          openInEditor
        }
      } catch (e) {}
    })

    processMatches(
      MARKDOWN_SYMBOL_REGEX,
      SymbolTag,
      (match: RegExpExecArray) => {
        const fullMatch = match[1]
        return {
          encodedSymbol: fullMatch,
          openInEditor
        }
      }
    )

    addTextNode(text.slice(lastIndex))

    return elements
  }

  const lookupSymbol = async (keyword: string) => {
    if (!onLookupSymbol) return
    if (symbolPositionMap.has(keyword)) return

    setSymbolLocationMap(map => new Map(map.set(keyword, undefined)))
    const hints: LookupSymbolHint[] = []

    attachmentClientCode?.forEach(item => {
      const code = item as AttachmentCodeItem

      // FIXME(Sma1lboy): using getFilepathFromContext after refactor FileContext
      hints.push({
        filepath: code.gitUrl
          ? {
              kind: 'git',
              gitUrl: code.gitUrl,
              filepath: code.filepath
            }
          : code.baseDir
          ? {
              kind: 'workspace',
              filepath: code.filepath,
              baseDir: code.baseDir
            }
          : {
              kind: 'uri',
              uri: code.filepath
            },
        location: getRangeFromAttachmentCode(code)
      })
    })

    const symbolInfo = await onLookupSymbol(keyword, hints)
    setSymbolLocationMap(map => new Map(map.set(keyword, symbolInfo)))
  }

  const encodedMessage = useMemo(() => {
    return encodeMentionPlaceHolder(message)
  }, [message])

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
              // FIXME
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
          },
          hr() {
            return null
          }
        }}
      >
        {encodedMessage}
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
    <span>
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

function FileTag({
  encodedFilepath,
  openInEditor,
  className
}: {
  encodedFilepath: string | undefined
  className?: string
  openInEditor?: MessageMarkdownProps['openInEditor']
}) {
  const filepath = useMemo(() => {
    if (!encodedFilepath) return null
    try {
      const decodedFilepath = decodeURIComponent(encodedFilepath)
      const filepath = JSON.parse(decodedFilepath) as Filepath
      return filepath
    } catch (e) {
      return null
    }
  }, [encodedFilepath])

  const filepathString = useMemo(() => {
    if (!filepath) return undefined

    return convertFilepath(filepath).filepath
  }, [filepath])

  const handleClick = () => {
    if (!openInEditor || !filepath) return
    openInEditor({ filepath })
  }

  if (!filepathString) return null

  return (
    <span
      className={cn(
        'symbol space-x-1 whitespace-nowrap border bg-muted py-0.5 align-middle leading-5',
        className,
        {
          'hover:bg-muted/50 cursor-pointer': !!openInEditor && !!filepath
        }
      )}
      onClick={handleClick}
    >
      <IconFile className="relative -top-px inline-block h-3.5 w-3.5" />
      <span className={cn('whitespace-normal font-medium')}>
        {resolveFileNameForDisplay(filepathString)}
      </span>
    </span>
  )
}

function SymbolTag({
  encodedSymbol,
  openInEditor,
  className
}: {
  encodedSymbol: string | undefined
  className?: string
  openInEditor?: MessageMarkdownProps['openInEditor']
}) {
  const symbol = useMemo(() => {
    if (!encodedSymbol) return null
    try {
      const decodedSymbol = decodeURIComponent(encodedSymbol)
      return JSON.parse(decodedSymbol) as ListSymbolItem
    } catch (e) {
      return null
    }
  }, [encodedSymbol])

  const handleClick = () => {
    if (!openInEditor || !symbol) return
    openInEditor({
      filepath: symbol.filepath,
      location: symbol.range
    })
  }

  if (!symbol?.label) return null

  return (
    <span
      className={cn(
        'symbol space-x-1 whitespace-nowrap border bg-muted py-0.5 align-middle leading-5',
        className,
        {
          'hover:bg-muted/50 cursor-pointer': !!openInEditor
        }
      )}
      onClick={handleClick}
    >
      <SquareFunctionIcon className="relative -top-px inline-block h-3.5 w-3.5" />
      <span className="font-medium">{symbol.label}</span>
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

  const context: RelevantCodeContext = useMemo(() => {
    return {
      kind: 'file',
      range: getRangeFromAttachmentCode(relevantCode),
      filepath: relevantCode.filepath || '',
      content: relevantCode.content,
      git_url: ''
    }
  }, [relevantCode])

  const isMultiLine =
    context.range &&
    !isNil(context.range?.start) &&
    !isNil(context.range?.end) &&
    context.range.start < context.range.end
  const pathSegments = context.filepath.split('/')
  const path = pathSegments.slice(0, pathSegments.length - 1).join('/')

  const fileName = useMemo(() => {
    return resolveFileNameForDisplay(context.filepath)
  }, [context.filepath])

  const rangeText = useMemo(() => {
    if (!context.range) return undefined

    let text = ''
    if (context.range.start) {
      text = String(context.range.start)
    }
    if (isMultiLine) {
      text += `-${context.range.end}`
    }
    return text
  }, [context.range])

  return (
    <HoverCard openDelay={100} closeDelay={100}>
      <HoverCardTrigger>
        <span
          className="relative -top-2 mx-0.5 inline-block h-4 w-4 cursor-pointer rounded-full bg-muted text-center text-xs font-medium"
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
      </HoverCardTrigger>
      <HoverCardContent
        className="max-w-[90vw] overflow-x-hidden bg-background py-2 text-sm text-foreground dark:border-muted-foreground/60 md:py-4 lg:w-96"
        collisionPadding={8}
      >
        <div
          className="cursor-pointer space-y-2 hover:opacity-70"
          onClick={() => onCodeCitationClick?.(relevantCode)}
        >
          <div className="truncate whitespace-nowrap font-medium">
            <span>{fileName}</span>
            {rangeText ? (
              <span className="text-muted-foreground">:{rangeText}</span>
            ) : null}
          </div>
          {!!path && (
            <div className="break-all text-xs text-muted-foreground">
              {path}
            </div>
          )}
        </div>
      </HoverCardContent>
    </HoverCard>
  )
}
