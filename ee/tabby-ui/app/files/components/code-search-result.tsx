'use client'

import React from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { Extension } from '@codemirror/state'
import { lineNumbers } from '@codemirror/view'
import { escapeRegExp, isNil } from 'lodash-es'
import { useTheme } from 'next-themes'
import LazyLoad from 'react-lazy-load'

import { GrepLine } from '@/lib/gql/generates/graphql'
import { filename2prism } from '@/lib/language-utils'
import CodeEditor from '@/components/codemirror/codemirror'
import { lineClickExtension } from '@/components/codemirror/line-click-extension'

import { searchMatchExtension } from '../lib/search-match-extension'
import { SourceCodeSearchResult as SourceCodeSearchResultType } from './code-search-result-view'
import { SourceCodeBrowserContext } from './source-code-browser'
import { generateEntryPath } from './utils'

interface SourceCodeSearchResultProps {
  result: SourceCodeSearchResultType
  query: string
}

const MemoizedFilePathView = React.memo(
  ({ path, pattern }: { path: string; pattern?: string }) => {
    if (!pattern) return path

    const regex = new RegExp(escapeRegExp(pattern), 'gi')
    let matches: Array<{ start: number; end: number }> = []
    let match: RegExpExecArray | null

    while ((match = regex.exec(path)) !== null) {
      const start = match.index
      const end = start + match[0].length
      matches.push({
        start,
        end
      })
    }

    return <HighlightText text={path} matches={matches} />
  }
)
MemoizedFilePathView.displayName = 'FilePathView'

export const SourceCodeSearchResult = ({
  result,
  query
}: SourceCodeSearchResultProps) => {
  const { theme } = useTheme()
  const { activeRepo, activeEntryInfo } = React.useContext(
    SourceCodeBrowserContext
  )
  const fileFilter = React.useMemo(() => {
    return query?.match(/f:(\S+)/)?.[1]
  }, [query])

  const language = filename2prism(result.path)[0]

  const ranges = React.useMemo(() => {
    const newRanges: { start: number; end: number }[] = []
    let start: number = 0
    let end: number = 0
    let lastLineNumber: number | undefined
    const lines = result.lines ?? []
    lines.forEach((line, index) => {
      if (index === 0) {
        start = index
        end = index
        lastLineNumber = line.lineNumber
      } else if (
        !isNil(lastLineNumber) &&
        line.lineNumber === lastLineNumber + 1
      ) {
        lastLineNumber = line.lineNumber
        end = index
      } else {
        lastLineNumber = line.lineNumber
        newRanges.push({ start, end })
        start = index
        end = index
      }
    })

    if (end !== lines?.length) {
      newRanges.push({ start, end })
    }

    return newRanges
  }, [result.lines])

  const pathname = `/files/${generateEntryPath(
    activeRepo,
    activeEntryInfo.rev,
    result.path,
    'file'
  )}`

  return (
    <>
      <div className="sticky top-9 z-10 border bg-secondary p-2 text-secondary-foreground">
        <Link
          href={{
            pathname
          }}
          className="inline-flex font-medium text-primary hover:underline"
        >
          <MemoizedFilePathView path={result.path} pattern={fileFilter} />
        </Link>
      </div>
      <div className="divide-y-border mb-6 grid divide-y overflow-x-auto border border-t-0">
        {ranges.map((range, index) => {
          const lines = result.lines.slice(range.start, range.end + 1)
          return (
            <LazyLoad
              height={lines.length * 20 + 9}
              key={`${result.path}-${range.start}`}
              offset={300}
            >
              <CodeSearchSnippet
                language={language}
                theme={theme}
                lines={lines}
                path={result.path}
              />
            </LazyLoad>
          )
        })}
      </div>
    </>
  )
}

function HighlightText({
  text,
  matches
}: {
  text: string
  matches?: Array<{ start: number; end: number }>
}) {
  if (!matches || matches.length === 0) {
    return <span>{text}</span>
  }

  const highlighted = []
  let lastIndex = 0

  matches.forEach((match, index) => {
    if (match.start > lastIndex) {
      highlighted.push(
        <span key={`text-${index}`}>
          {text.substring(lastIndex, match.start)}
        </span>
      )
    }
    highlighted.push(
      <span key={`match-${index}`} className="bg-[hsl(var(--mark-bg))]">
        {text.substring(match.start, match.end)}
      </span>
    )
    lastIndex = match.end
  })

  if (lastIndex < text.length) {
    highlighted.push(<span key="last">{text.substring(lastIndex)}</span>)
  }

  return <span>{highlighted}</span>
}

interface CodeSearchSnippetProps {
  language: string
  theme: string | undefined
  lines?: GrepLine[]
  path: string
}

function CodeSearchSnippet({
  theme,
  language,
  lines,
  path
}: CodeSearchSnippetProps) {
  const router = useRouter()
  const { activeRepo, activeEntryInfo } = React.useContext(
    SourceCodeBrowserContext
  )
  const value = React.useMemo(() => {
    const text =
      lines?.reduce((sum, cur) => {
        let text = cur.line.text || atob(cur.line.base64 || '') || ''
        return sum + text
      }, '') ?? ''
    return text.replace(/\n$/, '')
  }, [])
  const startLineNumber = lines?.[0]?.lineNumber || 0
  const extensions: Extension[] = React.useMemo(() => {
    if (lines?.length) {
      const matches: any[] = []
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i]
        for (const match of line.subMatches) {
          matches.push({
            bytesStart: match.bytesStart,
            bytesEnd: match.bytesEnd,
            lineNumber: i + 1
          })
        }
      }
      return [
        lineNumbers({
          formatNumber(lineNo) {
            return lines[lineNo - 1]?.lineNumber.toString() ?? lineNo
          }
        }),
        lineClickExtension((lineNo: number) => {
          const pathname = `/files/${generateEntryPath(
            activeRepo,
            activeEntryInfo.rev,
            path,
            'file'
          )}`
          const lineNumber = startLineNumber + lineNo - 1
          router.push(`${pathname}?plain=1#L${lineNumber}`)
        }),
        ...searchMatchExtension(matches)
      ]
    }
    return []
  }, [lines])

  return (
    <CodeEditor
      value={value}
      theme={theme}
      language={language}
      readonly
      extensions={extensions}
    />
  )
}
