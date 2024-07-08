'use client'

import React from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { Extension } from '@codemirror/state'
import { lineNumbers } from '@codemirror/view'
import { isNil } from 'lodash-es'
import { useTheme } from 'next-themes'
import LazyLoad from 'react-lazy-load'

import { GrepLine } from '@/lib/gql/generates/graphql'
import { filename2prism } from '@/lib/language-utils'
import CodeEditor from '@/components/codemirror/codemirror'
import { lineClickExtension } from '@/components/codemirror/line-click-extension'

import { SourceCodeSearchResult as SourceCodeSearchResultType } from './code-search-result-view'
import { searchMatchExtension } from './search-match-extension'
import { SourceCodeBrowserContext } from './source-code-browser'
import { generateEntryPath } from './utils'

interface SourceCodeSearchResultProps {
  result: SourceCodeSearchResultType
  query: string
}

export const SourceCodeSearchResult = ({
  ...props
}: SourceCodeSearchResultProps) => {
  const { theme } = useTheme()
  const { activeRepo, activeRepoRef } = React.useContext(
    SourceCodeBrowserContext
  )

  const language = filename2prism(props.result.path)[0]

  const ranges = React.useMemo(() => {
    const newRanges: { start: number; end: number }[] = []
    let start: number = 0
    let end: number = 0
    let lastLineNumber: number | undefined
    const lines = props.result.lines ?? []
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

    if (!newRanges?.length) {
      newRanges.push({ start, end })
    }

    return newRanges
  }, [props.result.lines])

  const pathname = `/files/${generateEntryPath(
    activeRepo,
    activeRepoRef?.name as string,
    props.result.path,
    'file'
  )}`

  return (
    <div>
      <div className="sticky top-0 z-10 bg-background">
        <Link
          href={{
            pathname
          }}
          className="mb-2 inline-flex font-medium text-primary hover:underline"
        >
          {props.result.path}
        </Link>
      </div>
      <div className="divide-y-border grid divide-y border border-border">
        {ranges.map((range, i) => {
          const lines = props.result.lines.slice(range.start, range.end + 1)
          return (
            <LazyLoad key={`${props.result.path}-${range.start}`} offset={300}>
              <CodeSearchSnippet
                language={language}
                theme={theme}
                lines={lines}
                path={props.result.path}
              />
            </LazyLoad>
          )
        })}
      </div>
    </div>
  )
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
  const { activeRepo, activeRepoRef } = React.useContext(
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
            activeRepoRef?.name as string,
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
