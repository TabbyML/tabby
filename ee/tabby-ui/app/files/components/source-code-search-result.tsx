'use client'

import React from 'react'
import Link from 'next/link'
import LazyLoad from 'react-lazy-load'

import { filename2prism } from '@/lib/language-utils'

import CodeEditorView from './code-editor-view'
import { SourceCodeBrowserContext } from './source-code-browser'
import { SourceCodeSearchResult as SourceCodeSearchResultType } from './source-code-search-results'
import { generateEntryPath } from './utils'

interface SourceCodeSearchResultProps {
  result: SourceCodeSearchResultType
  query: string
}

export const SourceCodeSearchResult = ({
  ...props
}: SourceCodeSearchResultProps) => {
  const { activeRepo, activeRepoRef } = React.useContext(
    SourceCodeBrowserContext
  )

  const language = filename2prism(props.result.path)[0]

  const [ranges, setRanges] = React.useState<{ start: number; end: number }[]>(
    []
  )

  const [firstLineWithSubMatch, setFirstLineWithSubMatch] = React.useState<
    number | null
  >(null)

  React.useEffect(() => {
    const newRanges: { start: number; end: number }[] = []
    let currentRange: { start: number; end: number } = { start: 0, end: 0 }
    let firstLineSet = false

    props.result.lines.forEach((line, index) => {
      if (line.subMatches.length > 0 && firstLineSet === false) {
        setFirstLineWithSubMatch(line.lineNumber)
        firstLineSet = true
      }
      if (index === 0) {
        currentRange.start = line.lineNumber
        currentRange.end = line.lineNumber
      } else {
        if (line.lineNumber === currentRange.end + 1) {
          currentRange.end = line.lineNumber
        } else {
          newRanges.push(currentRange)
          currentRange = { start: line.lineNumber, end: line.lineNumber }
        }
      }
    })

    newRanges.push(currentRange)

    setRanges(newRanges)
  }, [firstLineWithSubMatch, props.result.lines])

  const pathname = `/files/${generateEntryPath(
    activeRepo,
    activeRepoRef?.name as string,
    props.result.path,
    'file'
  )}`

  /**
   * We need to contextually offset the line number based on the additional
   * lines provided by the backend. If the `range.start` is greater than 1, we
   * need to add 3 in order to highlight the subMatch. Otherwise, we can just use the
   * `range.start` as is.
   */
  const getHash = (range: { start: number; end: number }, index: number) => {
    if (range.start > 1) {
      // Handle the default offset
      return `L${range.start + 3}`
    } else {
      return `L${firstLineWithSubMatch}`
    }
  }

  return (
    <div>
      <div className="sticky top-0 bg-background z-10">
        <Link
          href={{
            pathname,
            hash: `L${firstLineWithSubMatch}`
          }}
          className="mb-2 font-medium inline-flex text-primary hover:underline"
        >
          {props.result.path}
        </Link>
      </div>
      <div className="grid border divide-y divide-y-border border-border">
        {ranges.map((range, i) => (
          <LazyLoad key={`${props.result.path}-${i}`}>
            <Link
              href={{
                pathname,
                hash: getHash(range, i)
              }}
              className="group relative"
            >
              <div className="absolute left-0 w-full h-full top-0 hidden group-hover:block group-focus:block bg-accent"></div>
              <div className="group-hover:opacity-75 group-focus:opacity-75">
                <CodeEditorView
                  value={props.result.blob}
                  language={language}
                  stringToMatch={props.query}
                  lineRange={range}
                  interactionsAreDisabled={true}
                />
              </div>
            </Link>
          </LazyLoad>
        ))}
      </div>
    </div>
  )
}
