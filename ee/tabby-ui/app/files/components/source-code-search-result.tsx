'use client'

import React from 'react'
import Link from 'next/link'

import { GrepFile } from '@/lib/gql/generates/graphql'
import { filename2prism } from '@/lib/language-utils'
import { IconFile } from '@/components/ui/icons'

import CodeEditorView from './code-editor-view'
import { SourceCodeBrowserContext } from './source-code-browser'
import { generateEntryPath } from './utils'

export interface SourceCodeSearchResult extends GrepFile {
  blob: string
}

interface SourceCodeSearchResultProps {
  result: SourceCodeSearchResult
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

    props.result.lines.forEach((line, index) => {
      if (line.subMatches.length > 0) {
        setFirstLineWithSubMatch(line.lineNumber)
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
  }, [props.result.lines])

  return (
    <div>
      <Link
        href={{
          pathname: `/files/${generateEntryPath(
            activeRepo,
            activeRepoRef?.name as string,
            props.result.path,
            'file'
          )}`,
          // FIXME: this doesn't work when clicking a different line on the active file
          hash: `L${firstLineWithSubMatch}`
        }}
        className="mb-2 inline-flex items-center gap-2"
      >
        <IconFile />
        {/* TODO: this should highlight the first line subMatches */}
        {props.result.path}
      </Link>
      <div className="overflow-hidden grid border divide-y divide-y-border border-border rounded">
        {ranges.map((range, i) => (
          <CodeEditorView
            key={`${props.result.path}-${i}`}
            value={props.result.blob}
            language={language}
            stringToMatch={props.query}
            lineRange={range}
          />
        ))}
      </div>
    </div>
  )
}
