'use client'

import React, { useEffect } from 'react'
import Link from 'next/link'

import { GrepFile } from '@/lib/gql/generates/graphql'
import { filename2prism } from '@/lib/language-utils'
import { IconFile } from '@/components/ui/icons'

import CodeEditorView from '../code-editor-view'
import { SourceCodeBrowserContext } from '../source-code-browser'
import { generateEntryPath } from '../utils'

export interface GlobalSearchResult extends GrepFile {
  blob: string
}

interface GlobalSearchResultProps {
  result: GlobalSearchResult
  query: string
}

export const GlobalSearchResult = ({ ...props }: GlobalSearchResultProps) => {
  const { activeRepo, activeRepoRef } = React.useContext(
    SourceCodeBrowserContext
  )

  const language = filename2prism(props.result.path)[0]

  const [ranges, setRanges] = React.useState<{ start: number; end: number }[]>(
    []
  )

  /**
   *
   */
  const getBlobAtRange = (range: { start: number; end: number }) => {
    const lineArray = props.result.blob.split('\n')
    return lineArray.slice(range.start - 1, range.end).join('\n')
  }

  useEffect(() => {
    const newRanges: { start: number; end: number }[] = []
    let currentRange: { start: number; end: number } = { start: 0, end: 0 }

    props.result.lines.forEach((line, index) => {
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
        href={`/files/${generateEntryPath(
          activeRepo,
          activeRepoRef?.name as string,
          props.result.path,
          'file'
        )}`}
        className="font-bold mb-2 inline-flex items-center gap-2"
      >
        <IconFile />
        {props.result.path}
      </Link>
      {/* FIXME: are  */}
      <div className="overflow-hidden grid border divide-y divide-y-border border-border rounded">
        {/* Loop through each range to create some separation? */}
        {ranges.map((range, i) => (
          <>
            {/* Here we just want a blob from between the ranges */}

            <CodeEditorView
              key={`${props.result.path}-${i}`}
              value={getBlobAtRange(range)}
              language={language}
              stringToMatch={props.query}
              lineRange={range}
            />
          </>
        ))}
      </div>
    </div>
  )
}
