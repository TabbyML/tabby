'use client'

import React from 'react'
import Link from 'next/link'

import { GrepFile } from '@/lib/gql/generates/graphql'

import { SourceCodeBrowserContext } from '../source-code-browser'
import { generateEntryPath } from '../utils'
import { GlobalSearchResultSnippet } from './result/snippet'

export interface GlobalSearchResult extends GrepFile {
  blob: string
}

interface GlobalSearchResultProps {
  result: GlobalSearchResult
  key: number
  hidePopover: () => void
}

export const GlobalSearchResult = ({ ...props }: GlobalSearchResultProps) => {
  const { activeRepo, activeRepoRef } = React.useContext(
    SourceCodeBrowserContext
  )

  const linesWithSubMatches = props.result.lines.filter(
    line => line.subMatches.length > 0
  )

  return (
    <div>
      <Link
        href={`/files/${generateEntryPath(
          activeRepo,
          activeRepoRef?.name as string,
          props.result.path,
          'file'
        )}`}
        className="text-sm font-semibold mb-2"
        // TODO: Investigate an alternative to this
        onClick={props.hidePopover}
      >
        {props.result.path}
      </Link>
      <ol className="overflow-hidden grid gap-0.5">
        {linesWithSubMatches.map((line, i) => {
          return (
            // TODO: Replace with a `SearchableSelectItem` component
            <GlobalSearchResultSnippet
              key={i}
              text={props.result.blob}
              path={props.result.path}
              lineNumber={line.lineNumber}
              hidePopover={props.hidePopover}
              subMatches={line.subMatches}
            />
          )
        })}
      </ol>
    </div>
  )
}
