'use client'

import React from 'react'
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

  const linesWithSubMatches = props.result.lines.filter(
    line => line.subMatches.length > 0
  )

  const language = filename2prism(props.result.path)[0]

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
      <ol className="overflow-hidden grid border divide-y divide-y-border border-border rounded">
        {linesWithSubMatches.map((line, i) => {
          return (
            // FIXME: key should be unique
            <li key={i}>
              <CodeEditorView
                value={props.result.blob}
                language={language}
                stringToMatch={props.query}
              />
            </li>
          )
        })}
      </ol>
    </div>
  )
}
