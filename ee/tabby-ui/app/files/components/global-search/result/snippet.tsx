'use client'

import React from 'react'
import Link from 'next/link'

import { GrepSubMatch } from '@/lib/gql/generates/graphql'

import { HighlightMatches } from '../../file-tree-header'
import { SourceCodeBrowserContext } from '../../source-code-browser'
import { generateEntryPath } from '../../utils'

interface GlobalSearchResultSnippetProps {
  lineNumber: number
  subMatches: GrepSubMatch[]
  text: string
  path: string
}

export const GlobalSearchResultSnippet = ({
  ...props
}: GlobalSearchResultSnippetProps) => {
  const { activeRepo, activeRepoRef, currentFileRoutes } = React.useContext(
    SourceCodeBrowserContext
  )
  const lines = props.text.split('\n')

  const whitespaceCharacterCount = lines[props.lineNumber - 1]?.search(/\S/)

  // Remove whitespace since we don't care about the indentation context
  const snippet = lines[props.lineNumber - 1]?.trim()

  const indices = Array.from(
    { length: props.subMatches[0].bytesEnd - props.subMatches[0].bytesStart },
    (_, i) => i + props.subMatches[0].bytesStart - whitespaceCharacterCount
  )

  return (
    <Link
      href={{
        pathname: `/files/${generateEntryPath(
          activeRepo,
          activeRepoRef?.name as string,
          props.path,
          'file'
        )}`,
        // FIXME: this doesn't work when clicking a different line on the active file
        hash: `L${props.lineNumber}`
      }}
      className="font-mono py-2 px-2.5 text-sm truncate block text-muted-foreground"
    >
      {/* {indices} */}
      {/* TODO: these need line numbers now */}
      <HighlightMatches text={snippet} indices={indices} />
    </Link>
  )
}
