'use client'

import React from 'react'
import Link from 'next/link'

import { SourceCodeBrowserContext } from '../../source-code-browser'
import { generateEntryPath } from '../../utils'

interface GlobalSearchResultSnippetProps {
  lineNumber: number
  text: string
  path: string
  hidePopover: () => void
}

export const GlobalSearchResultSnippet = ({
  ...props
}: GlobalSearchResultSnippetProps) => {
  const { activeRepo, activeRepoRef, currentFileRoutes } = React.useContext(
    SourceCodeBrowserContext
  )

  const lines = props.text.split('\n')

  // Remove whitespace since we don't care about the indentation context
  const snippet = lines[props.lineNumber - 1]?.trim()

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
      className="font-mono text-sm truncate block text-muted-foreground"
      onClick={props.hidePopover}
    >
      {snippet}
    </Link>
  )
}
