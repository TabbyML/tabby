'use client'

import React, { useEffect, useState } from 'react'
import Link from 'next/link'
import { lineNumbers } from '@codemirror/view'

import { GrepFile, GrepLine } from '@/lib/gql/generates/graphql'
import { Skeleton } from '@/components/ui/skeleton'
import CodeEditor from '@/components/codemirror/codemirror'

import { SourceCodeBrowserContext } from '../source-code-browser'
import { generateEntryPath, resolveRepositoryInfoFromPath } from '../utils'

interface GlobalSearchSnippetProps {
  line: GrepLine
  blobText: string
  repoId: string
  file: GrepFile
  hidePopover: () => void
}

export const GlobalSearchSnippet = ({ ...props }: GlobalSearchSnippetProps) => {
  const { activeRepo, activeRepoRef, currentFileRoutes } = React.useContext(
    SourceCodeBrowserContext
  )

  const [snippet, setSnippet] = useState<string | undefined>(undefined)

  useEffect(() => {
    if (props.blobText) {
      const lines = props.blobText.split('\n')
      const line = lines[props.line.lineNumber - 1]

      // Remove whitespace since we don't care about the indentation context
      setSnippet(line?.trim())
    }
  }, [props.blobText, props.line, snippet])

  return (
    <>
      {snippet && (
        <Link
          href={{
            pathname: `/files/${generateEntryPath(
              activeRepo,
              activeRepoRef?.name as string,
              props.file.path,
              'file'
            )}`,
            // FIXME: this doesn't work when clicking a different line on the active file
            hash: `L${props.line.lineNumber}`
          }}
          className="font-mono text-sm truncate block text-muted-foreground"
          onClick={props.hidePopover}
        >
          {snippet}
          {/* <CodeEditor
            value={snippet}
            // FIXME: pass language in
            language="plain"
            extensions={[
              lineNumbers({
                formatNumber: () => props.line.lineNumber.toString()
              })
            ]}
            readonly
          /> */}
        </Link>
      )}
    </>
  )
}
