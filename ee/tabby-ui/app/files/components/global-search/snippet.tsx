'use client'

import React, { useEffect, useState } from 'react'
import { highlightSelectionMatches } from '@codemirror/search'
import { EditorState } from '@codemirror/state'
import { lineNumbers } from '@codemirror/view'

import { GrepLine } from '@/lib/gql/generates/graphql'
import CodeEditor from '@/components/codemirror/codemirror'

interface GlobalSearchSnippetProps {
  line: GrepLine
  blobText: string
}

export const GlobalSearchSnippet = ({ ...props }: GlobalSearchSnippetProps) => {
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
      {snippet ? (
        <CodeEditor
          value={snippet}
          // FIXME: pass language in
          language="plain"
          extensions={[
            lineNumbers({
              formatNumber: () => props.line.lineNumber.toString()
            })
          ]}
          readonly
        />
      ) : (
        <div>Loading</div>
      )}
    </>
  )
}
