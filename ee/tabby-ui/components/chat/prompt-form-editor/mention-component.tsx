/* eslint-disable no-console */
// mention-component.tsx
import React from 'react'
import { NodeViewWrapper } from '@tiptap/react'

import { cn } from '@/lib/utils'

export const PromptFormMentionComponent = ({ node }: { node: any }) => {
  console.log('mention comp:' + JSON.stringify(node.attrs))
  return (
    <NodeViewWrapper className="source-mention">
      <span
        className={cn(
          'inline-flex items-center rounded bg-muted px-1.5 py-0.5 text-sm font-medium',
          'ring-1 ring-inset ring-muted'
        )}
        data-category={node.attrs.category}
      >
        {node.attrs.category === 'file' ? (
          <svg
            className="mr-1 h-3 w-3 text-muted-foreground"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" />
            <polyline points="13 2 13 9 20 9" />
          </svg>
        ) : (
          <span className="mr-1">＠</span>
        )}
        <span>{node.attrs.name}</span>
      </span>
    </NodeViewWrapper>
  )
}
