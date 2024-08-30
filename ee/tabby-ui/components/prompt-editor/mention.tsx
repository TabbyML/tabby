import { NodeViewProps, NodeViewWrapper } from '@tiptap/react'

import { cn } from '@/lib/utils'

import { IconCode, IconFileText } from '../ui/icons'
import { getInfoFromMentionId, isRepositorySource } from './utils'

export function Mention(props: NodeViewProps) {
  const id = props.node.attrs.id
  const { kind } = getInfoFromMentionId(id)

  // style={{ background: theme === 'dark' ? '#333' : '#e8e1d3' }}
  return (
    <NodeViewWrapper as={'span'}>
      {/* <span className="leading-7 gap-1 bg-blue-100 text-blue-600 dark:bg-blue-500 dark:text-blue-100 px-1 py-0 rounded-sm space-x-0.5"> */}
      <span className="source-mention leading-7 gap-1 px-1 py-0 rounded-sm space-x-0.5">
        {isRepositorySource(kind) ? (
          <IconCode className="inline w-3.5 h-3.5 -mt-px" />
        ) : (
          <IconFileText className="inline w-3.5 h-3.5 -mt-px" />
        )}
        <span className="text-base">{props.node.attrs.label}</span>
      </span>
    </NodeViewWrapper>
  )
}
