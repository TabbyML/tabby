import React from 'react'
import { TooltipContent, TooltipTrigger } from '@radix-ui/react-tooltip'
import { NodeViewProps, NodeViewWrapper } from '@tiptap/react'

import { MentionAttributes } from '../lib/types'
import { isCodeSourceContext } from '../lib/utils'
import { IconCode, IconFileText } from './ui/icons'
import { Tooltip } from './ui/tooltip'

export function Mention({ kind, label }: MentionAttributes) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <NodeViewWrapper as={'span'}>
          <span className="source-mention gap-1 space-x-0.5 rounded-sm px-1 py-0 leading-7">
            {isCodeSourceContext(kind) ? (
              <IconCode className="-mt-px inline h-3.5 w-3.5" />
            ) : (
              <IconFileText className="-mt-px inline h-3.5 w-3.5" />
            )}
            <span className="text-base">{label}</span>
          </span>
        </NodeViewWrapper>
      </TooltipTrigger>
      <TooltipContent sideOffset={4}>
        <p className="rounded-md bg-popover px-3 py-1.5 text-popover-foreground">
          {label}
        </p>
      </TooltipContent>
    </Tooltip>
  )
}

export function MentionForNodeView(props: NodeViewProps) {
  const { kind, label, id } = props.node.attrs

  return <Mention kind={kind} label={label} id={id} />
}
