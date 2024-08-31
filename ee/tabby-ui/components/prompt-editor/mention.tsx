import { TooltipContent, TooltipTrigger } from '@radix-ui/react-tooltip'
import { NodeViewProps, NodeViewWrapper } from '@tiptap/react'

import { cn } from '@/lib/utils'

import { IconCode, IconFileText } from '../ui/icons'
import { Tooltip } from '../ui/tooltip'
import { getInfoFromMentionId, isRepositorySource } from './utils'

export function Mention(props: NodeViewProps) {
  const id = props.node.attrs.id
  const { kind } = getInfoFromMentionId(id)

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <NodeViewWrapper as={'span'}>
          <span className="source-mention leading-7 gap-1 px-1 py-0 rounded-sm space-x-0.5">
            {isRepositorySource(kind) ? (
              <IconCode className="inline w-3.5 h-3.5 -mt-px" />
            ) : (
              <IconFileText className="inline w-3.5 h-3.5 -mt-px" />
            )}
            <span className="text-base">{props.node.attrs.label}</span>
          </span>
        </NodeViewWrapper>
      </TooltipTrigger>
      <TooltipContent sideOffset={4}>
        <p className="bg-popover text-popover-foreground px-3 py-1.5 rounded-md">
          {props.node.attrs.label}
        </p>
      </TooltipContent>
    </Tooltip>
  )
}
