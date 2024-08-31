import { TooltipContent, TooltipTrigger } from '@radix-ui/react-tooltip'
import { NodeViewProps, NodeViewWrapper } from '@tiptap/react'

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
          <span className="source-mention gap-1 space-x-0.5 rounded-sm px-1 py-0 leading-7">
            {isRepositorySource(kind) ? (
              <IconCode className="-mt-px inline h-3.5 w-3.5" />
            ) : (
              <IconFileText className="-mt-px inline h-3.5 w-3.5" />
            )}
            <span className="text-base">{props.node.attrs.label}</span>
          </span>
        </NodeViewWrapper>
      </TooltipTrigger>
      <TooltipContent sideOffset={4}>
        <p className="rounded-md bg-popover px-3 py-1.5 text-popover-foreground">
          {props.node.attrs.label}
        </p>
      </TooltipContent>
    </Tooltip>
  )
}
