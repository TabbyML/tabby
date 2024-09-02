import React from 'react'
import { TooltipContent, TooltipTrigger } from '@radix-ui/react-tooltip'
import { NodeViewProps, NodeViewWrapper } from '@tiptap/react'

import { ContextKind } from '@/lib/gql/generates/graphql'
import { MentionAttributes } from '@/lib/types'

import {
  IconCode,
  IconFileText,
  IconGitHub,
  IconGitLab,
  IconGlobe
} from './ui/icons'
import { Tooltip } from './ui/tooltip'

export function Mention({ kind, label }: MentionAttributes) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <NodeViewWrapper as={'span'}>
          <span className="source-mention rounded-sm px-1 py-0">
            <SourceIcon kind={kind} className="inline h-3.5 w-3.5" />
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

function SourceIcon({
  kind,
  ...rest
}: {
  kind: ContextKind
  className?: string
}) {
  switch (kind) {
    case ContextKind.Doc:
      return <IconFileText {...rest} />
    case ContextKind.Web:
      return <IconGlobe {...rest} />
    case ContextKind.Git:
      return <IconCode {...rest} />
    case ContextKind.Github:
      return <IconGitHub {...rest} />
    case ContextKind.Gitlab:
      return <IconGitLab {...rest} />
    default:
      return null
  }
}
