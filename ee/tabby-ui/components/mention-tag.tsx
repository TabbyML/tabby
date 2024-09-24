import React from 'react'
import { NodeViewProps, NodeViewWrapper } from '@tiptap/react'

import { ContextSourceKind } from '@/lib/gql/generates/graphql'
import { MentionAttributes } from '@/lib/types'
import { cn } from '@/lib/utils'
import {
  IconCode,
  IconEmojiBook,
  IconEmojiGlobe,
  IconGitHub,
  IconGitLab
} from '@/components/ui/icons'

export function Mention({
  kind,
  label,
  className
}: MentionAttributes & { className?: string }) {
  return (
    <NodeViewWrapper
      as={'span'}
      className={cn('source-mention rounded-sm px-1', className)}
    >
      <SourceIcon kind={kind} className="self-center" />
      <span>{label}</span>
    </NodeViewWrapper>
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
  kind: ContextSourceKind
  className?: string
}) {
  switch (kind) {
    case ContextSourceKind.Doc:
      return <IconEmojiBook {...rest} />
    case ContextSourceKind.Web:
      return <IconEmojiGlobe {...rest} />
    case ContextSourceKind.Git:
      return <IconCode {...rest} />
    case ContextSourceKind.Github:
      return <IconGitHub {...rest} />
    case ContextSourceKind.Gitlab:
      return <IconGitLab {...rest} />
    default:
      return null
  }
}
