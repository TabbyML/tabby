'use client'

import React, { useMemo } from 'react'
import { NodeViewProps, NodeViewWrapper } from '@tiptap/react'

import { MARKDOWN_SOURCE_REGEX } from '@/lib/constants/regex'
import { ContextSource, ContextSourceKind } from '@/lib/gql/generates/graphql'
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
      <span className="whitespace-nowrap">{label}</span>
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

export function ThreadTitleWithMentions({
  message,
  sources,
  className
}: {
  sources: ContextSource[] | undefined
  message: string | undefined
  className?: string
}) {
  const contentWithTags = useMemo(() => {
    if (!message) return null

    const firstLine = message.split('\n')[0] ?? ''
    return firstLine.split(MARKDOWN_SOURCE_REGEX).map((part, index) => {
      if (index % 2 === 1) {
        const sourceId = part
        const source = sources?.find(s => s.sourceId === sourceId)
        if (source) {
          return (
            <Mention
              key={index}
              id={source.sourceId}
              kind={source.sourceKind}
              label={source.sourceName}
              className="rounded-md border border-[#b3ada0] border-opacity-30 bg-[#e8e1d3] py-[1px] text-sm dark:bg-[#333333]"
            />
          )
        } else {
          return null
        }
      }
      return part
    })
  }, [sources, message])

  return <div className={cn(className)}>{contentWithTags}</div>
}
