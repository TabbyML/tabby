'use client'

import React, { useMemo } from 'react'
import { NodeViewProps, NodeViewWrapper } from '@tiptap/react'
import { Filepath } from 'tabby-chat-panel/index'

import {
  MARKDOWN_COMMAND_REGEX,
  MARKDOWN_FILE_REGEX,
  MARKDOWN_SOURCE_REGEX,
  MARKDOWN_SYMBOL_REGEX
} from '@/lib/constants/regex'
import { ContextSource, ContextSourceKind } from '@/lib/gql/generates/graphql'
import { MentionAttributes } from '@/lib/types'
import { cn, resolveFileNameForDisplay } from '@/lib/utils'
import { convertContextBlockToPlaceholder } from '@/lib/utils/markdown'
import {
  IconCode,
  IconEmojiBook,
  IconEmojiGlobe,
  IconFolderUp,
  IconGitHub,
  IconGitLab
} from '@/components/ui/icons'

import { getFilepathStringByChatPanelFilePath } from './chat/form-editor/utils'

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
    case ContextSourceKind.Ingested:
      return <IconFolderUp {...rest} />
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
    let processedMessage = convertContextBlockToPlaceholder(message)
    const partsWithSources = processedMessage
      .split(MARKDOWN_SOURCE_REGEX)
      .map((part, index) => {
        if (index % 2 === 1) {
          const sourceId = part.replace(/\\/g, '')
          const source = sources?.find(s => s.sourceId === sourceId)
          if (source) {
            return (
              <Mention
                key={`source-${index}`}
                id={source.sourceId}
                kind={source.sourceKind}
                label={source.sourceName}
                className="rounded-md border border-[#b3ada0] border-opacity-30 bg-[#e8e1d3] py-[1px] text-sm dark:bg-[#333333]"
              />
            )
          }
          return null
        }
        return part
      })
    const finalContent = partsWithSources.map((part, index) => {
      if (!part || React.isValidElement(part)) {
        return part
      }
      let textPart = part as string

      textPart = textPart.replace(MARKDOWN_FILE_REGEX, (match, content) => {
        try {
          if (content.startsWith('{') && content.endsWith('}')) {
            const fileInfo = JSON.parse(content) as Filepath
            const filepathString =
              getFilepathStringByChatPanelFilePath(fileInfo)
            const filename = resolveFileNameForDisplay(filepathString)
            return `@${filename}`
          }
          // Otherwise just use the content as is
          return content
        } catch (e) {
          // If parse fails, return original
          return match
        }
      })

      textPart = textPart.replace(MARKDOWN_SYMBOL_REGEX, (match, content) => {
        try {
          if (content.startsWith('{') && content.endsWith('}')) {
            const symbolInfo = JSON.parse(content)
            if (symbolInfo.label) {
              return `@${symbolInfo.label}`
            }
            const filepathString = getFilepathStringByChatPanelFilePath(
              symbolInfo.filepath
            )
            const filename = resolveFileNameForDisplay(filepathString)
            const range = symbolInfo.range
              ? `:${symbolInfo.range.start}-${symbolInfo.range.end}`
              : ''
            return `@${filename}${range}`
          }
          return content
        } catch (e) {
          return match
        }
      })

      return textPart.replace(MARKDOWN_COMMAND_REGEX, (_, cmdPart) => {
        return `@${cmdPart.replace(/"/g, '')}`
      })
    })

    return finalContent
  }, [sources, message])
  return <div className={cn(className)}>{contentWithTags}</div>
}
