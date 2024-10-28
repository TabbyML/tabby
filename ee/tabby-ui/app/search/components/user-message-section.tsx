import { HTMLAttributes, useContext, useMemo, useRef, useState } from 'react'

import { MARKDOWN_SOURCE_REGEX } from '@/lib/constants/regex'
import { ContextInfo } from '@/lib/gql/generates/graphql'
import { cn, getMentionsFromText } from '@/lib/utils'
import { MessageMarkdown } from '@/components/message-markdown'
import { PromptEditor, PromptEditorRef } from '@/components/prompt-editor'

import { ConversationMessage, SearchContext } from './search'
import { ThreadMessagePairContext } from './thread-message-pair'

interface QuestionBlockProps extends HTMLAttributes<HTMLDivElement> {
  message: ConversationMessage
  isEditing: boolean
}

export function UserMessageSection({ message, isEditing }: QuestionBlockProps) {
  const { contextInfo, fetchingContextInfo } = useContext(SearchContext)
  const { setDraftUserMessage } = useContext(ThreadMessagePairContext)
  const editorRef = useRef<PromptEditorRef>(null)
  const [isFocus, setIsFocus] = useState(false)

  const JsonContent = useMemo(() => {
    if (!isEditing) return undefined

    return formatToTiptapJson(message.content, contextInfo)
  }, [isEditing, message.content])

  if (isEditing) {
    return (
      <div
        className={cn(
          'relative w-full overflow-hidden rounded-xl border hover:outline dark:border-muted-foreground/60',
          {
            outline: isFocus
          }
        )}
        onClick={() => {
          editorRef.current?.editor?.commands.focus()
        }}
      >
        <div className={cn('flex items-end px-4 min-h-[5.5rem]')}>
          <PromptEditor
            editable
            contextInfo={contextInfo}
            fetchingContextInfo={fetchingContextInfo}
            placeholder={'Edit message'}
            autoFocus={false}
            onFocus={() => setIsFocus(true)}
            onBlur={() => setIsFocus(false)}
            ref={editorRef}
            content={JsonContent}
            className={cn(
              'text-area-autosize mr-1 flex-1 resize-none rounded-lg !border-none bg-transparent !shadow-none !outline-none !ring-0 !ring-offset-0 py-3'
            )}
            editorClassName="min-h-[3.5em]"
            onUpdate={({ editor }) => {
              setDraftUserMessage(d => ({
                ...d,
                content: editor.getText().trim()
              }))
            }}
          />
        </div>
      </div>
    )
  }

  return (
    <div className="font-semibold">
      <MessageMarkdown
        message={message.content}
        contextInfo={contextInfo}
        fetchingContextInfo={fetchingContextInfo}
        className="text-xl prose-p:mb-2 prose-p:mt-0"
        headline
        canWrapLongLines
      />
    </div>
  )
}

interface MentionNode {
  type: 'mention'
  attrs: {
    id: string
    label: string
    kind: string
  }
}

interface ParagraphNode {
  type: 'paragraph'
  content: (TextNode | MentionNode)[]
}

interface TextNode {
  type: 'text'
  text: string
}

interface TiptapJsonContent {
  type: 'doc'
  content: ParagraphNode[]
}

function formatToTiptapJson(
  text: string,
  contextInfo?: ContextInfo
): TiptapJsonContent {
  const lines = text.split('\n')

  const mentions = getMentionsFromText(text, contextInfo?.sources)
  const content: ParagraphNode[] = lines.map(line => {
    const nodes: (TextNode | MentionNode)[] = []
    let lastIndex = 0

    line.replace(MARKDOWN_SOURCE_REGEX, (match, sourceId, offset) => {
      if (offset > lastIndex) {
        nodes.push({
          type: 'text',
          text: line.slice(lastIndex, offset)
        })
      }

      const targetSource = mentions.find(o => o.id === sourceId)

      if (targetSource) {
        nodes.push({
          type: 'mention',
          attrs: {
            id: sourceId,
            label: targetSource.label,
            kind: targetSource.kind
          }
        })
      } else {
        nodes.push({
          type: 'text',
          text: match
        })
      }

      lastIndex = offset + match.length
      return match
    })

    if (lastIndex < line.length) {
      nodes.push({
        type: 'text',
        text: line.slice(lastIndex)
      })
    }

    return {
      type: 'paragraph',
      content: nodes
    }
  })

  return {
    type: 'doc',
    content
  }
}
