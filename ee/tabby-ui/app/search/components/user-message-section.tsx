import { HTMLAttributes, useContext } from 'react'

import { cn } from '@/lib/utils'
import { ChatContext } from '@/components/chat/chat'
import { MessageMarkdown } from '@/components/message-markdown'

import { SearchContext } from './search-context'
import { ConversationMessage } from './types'

interface QuestionBlockProps extends HTMLAttributes<HTMLDivElement> {
  message: ConversationMessage
}

export function UserMessageSection({
  message,
  className,
  ...props
}: QuestionBlockProps) {
  const { contextInfo, fetchingContextInfo } = useContext(SearchContext)
  const { supportsOnApplyInEditorV2 } = useContext(ChatContext)
  const contentLen = message.content?.length
  const fontSizeClassname =
    contentLen > 400 ? 'text-base' : contentLen > 200 ? 'text-lg' : 'text-xl'

  return (
    <div className={cn('font-semibold', className)} {...props}>
      <MessageMarkdown
        message={message.content}
        contextInfo={contextInfo}
        supportsOnApplyInEditorV2={supportsOnApplyInEditorV2}
        fetchingContextInfo={fetchingContextInfo}
        className={cn('prose-p:mb-2 prose-p:mt-0', fontSizeClassname)}
        headline
        canWrapLongLines
      />
    </div>
  )
}
