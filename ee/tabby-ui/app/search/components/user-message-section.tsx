import { HTMLAttributes, useContext } from 'react'

import { cn } from '@/lib/utils'
import { MessageMarkdown } from '@/components/message-markdown'

import { ConversationMessage, SearchContext } from './search'

interface QuestionBlockProps extends HTMLAttributes<HTMLDivElement> {
  message: ConversationMessage
}

export function UserMessageSection({
  message,
  className,
  ...props
}: QuestionBlockProps) {
  const { contextInfo, fetchingContextInfo } = useContext(SearchContext)

  return (
    <div className={cn('font-semibold', className)} {...props}>
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
