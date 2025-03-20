import { HTMLAttributes, useContext } from 'react'

import { cn } from '@/lib/utils'
import { processingContextCommand } from '@/lib/utils/markdown'
import { CollapsibleContainer } from '@/components/collapsible-container'
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
  message.content = processingContextCommand(message.content)
  const { contextInfo, fetchingContextInfo } = useContext(SearchContext)
  return (
    <div className={cn('font-semibold', className)} {...props}>
      <CollapsibleContainer>
        <MessageMarkdown
          message={message.content}
          contextInfo={contextInfo}
          supportsOnApplyInEditorV2={false}
          fetchingContextInfo={fetchingContextInfo}
          className="text-xl prose-p:mb-2 prose-p:mt-0"
          headline
          canWrapLongLines
        />
      </CollapsibleContainer>
    </div>
  )
}
