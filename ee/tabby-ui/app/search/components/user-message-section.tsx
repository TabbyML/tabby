import { HTMLAttributes, useContext, useMemo } from 'react'

import { cn } from '@/lib/utils'
import { convertContextBlockToPlaceholder } from '@/lib/utils/markdown'
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
  const processedContent = useMemo(() => {
    return convertContextBlockToPlaceholder(message.content)
  }, [message.content])

  const { contextInfo, fetchingContextInfo } = useContext(SearchContext)
  return (
    <div className={cn('font-semibold', className)} {...props}>
      <CollapsibleContainer>
        <MessageMarkdown
          message={processedContent}
          contextInfo={contextInfo}
          supportsOnApplyInEditorV2={false}
          fetchingContextInfo={fetchingContextInfo}
          className="text-xl prose-p:mb-2 prose-p:mt-0"
          headline
        />
      </CollapsibleContainer>
    </div>
  )
}
