import { HTMLAttributes, useContext } from 'react'

import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { IconEdit } from '@/components/ui/icons'
import { ChatContext } from '@/components/chat/chat'
import { MessageMarkdown } from '@/components/message-markdown'

import { ConversationMessage, PageContext } from './page'

interface QuestionBlockProps extends HTMLAttributes<HTMLDivElement> {
  message: ConversationMessage
}

export function SectionTitle({
  message,
  className,
  ...props
}: QuestionBlockProps) {
  const { fetchingContextInfo, mode } = useContext(PageContext)
  const { supportsOnApplyInEditorV2 } = useContext(ChatContext)
  return (
    <div
      className={cn('font-semibold flex items-center gap-2', className)}
      id={message.id}
      {...props}
    >
      <MessageMarkdown
        message={message.content}
        contextInfo={undefined}
        supportsOnApplyInEditorV2={supportsOnApplyInEditorV2}
        fetchingContextInfo={fetchingContextInfo}
        className="text-3xl prose-p:mb-1 prose-p:mt-0 prose-h2:text-foreground"
        headline
        canWrapLongLines
      />
      {mode === 'edit' && (
        <Button variant="outline" className="px-2">
          <IconEdit className="h-6 w-6" />
        </Button>
      )}
    </div>
  )
}
