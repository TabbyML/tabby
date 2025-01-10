import { HTMLAttributes, useContext } from 'react'

import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { IconEdit, IconEmojiBook, IconGithub } from '@/components/ui/icons'
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
    <div>
      <div
        className={cn('flex items-center gap-2 font-semibold', className)}
        id={message.id}
        {...props}
      >
        {/* todo use markdown? */}
        <MessageMarkdown
          message={message.content}
          contextInfo={undefined}
          supportsOnApplyInEditorV2={supportsOnApplyInEditorV2}
          fetchingContextInfo={fetchingContextInfo}
          className="text-3xl prose-h2:text-foreground prose-p:mb-1 prose-p:mt-0"
          headline
          canWrapLongLines
        />
        {mode === 'edit' && (
          <Button variant="outline" className="px-2">
            <IconEdit className="h-6 w-6" />
          </Button>
        )}
      </div>
      <div className="flex items-center gap-2 mt-1 mb-4">
        <Badge variant="secondary">
          <IconGithub className="mr-1" />
          TabbyML/tabby
        </Badge>
        <Badge variant="secondary">
          <IconEmojiBook className="mr-1" />
          Tailwindcss
        </Badge>
        <Badge variant="secondary">
          <IconEmojiBook className="mr-1" />
          Pundit
        </Badge>
      </div>
    </div>
  )
}
