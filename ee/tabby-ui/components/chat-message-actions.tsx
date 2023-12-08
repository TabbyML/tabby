'use client'

import { type Message } from 'ai'

import { MessageActionType } from '@/lib/types'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { IconEdit, IconRefresh, IconTrash } from '@/components/ui/icons'
import { CopyButton } from '@/components/copy-button'

interface ChatMessageActionsProps extends React.ComponentProps<'div'> {
  message: Message
  handleMessageAction: (messageId: string, action: MessageActionType) => void
}

export function ChatMessageActions({
  message,
  className,
  handleMessageAction,
  ...props
}: ChatMessageActionsProps) {
  return (
    <div
      className={cn(
        'flex items-center justify-end transition-opacity group-hover:opacity-100 md:absolute md:-right-[5rem] md:-top-2 md:opacity-0',
        className
      )}
      {...props}
    >
      {message.role === 'user' ? (
        <Button
          variant="ghost"
          size="icon"
          onClick={e => handleMessageAction(message.id, 'edit')}
        >
          <IconEdit />
          <span className="sr-only">Edit message</span>
        </Button>
      ) : (
        <Button
          variant="ghost"
          size="icon"
          onClick={e => handleMessageAction(message.id, 'regenerate')}
        >
          <IconRefresh />
          <span className="sr-only">Regenerate message</span>
        </Button>
      )}
      <Button
        variant="ghost"
        size="icon"
        onClick={e => handleMessageAction(message.id, 'delete')}
      >
        <IconTrash />
        <span className="sr-only">Delete message</span>
      </Button>
      <CopyButton value={message.content} />
    </div>
  )
}
