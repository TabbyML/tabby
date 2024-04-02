'use client'

import React from 'react'

import { updateChat } from '@/lib/stores/chat-actions'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { IconArrowElbow, IconEdit } from '@/components/ui/icons'
import { Input } from '@/components/ui/input'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'

interface EditChatTitleDialogProps {
  initialValue: string | undefined
  chatId: string
  children?: React.ReactNode
}

export const EditChatTitleDialog = ({
  children,
  initialValue,
  chatId
}: EditChatTitleDialogProps) => {
  const [open, setOpen] = React.useState(false)
  const formRef = React.useRef<HTMLFormElement>(null)
  const [input, setInput] = React.useState(initialValue)

  const handleSubmit: React.FormEventHandler<HTMLFormElement> = async e => {
    e.preventDefault()
    if (!input?.trim()) {
      return
    }
    updateChat(chatId, { title: input })
    setOpen(false)
  }

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      formRef.current?.requestSubmit()
      e.preventDefault()
    }
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button variant="ghost" size="icon" onClick={() => setOpen(true)}>
            <IconEdit />
            <span className="sr-only">Edit</span>
          </Button>
        </TooltipTrigger>
        <TooltipContent side="bottom">Edit</TooltipContent>
      </Tooltip>
      <DialogContent className="bg-background">
        <DialogHeader className="gap-3">
          <DialogTitle>Set Chat Title</DialogTitle>
          <DialogDescription asChild>
            <form className="relative" onSubmit={handleSubmit} ref={formRef}>
              <Input
                className="h-10 pr-12"
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={onKeyDown}
              />
              <div className="absolute right-2 top-1">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button type="submit" size="icon" disabled={input === ''}>
                      <IconArrowElbow />
                      <span className="sr-only">Send message</span>
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Edit Title</TooltipContent>
                </Tooltip>
              </div>
            </form>
          </DialogDescription>
        </DialogHeader>
      </DialogContent>
    </Dialog>
  )
}
