'use client'

import React from 'react'

import { useStore } from '@/lib/hooks/use-store'
import {
  clearChats,
  deleteChat,
  setActiveChatId
} from '@/lib/stores/chat-actions'
import { useChatStore } from '@/lib/stores/chat-store'
import { cn, nanoid } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { IconPlus, IconTrash } from '@/components/ui/icons'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'
import { ListSkeleton } from '@/components/skeleton'

import { ClearChatsButton } from './clear-chats-button'
import { EditChatTitleDialog } from './edit-chat-title-dialog'

interface ChatSessionsProps {
  className?: string
}

export const ChatSessions = ({ className }: ChatSessionsProps) => {
  const _hasHydrated = useStore(useChatStore, state => state._hasHydrated)
  const chats = useStore(useChatStore, state => state.chats)
  const activeChatId = useStore(useChatStore, state => state.activeChatId)

  const onDeleteClick = (
    e: React.MouseEvent<HTMLButtonElement>,
    chatId: string
  ) => {
    deleteChat(chatId)
  }

  const onNewChatClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    setActiveChatId(nanoid())
  }

  const handleClearChats = () => {
    clearChats()
  }

  return (
    <>
      <div className={cn(className)}>
        <div className="fixed inset-y-0 left-0 flex w-[279px] flex-col gap-2 pt-16">
          <div className="shrink-0 pb-0 pl-3 pt-2">
            <Button
              className="h-12 w-full justify-start"
              variant="ghost"
              onClick={onNewChatClick}
            >
              <IconPlus />
              <span className="ml-2">New Chat</span>
            </Button>
          </div>
          <Separator />
          <ScrollArea className="flex flex-col gap-2 px-3">
            {!_hasHydrated ? (
              <ListSkeleton />
            ) : (
              <>
                {chats?.map(chat => {
                  const isActive = activeChatId === chat.id
                  return (
                    <div
                      key={chat.id}
                      onClick={e => setActiveChatId(chat.id)}
                      className={cn(
                        'my-2 flex w-[254px] cursor-pointer items-center justify-between gap-3 rounded-lg px-3 py-1 text-sm transition-all hover:bg-primary/10',
                        isActive && 'bg-primary/10'
                      )}
                    >
                      <span className="truncate leading-8">
                        {chat.title || '(Untitled)'}
                      </span>
                      {isActive && (
                        <div
                          className="flex items-center"
                          onClick={e => e.stopPropagation()}
                        >
                          <EditChatTitleDialog
                            initialValue={chat.title}
                            chatId={chat.id}
                          />
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Button
                                variant="ghost"
                                size="icon"
                                onClick={e => onDeleteClick(e, chat.id)}
                              >
                                <IconTrash />
                                <span className="sr-only">Delete</span>
                              </Button>
                            </TooltipTrigger>
                            <TooltipContent side="bottom">
                              Delete
                            </TooltipContent>
                          </Tooltip>
                        </div>
                      )}
                    </div>
                  )
                })}
              </>
            )}
          </ScrollArea>
          <Separator />
          <div className="shrink-0 pb-2">
            <ClearChatsButton
              disabled={chats?.length === 0}
              onClear={handleClearChats}
            />
          </div>
        </div>
      </div>
    </>
  )
}
