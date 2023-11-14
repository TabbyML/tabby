'use client'

import React from 'react'
import { cn } from '@/lib/utils'
import { useChatStore } from '@/lib/stores/chat-store'
import { addChat, deleteChat, setActiveChatId } from '@/lib/stores/chat-actions'
import { IconPlus, IconTrash } from '@/components/ui/icons'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'
import { EditChatTitleDialog } from '@/components/edit-chat-title-dialog'
import { useStore } from '@/lib/hooks/use-store'
import { Button } from '@/components/ui/button'
import { ListSkeleton } from '@/components/skeleton'

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

  return (
    <>
      <div className={cn(className)}>
        <div className="fixed bottom-0 left-0 top-16 flex w-[279px] flex-col gap-2 overflow-y-auto">
          <div className="bg-card p-2">
            <Button
              className="h-12 w-full"
              variant="ghost"
              onClick={e => addChat()}
            >
              <IconPlus />
              <span className="ml-2">New Chat</span>
            </Button>
          </div>
          <div className="flex flex-col gap-2 px-4">
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
                        'flex cursor-pointer items-center justify-between gap-3 rounded-lg px-3 py-2 text-zinc-900 transition-all hover:bg-zinc-200 hover:text-zinc-900  dark:text-zinc-50 hover:dark:bg-zinc-900 dark:hover:text-zinc-50',
                        isActive && '!bg-zinc-300 dark:!bg-zinc-800'
                      )}
                    >
                      <span className="leading-8">
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
          </div>
        </div>
      </div>
    </>
  )
}
