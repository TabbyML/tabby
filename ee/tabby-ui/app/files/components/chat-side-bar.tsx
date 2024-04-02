import React from 'react'

import { useStore } from '@/lib/hooks/use-store'
import {
  clearChats,
  deleteChat,
  setActiveChatId
} from '@/lib/stores/chat-actions'
import { useChatStore } from '@/lib/stores/chat-store'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { IconClose, IconTrash } from '@/components/ui/icons'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'
import { ClearChatsButton } from '@/components/clear-chats-button'
import { EditChatTitleDialog } from '@/components/edit-chat-title-dialog'
import { ListSkeleton } from '@/components/skeleton'

import { QuickActionEventPayload } from '../lib/event-emitter'
import { SourceCodeBrowserContext } from './source-code-browser'

interface ChatSideBarProps
  extends Omit<React.HTMLAttributes<HTMLDivElement>, 'children'> {}

export const ChatSideBar: React.FC<ChatSideBarProps> = ({
  className,
  ...props
}) => {
  const { pendingEvent, setPendingEvent } = React.useContext(
    SourceCodeBrowserContext
  )
  const activeChatId = useStore(useChatStore, state => state.activeChatId)
  const iframeRef = React.useRef<HTMLIFrameElement>(null)

  const getPrompt = ({ action, code, language }: QuickActionEventPayload) => {
    let builtInPrompt = ''
    switch (action) {
      case 'explain':
        builtInPrompt = 'Explain the following code:'
        break
      case 'generate_unittest':
        builtInPrompt = 'Generate a unit test for the following code:'
        break
      case 'generate_doc':
        builtInPrompt = 'Generate documentation for the following code:'
        break
      default:
        break
    }
    return `${builtInPrompt}\n${'```'}${language ?? ''}\n${code}\n${'```'}\n`
  }

  React.useEffect(() => {
    const contentWindow = iframeRef.current?.contentWindow

    if (pendingEvent) {
      contentWindow?.postMessage({
        action: 'append',
        payload: getPrompt(pendingEvent)
      })
      setPendingEvent(undefined)
    }
  }, [pendingEvent, iframeRef.current?.contentWindow])

  return (
    <div className={cn('flex h-full flex-col', className)} {...props}>
      <Header />
      <iframe
        src={`/playground`}
        className="w-full flex-1 border-0"
        key={activeChatId}
        ref={iframeRef}
      />
    </div>
  )
}

function ChatSessions({ className }: { className?: string }) {
  const _hasHydrated = useStore(useChatStore, state => state._hasHydrated)
  const chats = useStore(useChatStore, state => state.chats)
  const activeChatId = useStore(useChatStore, state => state.activeChatId)

  const onDeleteClick = (
    e: React.MouseEvent<HTMLButtonElement>,
    chatId: string
  ) => {
    deleteChat(chatId)
  }

  const handleClearChats = () => {
    clearChats()
  }

  const onSelectChat = (id: string) => {
    setActiveChatId(id)
  }

  return (
    <div className={cn('px-4', className)}>
      <div>
        {!_hasHydrated ? (
          <ListSkeleton />
        ) : (
          <>
            {chats?.map(chat => {
              const isActive = activeChatId === chat.id
              return (
                <div
                  key={chat.id}
                  onClick={e => onSelectChat(chat.id)}
                  className={cn(
                    'group my-2 flex cursor-pointer items-center justify-between gap-3 rounded-lg px-3 py-1 text-sm transition-all hover:bg-primary/10',
                    isActive && 'bg-primary/10'
                  )}
                >
                  <span className="truncate leading-8">
                    {chat.title || '(Untitled)'}
                  </span>
                  <div
                    className={cn(
                      'hidden items-center group-hover:flex',
                      isActive && 'flex'
                    )}
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
                      <TooltipContent side="bottom">Delete</TooltipContent>
                    </Tooltip>
                  </div>
                </div>
              )
            })}
          </>
        )}
      </div>
      <ClearChatsButton
        disabled={chats?.length === 0}
        onClear={handleClearChats}
        className="justify-center"
      />
    </div>
  )
}

function Header() {
  const { setChatSideBarVisible } = React.useContext(SourceCodeBrowserContext)

  return (
    <div className="sticky top-0 flex items-center justify-end bg-secondary px-2 py-1">
      <Button
        size="icon"
        variant="ghost"
        onClick={e => setChatSideBarVisible(false)}
      >
        <IconClose />
      </Button>
    </div>
  )
}
