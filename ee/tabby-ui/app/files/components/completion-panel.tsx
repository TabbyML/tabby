import React from 'react'
import Image from 'next/image'
import tabbyLogo from '@/assets/tabby.png'
import { Message } from 'ai'

import { useStore } from '@/lib/hooks/use-store'
import {
  addChat,
  clearChats,
  deleteChat,
  setActiveChatId
} from '@/lib/stores/chat-actions'
import { useChatStore } from '@/lib/stores/chat-store'
import { getChatById } from '@/lib/stores/utils'
import { cn, nanoid, truncateText } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  IconClose,
  IconHistory,
  IconPlus,
  IconTrash
} from '@/components/ui/icons'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'
import { Chat, ChatRef } from '@/components/chat'
import { ClearChatsButton } from '@/components/clear-chats-button'
import { EditChatTitleDialog } from '@/components/edit-chat-title-dialog'
import LoadingWrapper from '@/components/loading-wrapper'
import { ListSkeleton } from '@/components/skeleton'

import { CodeBrowserQuickAction } from '../lib/event-emitter'
import { SourceCodeBrowserContext } from './source-code-browser'

interface CompletionPanelProps
  extends Omit<React.HTMLAttributes<HTMLDivElement>, 'children'> {}

enum CompletionPanelView {
  CHAT,
  SESSIONS
}

export const CompletionPanel: React.FC<CompletionPanelProps> = ({
  className,
  ...props
}) => {
  const {
    completionPanelViewType,
    pendingEvent,
    setPendingEvent,
    setCompletionPanelViewType
  } = React.useContext(SourceCodeBrowserContext)
  const chats = useStore(useChatStore, state => state.chats)
  const activeChatId = useStore(useChatStore, state => state.activeChatId)
  const chatId = activeChatId
  const chat = getChatById(chats, chatId)
  const appending = React.useRef(false)
  const iframeRef = React.useRef<HTMLIFrameElement>(null)

  const getPrompt = ({
    action,
    payload
  }: {
    action: CodeBrowserQuickAction
    payload: string
  }) => {
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
    return `${builtInPrompt}\n${'```'}\n${payload}\n${'```'}\n`
  }

  React.useEffect(() => {
    const contentWindow = iframeRef.current?.contentWindow

    if (pendingEvent) {
      setCompletionPanelViewType(CompletionPanelView.CHAT)

      contentWindow?.postMessage({
        action: 'append',
        payload: getPrompt(pendingEvent)
      })
      setPendingEvent(undefined)
    }
  }, [pendingEvent, iframeRef.current?.contentWindow])

  return (
    <div className={cn('h-full flex flex-col', className)} {...props}>
      <Header />
      {completionPanelViewType === CompletionPanelView.CHAT && (
        <iframe
          src={`/playground`}
          className="border-0 w-full flex-1"
          key={activeChatId}
          ref={iframeRef}
        />
      )}
      {completionPanelViewType === CompletionPanelView.SESSIONS && (
        <ChatSessions />
      )}
    </div>
  )
}

function ChatSessions({ className }: { className?: string }) {
  const _hasHydrated = useStore(useChatStore, state => state._hasHydrated)
  const chats = useStore(useChatStore, state => state.chats)
  const activeChatId = useStore(useChatStore, state => state.activeChatId)
  const { setCompletionPanelViewType } = React.useContext(
    SourceCodeBrowserContext
  )

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
    setCompletionPanelViewType(CompletionPanelView.CHAT)
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
  const { setCompletionPanelVisible } = React.useContext(
    SourceCodeBrowserContext
  )

  // const onToggleChatHistory = () => {
  //   setCompletionPanelViewType(
  //     completionPanelViewType === CompletionPanelView.CHAT
  //       ? CompletionPanelView.SESSIONS
  //       : CompletionPanelView.CHAT
  //   )
  // }

  // const onNewChatClick = (e: React.MouseEvent<HTMLButtonElement>) => {
  //   setActiveChatId(nanoid())
  //   if (completionPanelViewType === CompletionPanelView.SESSIONS) {
  //     setCompletionPanelViewType(CompletionPanelView.CHAT)
  //   }
  // }

  return (
    <div className="flex items-center justify-between bg-secondary px-2 py-1 sticky top-0">
      {/* <div className="flex items-center gap-2">
        <Tooltip>
          <TooltipTrigger asChild>
            <Button size="icon" variant="ghost" onClick={onToggleChatHistory}>
              <IconHistory />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">Chat history</TooltipContent>
        </Tooltip>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button size="icon" variant="ghost" onClick={onNewChatClick}>
              <IconPlus />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">Start a new chat</TooltipContent>
        </Tooltip>
      </div> */}
      <div className="w-8"></div>
      <Image src={tabbyLogo} alt="logo" width={32} />
      <Button
        size="icon"
        variant="ghost"
        onClick={e => setCompletionPanelVisible(false)}
      >
        <IconClose />
      </Button>
    </div>
  )
}
