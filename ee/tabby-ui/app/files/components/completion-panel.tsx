import React from 'react'
import Image from 'next/image'
import tabbyLogo from '@/assets/tabby.png'
import { Message } from 'ai'

import { useStore } from '@/lib/hooks/use-store'
import {
  clearChats,
  deleteChat,
  setActiveChatId
} from '@/lib/stores/chat-actions'
import { useChatStore } from '@/lib/stores/chat-store'
import { getChatById } from '@/lib/stores/utils'
import { cn, nanoid } from '@/lib/utils'
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
import { ListSkeleton } from '@/components/skeleton'

import { SourceCodeBrowserContext } from './source-code-browser'
import LoadingWrapper from '@/components/loading-wrapper'
import { CodeBrowserQuickAction } from '../lib/event-emitter'

interface CompletionPanelProps
  extends Omit<React.HTMLAttributes<HTMLDivElement>, 'children'> {
  // open: boolean
  // onOpenChange: (v: boolean) => void
}

const emptyMessages: Message[] = []

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
  const _hasHydrated = useStore(useChatStore, state => state._hasHydrated)
  const chats = useStore(useChatStore, state => state.chats)
  const activeChatId = useStore(useChatStore, state => state.activeChatId)
  const chatId = activeChatId
  const chat = getChatById(chats, chatId)
  const chatRef = React.useRef<ChatRef>(null)
  const appending = React.useRef(false)

  const quickActionBarCallback = (action: CodeBrowserQuickAction) => {
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
    const view = editorRef.current?.editorView
    const text =
      view?.state.doc.sliceString(
        view?.state.selection.main.from,
        view?.state.selection.main.to
      ) || ''

    const initialMessage = `${builtInPrompt}\n${'```'}${
      language ?? ''
    }\n${text}\n${'```'}\n`
    if (initialMessage) {
      window.open(
        `/playground?initialMessage=${encodeURIComponent(initialMessage)}`
      )
    }
  }

  React.useEffect(() => {
    if (chatRef.current && pendingEvent) {
      // debugger
      setCompletionPanelViewType(CompletionPanelView.CHAT)
      if (!appending.current) {
        console.log('call event cb====', pendingEvent)
        appending.current = true
        chatRef.current
          ?.append({
            role: 'user',
            content: pendingEvent['payload']
          })
          .then(() => {
            setPendingEvent(undefined)
            appending.current = false
          })
      }
    }

    // return () => {
    //   chatRef.current?.stop()
    // }
  }, [pendingEvent])

  return (
    <div className={cn('h-full overflow relative', className)} {...props}>
      <Header />
      {completionPanelViewType === CompletionPanelView.CHAT && (
        <LoadingWrapper loading={!activeChatId}>
          <Chat
            id={activeChatId}
            // loading={_hasHydrated}
            chatPanelClassName="w-full bottom-0 absolute lg:ml-0"
            initialMessages={chat?.messages ?? emptyMessages}
            key={activeChatId}
            ref={chatRef}
          />
        </LoadingWrapper>
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
  const {
    completionPanelViewType,
    setCompletionPanelViewType,
    setCompletionPanelVisible
  } = React.useContext(SourceCodeBrowserContext)

  const onToggleChatHistory = () => {
    setCompletionPanelViewType(
      completionPanelViewType === CompletionPanelView.CHAT
        ? CompletionPanelView.SESSIONS
        : CompletionPanelView.CHAT
    )
  }

  const onNewChatClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    setActiveChatId(nanoid())
    if (completionPanelViewType === CompletionPanelView.SESSIONS) {
      setCompletionPanelViewType(CompletionPanelView.CHAT)
    }
  }

  return (
    <div className="flex items-center justify-between bg-secondary px-2 py-1">
      <div className="flex items-center gap-2">
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
      </div>
      <Image src={tabbyLogo} alt="logo" width={32} />
      <IconClose onClick={e => setCompletionPanelVisible(false)} />
    </div>
  )
}
