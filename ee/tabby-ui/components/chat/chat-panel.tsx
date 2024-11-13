import React, { RefObject } from 'react'
import type { UseChatHelpers } from 'ai/react'
import { AnimatePresence, motion } from 'framer-motion'
import type { Context } from 'tabby-chat-panel'

import { updateEnableActiveSelection } from '@/lib/stores/chat-actions'
import { useChatStore } from '@/lib/stores/chat-store'
import { cn, isFileContextContentEmpty } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  IconEye,
  IconEyeOff,
  IconRefresh,
  IconRemove,
  IconStop,
  IconTrash
} from '@/components/ui/icons'
import { ButtonScrollToBottom } from '@/components/button-scroll-to-bottom'
import { PromptForm, PromptFormRef } from '@/components/chat/prompt-form'
import { FooterText } from '@/components/footer'

import { ChatContext } from './chat'

export interface ChatPanelProps
  extends Pick<UseChatHelpers, 'stop' | 'input' | 'setInput'> {
  id?: string
  className?: string
  onSubmit: (content: string) => Promise<any>
  reload: () => void
  chatMaxWidthClass: string
  chatInputRef: RefObject<HTMLTextAreaElement>
}

export interface ChatPanelRef {
  focus: () => void
}

function ChatPanelRenderer(
  {
    stop,
    reload,
    input,
    setInput,
    className,
    onSubmit,
    chatMaxWidthClass,
    chatInputRef
  }: ChatPanelProps,
  ref: React.Ref<ChatPanelRef>
) {
  const promptFormRef = React.useRef<PromptFormRef>(null)
  const {
    container,
    onClearMessages,
    qaPairs,
    isLoading,
    relevantContext,
    removeRelevantContext,
    activeSelection
  } = React.useContext(ChatContext)
  const enableActiveSelection = useChatStore(
    state => state.enableActiveSelection
  )

  React.useImperativeHandle(
    ref,
    () => {
      return {
        focus: () => {
          promptFormRef.current?.focus()
        }
      }
    },
    []
  )

  return (
    <div className={className}>
      <ButtonScrollToBottom container={container} />
      <div className={`mx-auto md:px-4 ${chatMaxWidthClass}`}>
        <div className="flex h-10 items-center justify-center gap-2">
          {isLoading ? (
            <Button
              variant="outline"
              onClick={() => stop()}
              className="bg-background"
            >
              <IconStop className="mr-2" />
              Stop generating
            </Button>
          ) : (
            qaPairs?.length > 0 && (
              <Button
                variant="outline"
                onClick={() => reload()}
                className="bg-background"
              >
                <IconRefresh className="mr-2" />
                Regenerate response
              </Button>
            )
          )}
          {qaPairs?.length > 0 && (
            <Button
              variant="outline"
              onClick={onClearMessages}
              className="bg-background"
            >
              <IconTrash className="mr-2" />
              Clear
            </Button>
          )}
        </div>
        <div className="border-t bg-background px-4 py-2 shadow-lg sm:space-y-4 sm:rounded-t-xl sm:border md:py-4">
          <div className="flex flex-wrap gap-2">
            <AnimatePresence>
              {activeSelection ? (
                <motion.div
                  initial={{ opacity: 0, scale: 0.9, y: -5 }}
                  animate={{ opacity: 1, scale: 1, y: 0 }}
                  transition={{
                    ease: 'easeInOut',
                    duration: 0.1
                  }}
                  exit={{ opacity: 0, scale: 0.9, y: 5 }}
                >
                  <Badge
                    variant="outline"
                    className={cn(
                      'inline-flex h-7 flex-nowrap items-center gap-1.5 overflow-hidden rounded-md pr-0 text-sm font-semibold',
                      {
                        'border-dashed !text-muted-foreground italic line-through':
                          !enableActiveSelection
                      }
                    )}
                  >
                    <ContextLabel
                      context={activeSelection}
                      className="flex-1 truncate"
                    />
                    <span className="shrink-0 text-muted-foreground">
                      Current file
                    </span>
                    <Button
                      size="icon"
                      variant="ghost"
                      className="h-7 w-7 shrink-0 rounded-l-none"
                      onClick={e => {
                        updateEnableActiveSelection(!enableActiveSelection)
                      }}
                    >
                      {enableActiveSelection ? <IconEye /> : <IconEyeOff />}
                    </Button>
                  </Badge>
                </motion.div>
              ) : null}
              {relevantContext.map((item, idx) => {
                return (
                  <motion.div
                    // FIXME check the dedupe logic
                    key={item.filepath + item.range.start + item.range.end}
                    initial={{ opacity: 0, scale: 0.9, y: -5 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    transition={{
                      ease: 'easeInOut',
                      duration: 0.1
                    }}
                    exit={{ opacity: 0, scale: 0.9, y: 5 }}
                    layout
                  >
                    <Badge
                      variant="outline"
                      className="inline-flex h-7 flex-nowrap items-center gap-1 overflow-hidden rounded-md pr-0 text-sm font-semibold"
                    >
                      <ContextLabel context={item} />
                      <Button
                        size="icon"
                        variant="ghost"
                        className="h-7 w-7 shrink-0 rounded-l-none"
                        onClick={removeRelevantContext.bind(null, idx)}
                      >
                        <IconRemove />
                      </Button>
                    </Badge>
                  </motion.div>
                )
              })}
            </AnimatePresence>
          </div>
          <PromptForm
            ref={promptFormRef}
            onSubmit={onSubmit}
            input={input}
            setInput={setInput}
            isLoading={isLoading}
            chatInputRef={chatInputRef}
          />
          <FooterText className="hidden sm:block" />
        </div>
      </div>
    </div>
  )
}

export const ChatPanel = React.forwardRef<ChatPanelRef, ChatPanelProps>(
  ChatPanelRenderer
)

function ContextLabel({
  context,
  className
}: {
  context: Context
  className?: string
}) {
  const [fileName] = context.filepath.split('/').slice(-1)
  const line = isFileContextContentEmpty(context)
    ? ''
    : context.range.start === context.range.end
    ? `:${context.range.start}`
    : `:${context.range.start}-${context.range.end}`

  return (
    <span className={cn('truncate', className)}>
      {fileName}
      {!!line && <span className="text-muted-foreground">{line}</span>}
    </span>
  )
}
