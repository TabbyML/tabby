import React from 'react'
import type { UseChatHelpers } from 'ai/react'

import { Button } from '@/components/ui/button'
import { IconRefresh, IconStop, IconTrash } from '@/components/ui/icons'
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
}

export function ChatPanel({
  id,
  stop,
  reload,
  input,
  setInput,
  className,
  onSubmit,
  chatMaxWidthClass
}: ChatPanelProps) {
  const promptFormRef = React.useRef<PromptFormRef>(null)
  const { container, onClearMessages, qaPairs, isLoading } =
    React.useContext(ChatContext)
  React.useEffect(() => {
    promptFormRef?.current?.focus()
  }, [id])

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
              className="bg-background lg:hidden"
            >
              <IconTrash className="mr-2" />
              Clear
            </Button>
          )}
        </div>
        <div className="space-y-4 border-t bg-background px-4 py-2 shadow-lg sm:rounded-t-xl sm:border md:py-4">
          <PromptForm
            ref={promptFormRef}
            onSubmit={onSubmit}
            input={input}
            setInput={setInput}
            isLoading={isLoading}
          />
          <FooterText className="hidden sm:block" />
        </div>
      </div>
    </div>
  )
}
