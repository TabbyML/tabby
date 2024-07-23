import React from 'react'
import type { UseChatHelpers } from 'ai/react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
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
  const {
    container,
    onClearMessages,
    qaPairs,
    isLoading,
    clientSelectedContext,
    removeClientSelectedContext
  } = React.useContext(ChatContext)
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
        <div className="border-t bg-background px-4 py-2 shadow-lg sm:space-y-4 sm:rounded-t-xl sm:border md:py-4">
          {clientSelectedContext.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {clientSelectedContext.map((item, idx) => {
                const [fileName] = item.filepath.split('/').slice(-1)
                const line =
                  item.range.start === item.range.end
                    ? `${item.range.start}`
                    : `${item.range.start}-${item.range.end}`
                return (
                  <Badge
                    variant="outline"
                    key={item.filepath + idx}
                    className="inline-flex items-center gap-0.5 rounded text-sm font-semibold"
                  >
                    <span className="text-foreground">{`${fileName}: ${line}`}</span>
                    <IconRemove
                      className="cursor-pointer text-muted-foreground transition-all hover:text-red-300"
                      onClick={removeClientSelectedContext.bind(null, idx)}
                    />
                  </Badge>
                )
              })}
            </div>
          )}
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
