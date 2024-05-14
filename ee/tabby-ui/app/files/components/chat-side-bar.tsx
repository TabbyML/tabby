import React from 'react'

import { useStore } from '@/lib/hooks/use-store'
import { useChatStore } from '@/lib/stores/chat-store'
import { UserMessage } from '@/lib/types'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { IconClose } from '@/components/ui/icons'

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

  const getPrompt = ({ action }: QuickActionEventPayload) => {
    let builtInPrompt = ''
    switch (action) {
      case 'explain':
        builtInPrompt = 'Explain the selected code:'
        break
      case 'generate_unittest':
        builtInPrompt = 'Generate a unit test for the selected code:'
        break
      case 'generate_doc':
        builtInPrompt = 'Generate documentation for the selected code:'
        break
      default:
        break
    }

    return builtInPrompt
  }

  React.useEffect(() => {
    const contentWindow = iframeRef.current?.contentWindow

    if (pendingEvent) {
      const { lineFrom, lineTo, language, code, path } = pendingEvent
      contentWindow?.postMessage({
        action: 'sendUserChat',
        payload: {
          message: getPrompt(pendingEvent),
          selectContext: {
            content: code,
            range: {
              start: lineFrom,
              end: lineTo
            },
            language,
            filePath: path,
            link: `/files?path=${encodeURIComponent(path ?? '')}&line=${
              lineFrom ?? ''
            }`
          }
        } as UserMessage
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

function Header() {
  const { setChatSideBarVisible } = React.useContext(SourceCodeBrowserContext)

  return (
    <div className="sticky top-0 flex items-center justify-end px-2 py-1">
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
