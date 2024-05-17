import React from 'react'
import type { Context } from 'tabby-chat-panel'
import { useClient } from 'tabby-chat-panel/react'

import { useMe } from '@/lib/hooks/use-me'
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
  const [{ data }] = useMe()
  const { pendingEvent, setPendingEvent } = React.useContext(
    SourceCodeBrowserContext
  )
  const activeChatId = useStore(useChatStore, state => state.activeChatId)
  const iframeRef = React.useRef<HTMLIFrameElement>(null)
  const client = useClient(iframeRef, {
    navigate: (context: Context) => {
      console.log('todo: ', context)
    }
  })

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
            filepath: path
          }
        } as UserMessage
      })
      setPendingEvent(undefined)
    }
  }, [pendingEvent, iframeRef.current?.contentWindow])

  React.useEffect(() => {
    if (iframeRef?.current && data) {
      client?.init({
        fetcherOptions: {
          authorization: data.me.authToken
        }
      })
    }
  }, [iframeRef?.current, client, data])

  React.useEffect(() => {
    if (pendingEvent && client) {
      const { lineFrom, lineTo, code, path } = pendingEvent
      client.sendMessage({
        message: getPrompt(pendingEvent),
        selectContext: {
          kind: 'file',
          content: code,
          range: {
            start: lineFrom,
            end: lineTo ?? lineFrom
          },
          filepath: path
        }
      })
    }
    setPendingEvent(undefined)
  }, [pendingEvent, client])

  if (!data?.me) return <></>
  return (
    <div className={cn('flex h-full flex-col', className)} {...props}>
      <Header />
      <iframe
        src={`/chat`}
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
