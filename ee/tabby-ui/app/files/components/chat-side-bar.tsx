import React from 'react'
import type { Context } from 'tabby-chat-panel'
import { useClient } from 'tabby-chat-panel/react'

import { useMe } from '@/lib/hooks/use-me'
import useRouterStuff from '@/lib/hooks/use-router-stuff'
import { useStore } from '@/lib/hooks/use-store'
import { useChatStore } from '@/lib/stores/chat-store'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { IconClose } from '@/components/ui/icons'

import { QuickActionEventPayload } from '../lib/event-emitter'
import { SourceCodeBrowserContext } from './source-code-browser'
import { find } from 'lodash-es'
import { resolveRepoSpecifierFromRepoInfo } from './utils'
import { useLatest } from '@/lib/hooks/use-latest'

interface ChatSideBarProps
  extends Omit<React.HTMLAttributes<HTMLDivElement>, 'children'> { }

export const ChatSideBar: React.FC<ChatSideBarProps> = ({
  className,
  ...props
}) => {
  const { updateSearchParams } = useRouterStuff()
  const [{ data }] = useMe()
  const { pendingEvent, setPendingEvent, repoMap } = React.useContext(
    SourceCodeBrowserContext
  )
  const activeChatId = useStore(useChatStore, state => state.activeChatId)
  const iframeRef = React.useRef<HTMLIFrameElement>(null)
  const repoMapRef = useLatest(repoMap)

  const onNavigate = (context: Context) => {
    if (context?.filepath && context?.git_url) {
      const repoMap = repoMapRef.current
      const matchedRepositoryKey = find(Object.keys(repoMap), key => repoMap?.[key]?.gitUrl === context.git_url)
      if (matchedRepositoryKey) {
        const repository = repoMap[matchedRepositoryKey]
        const repositorySpecifier = resolveRepoSpecifierFromRepoInfo(repository)
        updateSearchParams({
          set: {
            path: `${repositorySpecifier ?? ''}/${context.filepath}`,
            line: String(context.range.start ?? '')
          },
          del: 'plain'
        })
      }
    }
  }

  const client = useClient(iframeRef, {
    navigate: onNavigate
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
      const { lineFrom, lineTo, code, path, gitUrl } = pendingEvent
      client.sendMessage({
        message: getPrompt(pendingEvent),
        selectContext: {
          kind: 'file',
          content: code,
          range: {
            start: lineFrom,
            end: lineTo ?? lineFrom
          },
          filepath: path,
          git_url: gitUrl
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
