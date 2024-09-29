import React from 'react'
import { find } from 'lodash-es'
import type { Context } from 'tabby-chat-panel'
import { useClient } from 'tabby-chat-panel/react'

import { useLatest } from '@/lib/hooks/use-latest'
import { useMe } from '@/lib/hooks/use-me'
import { useStore } from '@/lib/hooks/use-store'
import { filename2prism } from '@/lib/language-utils'
import { useChatStore } from '@/lib/stores/chat-store'
import { cn, formatLineHashForCodeBrowser } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { IconClose } from '@/components/ui/icons'

import { QuickActionEventPayload } from '../lib/event-emitter'
import { SourceCodeBrowserContext } from './source-code-browser'
import { generateEntryPath, getDefaultRepoRef, resolveRepoRef } from './utils'

interface ChatSideBarProps
  extends Omit<React.HTMLAttributes<HTMLDivElement>, 'children'> {}

export const ChatSideBar: React.FC<ChatSideBarProps> = ({
  className,
  ...props
}) => {
  const [{ data }] = useMe()
  const { pendingEvent, setPendingEvent, repoMap, updateActivePath } =
    React.useContext(SourceCodeBrowserContext)
  const activeChatId = useStore(useChatStore, state => state.activeChatId)
  const iframeRef = React.useRef<HTMLIFrameElement>(null)
  const repoMapRef = useLatest(repoMap)
  const onNavigate = async (context: Context) => {
    if (context?.filepath && context?.git_url) {
      const lineHash = formatLineHashForCodeBrowser(context?.range)
      const repoMap = repoMapRef.current
      const matchedRepositoryKey = find(
        Object.keys(repoMap),
        key => repoMap?.[key]?.gitUrl === context.git_url
      )
      if (matchedRepositoryKey) {
        const targetRepo = repoMap[matchedRepositoryKey]
        if (targetRepo) {
          const defaultRef = getDefaultRepoRef(targetRepo.refs)
          // navigate to files of the default branch
          const refName = resolveRepoRef(defaultRef)?.name
          const detectedLanguage = context.filepath
            ? filename2prism(context.filepath)[0]
            : undefined
          const isMarkdown = detectedLanguage === 'markdown'
          updateActivePath(
            generateEntryPath(
              targetRepo,
              refName,
              context.filepath,
              context.kind
            ),
            {
              hash: lineHash,
              replace: false,
              plain: isMarkdown && !!lineHash
            }
          )
          return
        }
      }
    }
  }

  const client = useClient(iframeRef, {
    navigate: onNavigate,
    refresh: async () => {
      window.location.reload()

      // Ensure the loading effect is maintained
      await new Promise(resolve => {
        setTimeout(() => resolve(null), 1000)
      })
    },
    async onSubmitMessage(_msg, _relevantContext) {},
    onApplyInEditor(_content) {},
    onLoaded() {},
    onCopy(_content) {},
    onKeyboardEvent() {}
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
