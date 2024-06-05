import React from 'react'
import { find } from 'lodash-es'
import type { Context } from 'tabby-chat-panel'
import { useClient } from 'tabby-chat-panel/react'

import { useLatest } from '@/lib/hooks/use-latest'
import { useMe } from '@/lib/hooks/use-me'
import useRouterStuff from '@/lib/hooks/use-router-stuff'
import { useStore } from '@/lib/hooks/use-store'
import { useChatStore } from '@/lib/stores/chat-store'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { IconClose } from '@/components/ui/icons'
import { useTopbarProgress } from '@/components/topbar-progress-indicator'

import { QuickActionEventPayload } from '../lib/event-emitter'
import { SourceCodeBrowserContext, TFileMap } from './source-code-browser'
import {
  fetchEntriesFromPath,
  getDirectoriesFromBasename,
  resolveFileNameFromPath,
  resolveRepoSpecifierFromRepoInfo
} from './utils'

interface ChatSideBarProps
  extends Omit<React.HTMLAttributes<HTMLDivElement>, 'children'> {}

export const ChatSideBar: React.FC<ChatSideBarProps> = ({
  className,
  ...props
}) => {
  const { setProgress } = useTopbarProgress()
  const { updateSearchParams, updatePathnameAndSearch } = useRouterStuff()
  const [{ data }] = useMe()
  const {
    pendingEvent,
    setPendingEvent,
    repoMap,
    setExpandedKeys,
    updateFileMap,
    activeRepoRef
  } = React.useContext(SourceCodeBrowserContext)
  const activeChatId = useStore(useChatStore, state => state.activeChatId)
  const iframeRef = React.useRef<HTMLIFrameElement>(null)
  const repoMapRef = useLatest(repoMap)
  const onNavigate = async (context: Context) => {
    if (context?.filepath && context?.git_url) {
      const repoMap = repoMapRef.current
      const matchedRepositoryKey = find(
        Object.keys(repoMap),
        key => repoMap?.[key]?.gitUrl === context.git_url
      )
      if (matchedRepositoryKey) {
        const repository = repoMap[matchedRepositoryKey]
        const repositorySpecifier = resolveRepoSpecifierFromRepoInfo(repository)
        const rev = activeRepoRef?.name

        const fullPath = `${repositorySpecifier}/${rev}/${context.filepath}`
        if (!fullPath) return
        try {
          setProgress(true)
          const entries = await fetchEntriesFromPath(
            fullPath,
            repositorySpecifier
              ? repoMap?.[`${repositorySpecifier}/${rev}`]
              : undefined
          )
          const initialExpandedDirs = getDirectoriesFromBasename(
            context.filepath
          )

          const patchMap: TFileMap = {}
          // fetch dirs
          for (const entry of entries) {
            const path = `${repositorySpecifier}/${entry.basename}`
            patchMap[path] = {
              file: entry,
              name: resolveFileNameFromPath(path),
              fullPath: path,
              treeExpanded: initialExpandedDirs.includes(entry.basename)
            }
          }
          const expandedKeys = initialExpandedDirs.map(dir =>
            [repositorySpecifier, dir].filter(Boolean).join('/')
          )
          if (patchMap) {
            updateFileMap(patchMap)
          }
          if (expandedKeys?.length) {
            setExpandedKeys(prevKeys => {
              const newSet = new Set(prevKeys)
              for (const k of expandedKeys) {
                newSet.add(k)
              }
              return newSet
            })
          }
        } catch (e) {
        } finally {
          updatePathnameAndSearch(
            `${repositorySpecifier ?? ''}/${context.filepath}`,
            {
              set: {
                line: String(context.range.start ?? '')
              },
              del: 'plain'
            }
          )
          setProgress(false)
        }
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
