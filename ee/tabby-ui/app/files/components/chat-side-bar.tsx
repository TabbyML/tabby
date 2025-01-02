import React, { useRef, useState } from 'react'
import { find } from 'lodash-es'
import type { FileLocation, GitRepository } from 'tabby-chat-panel'
import { useClient } from 'tabby-chat-panel/react'

import { RepositoryListQuery } from '@/lib/gql/generates/graphql'
import { useLatest } from '@/lib/hooks/use-latest'
import { useMe } from '@/lib/hooks/use-me'
import { filename2prism } from '@/lib/language-utils'
import { useChatStore } from '@/lib/stores/chat-store'
import { cn, formatLineHashForLocation } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { IconClose } from '@/components/ui/icons'

import { QuickActionEventPayload } from '../lib/event-emitter'
import { SourceCodeBrowserContext } from './source-code-browser'
import { generateEntryPath, getDefaultRepoRef, resolveRepoRef } from './utils'

interface ChatSideBarProps
  extends Omit<React.HTMLAttributes<HTMLDivElement>, 'children'> {
  activeRepo: RepositoryListQuery['repositoryList'][0] | undefined
}

export const ChatSideBar: React.FC<ChatSideBarProps> = ({
  activeRepo,
  className,
  ...props
}) => {
  const [{ data }] = useMe()
  const [initialized, setInitialized] = useState(false)
  const { pendingEvent, setPendingEvent, repoMap, updateActivePath } =
    React.useContext(SourceCodeBrowserContext)
  const activeChatId = useChatStore(state => state.activeChatId)
  const iframeRef = React.useRef<HTMLIFrameElement>(null)
  const executedCommand = useRef(false)
  const repoMapRef = useLatest(repoMap)
  const openInCodeBrowser = async (fileLocation: FileLocation) => {
    const { filepath, location } = fileLocation
    if (filepath.kind === 'git') {
      const lineHash = formatLineHashForLocation(location)
      const repoMap = repoMapRef.current
      const matchedRepositoryKey = find(
        Object.keys(repoMap),
        key => repoMap?.[key]?.gitUrl === filepath.gitUrl
      )
      if (matchedRepositoryKey) {
        const targetRepo = repoMap[matchedRepositoryKey]
        if (targetRepo) {
          const defaultRef = getDefaultRepoRef(targetRepo.refs)
          // navigate to files of the default branch
          const refName = resolveRepoRef(defaultRef)?.name
          const detectedLanguage = filename2prism(filepath.filepath)[0]
          const isMarkdown = detectedLanguage === 'markdown'
          updateActivePath(
            generateEntryPath(targetRepo, refName, filepath.filepath, 'file'),
            {
              hash: lineHash,
              replace: false,
              plain: isMarkdown && !!lineHash
            }
          )
          return true
        }
      }
    }
    return false
  }

  const readWorkspaceGitRepositories = useLatest(() => {
    if (!activeRepo) return []
    const list: GitRepository[] = [{ url: activeRepo.gitUrl }]
    return list
  })

  const client = useClient(iframeRef, {
    refresh: async () => {
      window.location.reload()

      // Ensure the loading effect is maintained
      await new Promise(resolve => {
        setTimeout(() => resolve(null), 1000)
      })
    },
    onApplyInEditor(_content) {},
    onLoaded() {
      setInitialized(true)
    },
    onCopy(_content) {},
    onKeyboardEvent() {},
    openInEditor: async (fileLocation: FileLocation) => {
      return openInCodeBrowser(fileLocation)
    },
    openExternal: async (url: string) => {
      window.open(url, '_blank')
    },
    readWorkspaceGitRepositories: async () => {
      return readWorkspaceGitRepositories.current?.()
    },
    getActiveEditorSelection: async() => {
      // FIXME(@jueliang) implement
      return null
    },
  })

  const getCommand = ({ action }: QuickActionEventPayload) => {
    switch (action) {
      case 'explain':
        return 'explain'
      case 'generate_unittest':
        return 'generate-tests'
      case 'generate_doc':
        return 'generate-docs'
    }
  }

  React.useEffect(() => {
    if (iframeRef?.current && data) {
      client?.init({
        fetcherOptions: {
          authorization: data.me.authToken
        }
      })
    }
  }, [iframeRef?.current, client?.init, data])

  React.useEffect(() => {
    if (pendingEvent && client && initialized) {
      const execute = async () => {
        const { lineFrom, lineTo, code, path, gitUrl } = pendingEvent
        client.updateActiveSelection({
          kind: 'file',
          content: code,
          range: {
            start: lineFrom,
            end: lineTo ?? lineFrom
          },
          filepath: {
            kind: 'git',
            filepath: path,
            gitUrl
          }
        })
        const command = getCommand(pendingEvent)
        // FIXME: this delay is a workaround for waiting for the active selection to be updated
        setTimeout(() => {
          client.executeCommand(command)
        }, 500)
        setPendingEvent(undefined)
      }

      execute()
    }
  }, [initialized, pendingEvent])

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
