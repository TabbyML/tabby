import React, { useState } from 'react'
import { find } from 'lodash-es'
import type {
  ChatCommand,
  ChatView,
  EditorFileContext,
  FileLocation,
  GitRepository
} from 'tabby-chat-panel'
import { useClient } from 'tabby-chat-panel/react'

import { RepositoryListQuery } from '@/lib/gql/generates/graphql'
import { useLatest } from '@/lib/hooks/use-latest'
import { useMe } from '@/lib/hooks/use-me'
import { filename2prism } from '@/lib/language-utils'
import { useChatStore } from '@/lib/stores/chat-store'
import { cn, formatLineHashForLocation } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { IconClose, IconHistory, IconPlus } from '@/components/ui/icons'

import { emitter } from '../lib/event-emitter'
import { getActiveSelection } from '../lib/selection-extension'
import { SourceCodeBrowserContext } from './source-code-browser'
import { generateEntryPath, getDefaultRepoRef, resolveRepoRef } from './utils'

interface ChatSideBarProps
  extends Omit<React.HTMLAttributes<HTMLDivElement>, 'children'> {
  activeRepo: RepositoryListQuery['repositoryList'][0] | undefined
  pendingCommand?: ChatCommand
}
export const ChatSideBar: React.FC<ChatSideBarProps> = props => {
  const [shouldInitialize, setShouldInitialize] = useState(false)
  const { chatSideBarVisible, setChatSideBarVisible } = React.useContext(
    SourceCodeBrowserContext
  )
  const [pendingCommand, setPendingCommand] = React.useState<
    ChatCommand | undefined
  >()

  React.useEffect(() => {
    if (chatSideBarVisible && !shouldInitialize) {
      setShouldInitialize(true)
    }
  }, [chatSideBarVisible])

  React.useEffect(() => {
    const quickActionCallback = (command: ChatCommand) => {
      setChatSideBarVisible(true)

      if (!shouldInitialize) {
        setPendingCommand(command)
      }
    }

    emitter.on('quick_action_command', quickActionCallback)
    return () => {
      emitter.off('quick_action_command', quickActionCallback)
    }
  }, [])

  if (!shouldInitialize) return null

  return <ChatSideBarRenderer pendingCommand={pendingCommand} {...props} />
}

function ChatSideBarRenderer({
  activeRepo,
  className,
  pendingCommand,
  ...props
}: ChatSideBarProps) {
  const [{ data }] = useMe()
  const { repoMap, updateActivePath, activeEntryInfo, textEditorViewRef } =
    React.useContext(SourceCodeBrowserContext)
  const activeChatId = useChatStore(state => state.activeChatId)
  const iframeRef = React.useRef<HTMLIFrameElement>(null)
  const repoMapRef = useLatest(repoMap)

  const client = useClient(iframeRef, {
    refresh: async () => {
      window.location.reload()

      // Ensure the loading effect is maintained
      await new Promise(resolve => {
        setTimeout(() => resolve(null), 1000)
      })
    },
    openInEditor: async (fileLocation: FileLocation) => {
      return openInCodeBrowser(fileLocation)
    },
    openExternal: async (url: string) => {
      window.open(url, '_blank')
    },
    onApplyInEditor: async _content => {},
    onCopy: async _content => {},
    readWorkspaceGitRepositories: async () => {
      return readWorkspaceGitRepositories.current()
    },
    getActiveEditorSelection: async () => {
      return getActiveEditorSelection.current()
    }
  })

  React.useEffect(() => {
    const quickActionCallback = (command: ChatCommand) => {
      client?.['0.8.0'].executeCommand(command)
    }

    emitter.on('quick_action_command', quickActionCallback)

    return () => {
      emitter.off('quick_action_command', quickActionCallback)
    }
  }, [client])

  React.useEffect(() => {
    const notifyActiveEditorSelectionChange = (
      editorFileContext: EditorFileContext | null
    ) => {
      client?.['0.8.0'].updateActiveSelection(editorFileContext)
    }

    emitter.on('selection_change', notifyActiveEditorSelectionChange)

    return () => {
      emitter.off('selection_change', notifyActiveEditorSelectionChange)
    }
  }, [client])

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

  const getActiveEditorSelection = useLatest(() => {
    if (!textEditorViewRef.current || !activeEntryInfo) return null

    const context = getActiveSelection(textEditorViewRef.current)

    if (!context || !activeEntryInfo.basename || !activeRepo) {
      return null
    }

    const editorFileContext: EditorFileContext = {
      kind: 'file',
      filepath: {
        kind: 'git',
        filepath: activeEntryInfo.basename,
        gitUrl: activeRepo?.gitUrl
      },
      range:
        'startLine' in context
          ? {
              start: context.startLine,
              end: context.endLine
            }
          : undefined,
      content: context.content
    }

    return editorFileContext
  })

  const navigate = useLatest((view: ChatView) => {
    client?.['0.8.0'].navigate(view)
  })

  const onNavigate = navigate.current

  React.useEffect(() => {
    if (client && data) {
      client['0.8.0'].init({
        fetcherOptions: {
          authorization: data.me.authToken
        }
      })
    }
  }, [iframeRef?.current, data, client])

  React.useEffect(() => {
    if (pendingCommand && client) {
      const execute = async () => {
        client['0.8.0'].executeCommand(pendingCommand)
      }

      execute()
    }
  }, [client])

  return (
    <div className={cn('flex h-full flex-col', className)} {...props}>
      <Header onNavigate={onNavigate} initialized={!!client} />
      <iframe
        src={`/chat`}
        className="w-full flex-1 border-0"
        key={activeChatId}
        ref={iframeRef}
      />
    </div>
  )
}

interface HeaderProps {
  onNavigate: (view: ChatView) => void
  initialized: boolean
}
function Header({ onNavigate, initialized }: HeaderProps) {
  const { setChatSideBarVisible } = React.useContext(SourceCodeBrowserContext)

  return (
    <div className="sticky top-0 flex items-center justify-end px-2 py-1">
      <Button
        size="icon"
        variant="ghost"
        disabled={!initialized}
        onClick={e => onNavigate('new-chat')}
      >
        <IconPlus />
      </Button>
      <Button
        size="icon"
        variant="ghost"
        disabled={!initialized}
        onClick={e => onNavigate('history')}
      >
        <IconHistory />
      </Button>
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
