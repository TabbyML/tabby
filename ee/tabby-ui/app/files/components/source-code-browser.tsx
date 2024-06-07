'use client'

import React, { PropsWithChildren } from 'react'
import { usePathname } from 'next/navigation'
import { createRequest } from '@urql/core'
import { compact, forEach, isEmpty, toNumber } from 'lodash-es'
import { ImperativePanelHandle } from 'react-resizable-panels'
import { toast } from 'sonner'
import { SWRResponse } from 'swr'
import useSWRImmutable from 'swr/immutable'

import { graphql } from '@/lib/gql/generates'
import { RepositoryListQuery } from '@/lib/gql/generates/graphql'
import useRouterStuff from '@/lib/hooks/use-router-stuff'
import { useIsChatEnabled } from '@/lib/hooks/use-server-info'
import { filename2prism } from '@/lib/language-utils'
import fetcher from '@/lib/tabby/fetcher'
import { client } from '@/lib/tabby/gql'
import type { ResolveEntriesResponse, TFile } from '@/lib/types'
import { cn } from '@/lib/utils'
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup
} from '@/components/ui/resizable'
import { BANNER_HEIGHT, useShowDemoBanner } from '@/components/demo-banner'
import { ListSkeleton } from '@/components/skeleton'
import { useTopbarProgress } from '@/components/topbar-progress-indicator'

import { emitter, QuickActionEventPayload } from '../lib/event-emitter'
import { ChatSideBar } from './chat-side-bar'
import { FileDirectoryBreadcrumb } from './file-directory-breadcrumb'
import { DirectoryView } from './file-directory-view'
import { mapToFileTree, sortFileTree, type TFileTreeNode } from './file-tree'
import { FileTreePanel } from './file-tree-panel'
import { RawFileView } from './raw-file-view'
import { TextFileView } from './text-file-view'
import {
  encodeURIComponentIgnoringSlash,
  fetchEntriesFromPath,
  getDefaultRepoRef,
  getDirectoriesFromBasename,
  getProviderVariantFromKind,
  repositoryList2Map,
  resolveFileNameFromPath,
  resolveRepoRef,
  resolveRepositoryInfoFromPath,
  resolveRepoSpecifierFromRepoInfo
} from './utils'

/**
 * FileMap example
 * {
 *   'tabby/ee/tabby-ui/README.md': {
 *     file: {
 *      kind: 'file',
 *      basename: 'ee/tabby-ui/README.md'
 *     },
 *     name: 'README.md',
 *     fullPath: 'tabby/ee/tabby-ui/README.md',
 *     treeExpanded: false
 *   },
 *   ...
 * }
 */
type TFileMapItem = {
  file: TFile
  name: string
  fullPath: string
  treeExpanded?: boolean
  isRepository?: boolean
  repository?: RepositoryItem | undefined
}
type TFileMap = Record<string, TFileMapItem>
type RepositoryItem = RepositoryListQuery['repositoryList'][0]

const repositoryListQuery = graphql(/* GraphQL */ `
  query RepositoryList {
    repositoryList {
      id
      name
      kind
      gitUrl
      refs
    }
  }
`)

type SourceCodeBrowserContextValue = {
  activePath: string | undefined
  updateActivePath: (
    path: string | undefined,
    options?: {
      params?: Record<'line', string>
      shouldFetchAllEntries?: boolean
      replace?: boolean
    }
  ) => Promise<void>
  repoMap: Record<string, RepositoryItem>
  setRepoMap: (map: Record<string, RepositoryItem>) => void
  fileMap: TFileMap
  updateFileMap: (map: TFileMap) => void
  expandedKeys: Set<string>
  setExpandedKeys: React.Dispatch<React.SetStateAction<Set<string>>>
  toggleExpandedKey: (key: string) => void
  currentFileRoutes: TFileMapItem[]
  initialized: boolean
  setInitialized: React.Dispatch<React.SetStateAction<boolean>>
  fileTreeData: TFileTreeNode[]
  chatSideBarVisible: boolean
  setChatSideBarVisible: React.Dispatch<React.SetStateAction<boolean>>
  pendingEvent: QuickActionEventPayload | undefined
  setPendingEvent: (d: QuickActionEventPayload | undefined) => void
  isChatEnabled: boolean | undefined
  activeRepo: RepositoryItem | undefined
  activeRepoRef:
    | { kind?: 'branch' | 'tag'; name?: string; ref: string }
    | undefined
  isPathInitialized: boolean
}

const SourceCodeBrowserContext =
  React.createContext<SourceCodeBrowserContextValue>(
    {} as SourceCodeBrowserContextValue
  )

const SourceCodeBrowserContextProvider: React.FC<PropsWithChildren> = ({
  children
}) => {
  const pathname = usePathname()
  const { setProgress } = useTopbarProgress()
  const { updatePathnameAndSearch } = useRouterStuff()
  const [isPathInitialized, setIsPathInitialized] = React.useState(false)
  const [activePath, setActivePath] = React.useState<string | undefined>()
  const isChatEnabled = useIsChatEnabled()
  const [initialized, setInitialized] = React.useState(false)
  const [fileMap, setFileMap] = React.useState<TFileMap>({})
  const [repoMap, setRepoMap] = React.useState<
    SourceCodeBrowserContextValue['repoMap']
  >({})
  const [expandedKeys, setExpandedKeys] = React.useState<Set<string>>(new Set())
  const [chatSideBarVisible, setChatSideBarVisible] = React.useState(false)
  const [pendingEvent, setPendingEvent] = React.useState<
    QuickActionEventPayload | undefined
  >()

  // todo rename
  const fetchAllEntries = React.useCallback(
    async (fullPath: string) => {
      if (!fullPath) return

      const { repositorySpecifier, rev, basename } =
        resolveRepositoryInfoFromPath(fullPath)
      try {
        setProgress(true)
        // fetch dirs
        const entries = await fetchEntriesFromPath(
          fullPath,
          repositorySpecifier ? repoMap?.[repositorySpecifier] : undefined
        )
        const expandedDirs = getDirectoriesFromBasename(basename)

        const patchMap: TFileMap = {}
        if (entries.length) {
          forEach(repoMap, (repo, repoKey) => {
            patchMap[`${repoKey}/${rev}`] = {
              file: {
                kind: 'dir',
                basename: repo.name
              },
              name: repo.name,
              fullPath: `${repoKey}/${rev}`,
              treeExpanded: repoKey === repositorySpecifier,
              isRepository: true,
              repository: repo
            }
          })

          for (const entry of entries) {
            const path = `${repositorySpecifier}/${rev}/${entry.basename}`
            patchMap[path] = {
              file: entry,
              name: resolveFileNameFromPath(path),
              fullPath: path,
              treeExpanded: expandedDirs.includes(entry.basename)
            }
          }
        }
        const expandedKeys = expandedDirs.map(dir =>
          [repositorySpecifier, dir].filter(Boolean).join('/')
        )
        if (patchMap) {
          setFileMap(patchMap)
        }
        if (expandedKeys?.length) {
          setExpandedKeys(new Set(expandedKeys))
        }
      } catch (e) {
      } finally {
        setProgress(false)
      }
    },
    [repoMap]
  )

  const updateActivePath: SourceCodeBrowserContextValue['updateActivePath'] =
    React.useCallback(
      async (path, options) => {
        const replace = options?.replace
        if (!path) {
          // To maintain compatibility with older versions, remove the path params
          updatePathnameAndSearch('/files', {
            del: ['path', 'plain', 'line'],
            replace
          })
        } else {
          if (options?.shouldFetchAllEntries) {
            await fetchAllEntries(path)
          }
          const setParams: Record<string, string> = {}
          let delList = [
            'plain',
            'line',
            'redirect_filepath',
            'redirect_git_url'
          ]
          if (options?.params?.line) {
            setParams['line'] = options.params.line
            delList = delList.filter(o => o !== 'line')
          }
          updatePathnameAndSearch(`/files/${path}`, {
            set: setParams,
            del: delList,
            replace
          })
        }
      },
      [fetchAllEntries]
    )

  const updateFileMap = (map: TFileMap) => {
    if (!map) return

    setFileMap(prevMap => ({
      ...prevMap,
      ...map
    }))
  }

  const toggleExpandedKey = (key: string) => {
    const expanded = expandedKeys.has(key)
    const newSet = new Set(expandedKeys)
    if (expanded) {
      newSet.delete(key)
    } else {
      newSet.add(key)
    }
    setExpandedKeys(newSet)
  }

  const currentFileRoutes = React.useMemo(() => {
    if (!activePath) return []
    let result: TFileMapItem[] = []
    const {
      repositorySpecifier,
      rev,
      basename = ''
    } = resolveRepositoryInfoFromPath(activePath)
    const pathSegments = [
      `${repositorySpecifier}/${rev}`,
      ...basename?.split('/')
    ]
    for (let i = 0; i < pathSegments.length; i++) {
      const p = pathSegments.slice(0, i + 1).join('/')
      result.push(fileMap?.[p])
    }
    return compact(result)
  }, [activePath, fileMap])

  const fileTreeData: TFileTreeNode[] = React.useMemo(() => {
    return sortFileTree(mapToFileTree(fileMap))
  }, [fileMap])

  const activeRepo = React.useMemo(() => {
    const { repositoryKind, repositoryName, repositorySpecifier } =
      resolveRepositoryInfoFromPath(activePath)
    if (!repositoryKind || !repositoryName) return undefined
    return repositorySpecifier ? repoMap[repositorySpecifier] : undefined
  }, [activePath, repoMap])

  const activeRepoRef = React.useMemo(() => {
    if (!activePath || !activeRepo) return undefined
    const rev = decodeURIComponent(
      resolveRepositoryInfoFromPath(activePath)?.rev ?? ''
    )
    const activeRepoRef = activeRepo.refs?.find(
      ref => ref === `refs/heads/${rev}` || ref === `refs/tags/${rev}`
    )
    if (activeRepoRef) {
      return resolveRepoRef(activeRepoRef)
    }
  }, [activePath, activeRepo])

  React.useEffect(() => {
    const regex = /^\/files\/(.*)/
    const path = pathname.match(regex)?.[1]

    setActivePath(path ?? '')
    if (!isPathInitialized) {
      setIsPathInitialized(true)
    }
  }, [pathname])

  return (
    <SourceCodeBrowserContext.Provider
      value={{
        initialized,
        setInitialized,
        activePath,
        updateActivePath,
        fileMap,
        updateFileMap,
        expandedKeys,
        setExpandedKeys,
        toggleExpandedKey,
        currentFileRoutes,
        fileTreeData,
        chatSideBarVisible,
        setChatSideBarVisible,
        pendingEvent,
        setPendingEvent,
        isChatEnabled,
        repoMap,
        setRepoMap,
        activeRepo,
        activeRepoRef,
        isPathInitialized
      }}
    >
      {children}
    </SourceCodeBrowserContext.Provider>
  )
}

interface SourceCodeBrowserProps {
  className?: string
}

type FileDisplayType = 'image' | 'text' | 'raw' | ''

const SourceCodeBrowserRenderer: React.FC<SourceCodeBrowserProps> = ({
  className
}) => {
  const {
    activePath,
    updateActivePath,
    updateFileMap,
    fileMap,
    initialized,
    setInitialized,
    setExpandedKeys,
    chatSideBarVisible,
    setChatSideBarVisible,
    setPendingEvent,
    setRepoMap,
    activeRepo,
    activeRepoRef,
    isPathInitialized
  } = React.useContext(SourceCodeBrowserContext)
  const { searchParams } = useRouterStuff()

  const initializing = React.useRef(false)
  const { setProgress } = useTopbarProgress()
  const chatSideBarPanelRef = React.useRef<ImperativePanelHandle>(null)
  const [chatSideBarPanelSize, setChatSideBarPanelSize] = React.useState(35)

  const activeRepoIdentity = React.useMemo(() => {
    const repoId = activeRepo?.id
    const kind = activeRepo?.kind
    if (!repoId || !kind) return ''
    return `${getProviderVariantFromKind(kind)}/${repoId}`
  }, [activeRepo])

  const activeBasename = React.useMemo(() => {
    return resolveRepositoryInfoFromPath(activePath)?.basename
  }, [activePath])

  const [fileViewType, setFileViewType] = React.useState<FileDisplayType>()

  const isFileSelected =
    activePath && fileMap?.[activePath]?.file?.kind === 'file'
  const activeEntry = activePath ? fileMap?.[activePath]?.file : undefined

  const shouldFetchSubDir = React.useMemo(() => {
    if (!initialized) return false

    const isDir = activePath && fileMap?.[activePath]?.file?.kind === 'dir'
    return isDir && !fileMap?.[activePath]?.treeExpanded
  }, [activePath, fileMap, initialized])

  // fetch raw file
  const { data: rawFileResponse, isLoading: fetchingRawFile } =
    useSWRImmutable<{
      blob?: Blob
      contentLength?: number
    }>(
      isFileSelected
        ? `/repositories/${activeRepoIdentity}/rev/${
            activeRepoRef?.name ?? 'main'
          }/${encodeURIComponentIgnoringSlash(activeBasename ?? '')}`
        : null,
      (url: string) =>
        fetcher(url, {
          responseFormatter: async response => {
            if (!response.ok) return undefined

            const contentLength = toNumber(
              response.headers.get('Content-Length')
            )
            // todo abort big size request and truncate
            const blob = await response.blob()
            return {
              contentLength,
              blob
            }
          }
        }),
      {
        onError(err) {
          toast.error('Fail to fetch')
        }
      }
    )

  const fileBlob = rawFileResponse?.blob
  const contentLength = rawFileResponse?.contentLength

  // fetch active dir
  const {
    data: subTree,
    isLoading: fetchingSubTree
  }: SWRResponse<ResolveEntriesResponse> = useSWRImmutable(
    shouldFetchSubDir
      ? `/repositories/${activeRepoIdentity}/rev/${
          activeRepoRef?.name ?? 'main'
        }/${encodeURIComponentIgnoringSlash(activeBasename ?? '')}`
      : null,
    fetcher
  )

  const showDirectoryView = activeEntry?.kind === 'dir' || activePath === ''
  const showTextFileView = isFileSelected && fileViewType === 'text'
  const showRawFileView =
    isFileSelected && (fileViewType === 'image' || fileViewType === 'raw')

  const onPanelLayout = (sizes: number[]) => {
    if (sizes?.[2]) {
      setChatSideBarPanelSize(sizes[2])
    }
  }

  // todo check if params is valid
  React.useEffect(() => {
    const init = async () => {
      if (initializing.current) return

      initializing.current = true
      const { patchMap, expandedKeys, repos } = await getInitialFileData(
        activePath
      )
      // todo if no rev, set default rev

      const redirect_filepath = searchParams.get('redirect_filepath')
      const redirect_git_url = searchParams.get('redirect_git_url')
      const line = searchParams.get('line')

      // todo need ref info
      if (repos?.length && redirect_filepath && redirect_git_url) {
        // get repo from repos and redirect_git_url
        const targetRepo = repos.find(repo => repo.gitUrl === redirect_git_url)
        if (targetRepo) {
          const defaultRef = getDefaultRepoRef(targetRepo.refs)
          // use 'main' as fallback
          const refName = resolveRepoRef(defaultRef ?? '')?.name || 'main'
          const repoSpecifier = resolveRepoSpecifierFromRepoInfo(targetRepo)
          updateActivePath(`${repoSpecifier}/${refName}/${redirect_filepath}`, {
            params: { line: line ?? '' },
            replace: true
          })
          initializing.current = false
          return
        }
      }

      // todo if there's no rev, set default rev, should make a function to do it
      // By default, selecting the first repository if initialPath is empty
      if (repos?.length && !activePath) {
        const targetRepo = repos?.[0]
        const defaultRef = getDefaultRepoRef(targetRepo.refs)
        // use 'main' as fallback
        const refName = resolveRepoRef(defaultRef ?? '')?.name || 'main'
        const repoSpecifier = resolveRepoSpecifierFromRepoInfo(targetRepo)
        updateActivePath(`${repoSpecifier}/${refName}`, {
          replace: true
        })
        initializing.current = false
        return
      }

      // todo
      if (repos) setRepoMap(repositoryList2Map(repos))
      if (patchMap) updateFileMap(patchMap)
      if (expandedKeys?.length) setExpandedKeys(new Set(expandedKeys))
      setInitialized(true)
    }

    if (!initialized && isPathInitialized) {
      init()
    }
  }, [activePath, initialized, isPathInitialized])

  React.useEffect(() => {
    if (!initialized) return
    if (fetchingSubTree || fetchingRawFile) {
      setProgress(true)
    } else if (!fetchingSubTree && !fetchingRawFile) {
      setProgress(false)
    }
  }, [fetchingSubTree, fetchingRawFile])

  React.useEffect(() => {
    const onFetchSubTree = () => {
      if (Array.isArray(subTree?.entries) && activePath) {
        const { repositorySpecifier, rev } =
          resolveRepositoryInfoFromPath(activePath)
        let patchMap: TFileMap = {}
        if (fileMap?.[activePath]) {
          patchMap[activePath] = {
            ...fileMap[activePath],
            treeExpanded: true
          }
        }
        if (subTree?.entries?.length) {
          for (const entry of subTree.entries) {
            const path = `${repositorySpecifier}/${rev}/${entry.basename}`
            patchMap[path] = {
              file: entry,
              name: resolveFileNameFromPath(path),
              fullPath: path,
              treeExpanded: false
            }
          }
        }
        updateFileMap(patchMap)
        const expandedKeysToAdd = getDirectoriesFromBasename(activePath, true)
        if (expandedKeysToAdd?.length) {
          setExpandedKeys(keys => {
            const newSet = new Set(keys)
            for (const k of expandedKeysToAdd) {
              newSet.add(k)
            }
            return newSet
          })
        }
      }
    }

    onFetchSubTree()
  }, [subTree])

  React.useEffect(() => {
    const calculateViewType = async () => {
      const displayType = await getFileViewType(activePath ?? '', fileBlob)
      setFileViewType(displayType)
    }

    if (isFileSelected) {
      calculateViewType()
    } else {
      setFileViewType('')
    }
  }, [activePath, isFileSelected, fileBlob])

  React.useEffect(() => {
    const onCallCompletion = (data: QuickActionEventPayload) => {
      setChatSideBarVisible(true)
      setPendingEvent(data)
    }
    emitter.on('code_browser_quick_action', onCallCompletion)

    return () => {
      emitter.off('code_browser_quick_action', onCallCompletion)
    }
  }, [])

  React.useEffect(() => {
    if (chatSideBarVisible) {
      chatSideBarPanelRef.current?.expand()
      chatSideBarPanelRef.current?.resize(chatSideBarPanelSize)
    } else {
      chatSideBarPanelRef.current?.collapse()
    }
  }, [chatSideBarVisible])

  return (
    <ResizablePanelGroup
      direction="horizontal"
      className={cn(className)}
      onLayout={onPanelLayout}
    >
      <ResizablePanel
        defaultSize={20}
        minSize={20}
        maxSize={40}
        className="hidden lg:block"
      >
        <FileTreePanel />
      </ResizablePanel>
      <ResizableHandle className="hidden w-1 bg-border/40 hover:bg-border active:bg-blue-500 lg:block" />
      <ResizablePanel defaultSize={80} minSize={30}>
        <div className="flex h-full flex-col overflow-y-auto px-4 pb-4">
          <FileDirectoryBreadcrumb className="py-4" />
          {!initialized ? (
            <ListSkeleton className="rounded-lg border p-4" />
          ) : (
            <div>
              {showDirectoryView && (
                <DirectoryView
                  loading={fetchingSubTree}
                  initialized={initialized}
                  className={`rounded-lg border`}
                />
              )}
              {showTextFileView && (
                <TextFileView blob={fileBlob} contentLength={contentLength} />
              )}
              {showRawFileView && (
                <RawFileView
                  blob={fileBlob}
                  isImage={fileViewType === 'image'}
                  contentLength={contentLength}
                />
              )}
            </div>
          )}
        </div>
      </ResizablePanel>
      <>
        <ResizableHandle
          className={cn(
            'hidden w-1 bg-border/40 hover:bg-border active:bg-blue-500',
            chatSideBarVisible && 'block'
          )}
        />
        <ResizablePanel
          collapsible
          collapsedSize={0}
          defaultSize={0}
          minSize={25}
          ref={chatSideBarPanelRef}
          onCollapse={() => setChatSideBarVisible(false)}
        >
          <ChatSideBar />
        </ResizablePanel>
      </>
    </ResizablePanelGroup>
  )
}

const SourceCodeBrowser: React.FC<SourceCodeBrowserProps> = props => {
  const [isShowDemoBanner] = useShowDemoBanner()
  const style = isShowDemoBanner
    ? { height: `calc(100vh - ${BANNER_HEIGHT})` }
    : { height: '100vh' }
  return (
    <SourceCodeBrowserContextProvider>
      <div className="transition-all" style={style}>
        <SourceCodeBrowserRenderer className="source-code-browser" {...props} />
      </div>
    </SourceCodeBrowserContextProvider>
  )
}

async function getInitialFileData(path?: string) {
  try {
    const {
      repositoryKind: initialRepositoryKind,
      repositoryName: initialRepositoryName,
      basename: initialBasename,
      repositorySpecifier,
      rev
    } = resolveRepositoryInfoFromPath(path)
    const repos = await fetchAllRepositories()
    const repoMap = repositoryList2Map(repos)
    const initialRepo = repositorySpecifier
      ? repoMap?.[repositorySpecifier]
      : undefined
    const initialEntries =
      path && initialRepo ? await fetchInitialEntries(path, repoMap) : []
    const initialExpandedDirs = path
      ? getDirectoriesFromBasename(initialBasename)
      : []

    const patchMap: TFileMap = {}
    for (const repo of repos) {
      const specifier = resolveRepoSpecifierFromRepoInfo(repo)
      const defaultRevName =
        resolveRepoRef(getDefaultRepoRef(repo.refs) ?? '')?.name ?? 'main'
      const fullPath =
        repositorySpecifier === specifier
          ? `${specifier}/${rev}`
          : `${specifier}/${defaultRevName}`
      if (specifier) {
        patchMap[fullPath] = {
          file: {
            kind: 'dir',
            basename: repo.name
          },
          name: repo.name,
          fullPath,
          treeExpanded:
            repo.name === initialRepositoryName &&
            repo.kind === initialRepositoryKind,
          isRepository: true,
          repository: repo
        }
      }
    }
    for (const entry of initialEntries) {
      const path = `${resolveRepoSpecifierFromRepoInfo({
        kind: initialRepositoryKind,
        name: initialRepositoryName
      })}/${rev}/${entry.basename}`
      patchMap[path] = {
        file: entry,
        name: resolveFileNameFromPath(entry.basename),
        fullPath: path,
        treeExpanded: initialExpandedDirs.includes(entry.basename)
      }
    }
    const expandedKeys = initialExpandedDirs.map(dir =>
      [resolveRepoSpecifierFromRepoInfo(initialRepo), dir]
        .filter(Boolean)
        .join('/')
    )

    return { patchMap, expandedKeys, repos }
  } catch (e) {
    // eslint-disable-next-line no-console
    console.error(e)
    return {}
  }

  async function fetchAllRepositories(): Promise<
    RepositoryListQuery['repositoryList']
  > {
    const query = client.createRequestOperation(
      'query',
      createRequest(repositoryListQuery, {})
    )
    return client
      .executeQuery(query)
      .then(data => data?.data?.repositoryList || [])
  }

  async function fetchInitialEntries(
    path: string,
    repoMap: Record<string, RepositoryItem>
  ) {
    let result: TFile[] = []
    try {
      const { repositorySpecifier, rev } = resolveRepositoryInfoFromPath(path)
      if (!isEmpty(repoMap) && repositorySpecifier) {
        const repo = repoMap[repositorySpecifier]
        if (!repo) {
          return result
        }

        const defaultEntries = await fetchEntriesFromPath(path, repo)
        result = defaultEntries ?? []
      }
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error(e)
      return result
    }
    return result
  }
}

function isReadableTextFile(blob: Blob) {
  return new Promise((resolve, reject) => {
    const chunkSize = 1024
    const blobPart = blob.slice(0, chunkSize)
    const reader = new FileReader()

    reader.onloadend = function (e) {
      if (e?.target?.readyState === FileReader.DONE) {
        const text = e.target.result
        const nonPrintableRegex = /[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]/
        if (typeof text !== 'string') {
          resolve(false)
        } else if (nonPrintableRegex.test(text)) {
          resolve(false)
        } else {
          resolve(true)
        }
      }
    }

    reader.onerror = function () {
      resolve(false)
    }

    reader.readAsText(blobPart, 'UTF-8')
  })
}

async function getFileViewType(
  path: string,
  blob: Blob | undefined
): Promise<FileDisplayType> {
  if (!blob) return ''
  const mimeType = blob?.type
  const detectedLanguage = filename2prism(path)?.[0]

  if (mimeType?.startsWith('image')) return 'image'
  if (detectedLanguage || mimeType?.startsWith('text')) return 'text'

  const isReadableText = await isReadableTextFile(blob)
  return isReadableText ? 'text' : 'raw'
}

export type { TFileMap, TFileMapItem }

export { SourceCodeBrowserContext, SourceCodeBrowser }
