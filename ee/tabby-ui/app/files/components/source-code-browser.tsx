'use client'

import React, { PropsWithChildren, useState } from 'react'
import { usePathname } from 'next/navigation'
import { createRequest } from '@urql/core'
import { compact, isEmpty, isNil, toNumber } from 'lodash-es'
import { ImperativePanelHandle } from 'react-resizable-panels'
import useSWR from 'swr'

import { graphql } from '@/lib/gql/generates'
import { GitReference, RepositoryListQuery } from '@/lib/gql/generates/graphql'
import useRouterStuff from '@/lib/hooks/use-router-stuff'
import { useIsChatEnabled } from '@/lib/hooks/use-server-info'
import { filename2prism } from '@/lib/language-utils'
import fetcher from '@/lib/tabby/fetcher'
import { client } from '@/lib/tabby/gql'
import { repositoryListQuery } from '@/lib/tabby/query'
import type { ResolveEntriesResponse, TFile } from '@/lib/types'
import { cn, formatLineHashForCodeBrowser } from '@/lib/utils'
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup
} from '@/components/ui/resizable'
import { BANNER_HEIGHT, useShowDemoBanner } from '@/components/demo-banner'
import { ListSkeleton } from '@/components/skeleton'
import { useTopbarProgress } from '@/components/topbar-progress-indicator'

import { emitter, QuickActionEventPayload } from '../lib/event-emitter'
import { BlobModeView } from './blob-mode-view'
import { ChatSideBar } from './chat-side-bar'
import { CodeSearchBar } from './code-search-bar'
import { CodeSearchResultView } from './code-search-result-view'
import { ErrorView } from './error-view'
import { FileDirectoryBreadcrumb } from './file-directory-breadcrumb'
import { mapToFileTree, sortFileTree, type TFileTreeNode } from './file-tree'
import { FileTreePanel } from './file-tree-panel'
import { TreeModeView } from './tree-mode-view'
import type { FileDisplayType, RepositoryRefKind } from './types'
import {
  CodeBrowserError,
  generateEntryPath,
  getDefaultRepoRef,
  getDirectoriesFromBasename,
  parseLineNumberFromHash,
  repositoryList2Map,
  resolveFileNameFromPath,
  resolveRepoRef,
  resolveRepositoryInfoFromPath,
  toEntryRequestUrl
} from './utils'

/**
 * FileMap example
 * {
 *   'ee/tabby-ui/README.md': {
 *     file: {
 *      kind: 'file',
 *      basename: 'ee/tabby-ui/README.md'
 *     },
 *     name: 'README.md',
 *     fullPath: 'ee/tabby-ui/README.md',
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

const repositoryGrepQuery = graphql(/* GraphQL */ `
  query RepositoryGrep(
    $id: ID!
    $kind: RepositoryKind!
    $rev: String
    $query: String!
  ) {
    repositoryGrep(kind: $kind, id: $id, rev: $rev, query: $query) {
      files {
        path
        lines {
          line {
            text
            base64
          }
          byteOffset
          lineNumber
          subMatches {
            bytesStart
            bytesEnd
          }
        }
      }
      elapsedMs
    }
  }
`)

type SourceCodeBrowserContextValue = {
  activePath: string | undefined
  updateActivePath: (
    path: string | undefined,
    options?: {
      hash?: string
      replace?: boolean
      plain?: boolean
    }
  ) => Promise<void>
  repoMap: Record<string, RepositoryItem>
  setRepoMap: (map: Record<string, RepositoryItem>) => void
  fileMap: TFileMap
  updateFileMap: (map: TFileMap, replace?: boolean) => void
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
    | {
        kind?: 'branch' | 'tag' | 'commit'
        name?: string
        ref: GitReference | undefined
      }
    | undefined
  isPathInitialized: boolean
  activeEntryInfo: ReturnType<typeof resolveRepositoryInfoFromPath>
  prevActivePath: React.MutableRefObject<string | undefined>
  error: Error | undefined
  setError: (e: Error | undefined) => void
}

const SourceCodeBrowserContext =
  React.createContext<SourceCodeBrowserContextValue>(
    {} as SourceCodeBrowserContextValue
  )

const SourceCodeBrowserContextProvider: React.FC<PropsWithChildren> = ({
  children
}) => {
  const pathname = usePathname()
  const { updateUrlComponents, searchParams } = useRouterStuff()
  const redirectGitUrl = searchParams.get('redirect_git_url')?.toString()
  const [isPathInitialized, setIsPathInitialized] = React.useState(false)
  const [activePath, setActivePath] = React.useState<string | undefined>()
  const activeEntryInfo = React.useMemo(() => {
    return resolveRepositoryInfoFromPath(activePath)
  }, [activePath])

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
  const [error, setError] = useState<Error | undefined>()
  const prevActivePath = React.useRef<string | undefined>()

  const updateActivePath: SourceCodeBrowserContextValue['updateActivePath'] =
    React.useCallback(async (path, options) => {
      const replace = options?.replace
      if (!path) {
        // To maintain compatibility with older versions, remove the path params
        updateUrlComponents({
          pathname: '/files',
          searchParams: {
            del: ['path', 'plain', 'line']
          },
          hash: options?.hash,
          replace
        })
      } else {
        const setParams: Record<string, string> = {}
        let delList = ['redirect_filepath', 'redirect_git_url', 'line']
        if (options?.plain) {
          setParams['plain'] = '1'
        } else {
          delList.push('plain')
        }

        updateUrlComponents({
          pathname: `/files/${path}`,
          searchParams: {
            set: setParams,
            del: delList
          },
          replace,
          hash: options?.hash
        })
      }
    }, [])

  const updateFileMap = (map: TFileMap, replace?: boolean) => {
    if (!map) return
    if (replace) {
      setFileMap(map)
    } else {
      setFileMap(prevMap => ({
        ...prevMap,
        ...map
      }))
    }
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

  const fileTreeData: TFileTreeNode[] = React.useMemo(() => {
    return sortFileTree(mapToFileTree(fileMap))
  }, [fileMap])

  const activeRepo = React.useMemo(() => {
    const { repositoryKind, repositoryName, repositorySpecifier } =
      activeEntryInfo
    if (!repositoryKind || !repositoryName) return undefined
    return repositorySpecifier ? repoMap[repositorySpecifier] : undefined
  }, [repoMap, activeEntryInfo])

  const activeRepoRef = React.useMemo(() => {
    if (!activeEntryInfo || !activeRepo) return undefined
    const rev = activeEntryInfo?.rev ?? ''
    const _activeRepoRef = activeRepo.refs?.find(
      ref =>
        ref?.name === `refs/heads/${rev}` ||
        ref?.name === `refs/tags/${rev}` ||
        ref?.commit === rev
    )
    if (_activeRepoRef) {
      let refKind: RepositoryRefKind | undefined
      if (_activeRepoRef.name === `refs/heads/${rev}`) {
        refKind = 'branch'
      } else if (_activeRepoRef.name === `refs/tags/${rev}`) {
        refKind = 'tag'
      } else if (_activeRepoRef.commit === rev) {
        refKind = 'commit'
      }
      return resolveRepoRef(_activeRepoRef, refKind)
    }
  }, [activeEntryInfo, activeRepo])

  const currentFileRoutes = React.useMemo(() => {
    if (!activePath) return []
    const { basename = '' } = activeEntryInfo
    let result: TFileMapItem[] = [
      {
        file: {
          kind: 'dir',
          basename: ''
        },
        isRepository: true,
        repository: activeRepo,
        name: activeRepo?.name ?? '',
        fullPath: ''
      }
    ]

    const pathSegments = basename?.split('/') || []
    for (let i = 0; i < pathSegments.length; i++) {
      const p = pathSegments.slice(0, i + 1).join('/')
      result.push(fileMap?.[p])
    }
    return compact(result)
  }, [activePath, fileMap, activeEntryInfo])

  React.useEffect(() => {
    const regex = /^\/files\/(.*)/
    const path = pathname.match(regex)?.[1]
    prevActivePath.current = activePath
    setActivePath(path ?? '')
    if (!isPathInitialized) {
      setIsPathInitialized(true)
    }

    if (error) {
      setError(undefined)
    }
  }, [pathname, redirectGitUrl])

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
        isPathInitialized,
        activeEntryInfo,
        prevActivePath,
        error,
        setError
      }}
    >
      {children}
    </SourceCodeBrowserContext.Provider>
  )
}

interface SourceCodeBrowserProps {
  className?: string
}

const ENTRY_CONTENT_TYPE = 'application/vnd.directory+json'

const SourceCodeBrowserRenderer: React.FC<SourceCodeBrowserProps> = ({
  className
}) => {
  const {
    activePath,
    updateActivePath,
    initialized,
    setInitialized,
    chatSideBarVisible,
    setChatSideBarVisible,
    setPendingEvent,
    repoMap,
    setRepoMap,
    activeRepo,
    isPathInitialized,
    activeEntryInfo,
    prevActivePath,
    updateFileMap,
    setExpandedKeys,
    error,
    setError
  } = React.useContext(SourceCodeBrowserContext)

  const { searchParams } = useRouterStuff()
  const initializing = React.useRef(false)
  const { progress, setProgress } = useTopbarProgress()
  const chatSideBarPanelRef = React.useRef<ImperativePanelHandle>(null)
  const [chatSideBarPanelSize, setChatSideBarPanelSize] = React.useState(35)
  const searchQuery = searchParams.get('q')?.toString()

  const parsedEntryInfo = React.useMemo(() => {
    return resolveRepositoryInfoFromPath(activePath)
  }, [activePath])

  const activeBasename = parsedEntryInfo?.basename

  const isBlobMode = activeEntryInfo?.viewMode === 'blob'
  const isSearchMode = activeEntryInfo?.viewMode === 'search'

  const shouldFetchTree =
    !!initialized && !isEmpty(repoMap) && !!activePath && !isSearchMode
  const shouldFetchRepositoryGrep =
    !!initialized && !isEmpty(repoMap) && !!activePath && isSearchMode
  const shouldFetchRawFile = !!initialized && isBlobMode && activeRepo

  // fetch tree
  const {
    data: entriesResponse,
    isLoading: fetchingTreeEntries,
    error: entriesError
  } = useSWR<{
    entries: TFile[]
    requestPathname: string
  }>(
    shouldFetchTree ? activePath : null,
    (path: string) => {
      const { repositorySpecifier } = resolveRepositoryInfoFromPath(path)
      return fetchEntriesFromPath(
        path,
        repositorySpecifier ? repoMap?.[repositorySpecifier] : undefined
      ).then(data => ({ entries: data, requestPathname: path }))
    },
    {
      revalidateOnFocus: false,
      shouldRetryOnError: false
    }
  )

  // fetch raw file
  const {
    data: rawFileResponse,
    isLoading: fetchingRawFile,
    error: rawFileError
  } = useSWR<{
    blob?: Blob
    contentLength?: number
    fileDisplayType: FileDisplayType
  }>(
    shouldFetchRawFile
      ? [
          toEntryRequestUrl(activeRepo, activeEntryInfo.rev, activeBasename),
          activeBasename
        ]
      : null,
    ([url, basename]: [string, string]) =>
      fetcher(url, {
        responseFormatter: async response => {
          const contentType = response.headers.get('Content-Type')
          if (contentType === ENTRY_CONTENT_TYPE) {
            throw new Error(CodeBrowserError.INVALID_URL)
          }
          const contentLength = toNumber(response.headers.get('Content-Length'))
          // FIXME(jueliang) abort big size request and truncate the response data
          const blob = await response.blob()
          const fileDisplayType = await getFileViewType(basename ?? '', blob)
          return {
            contentLength,
            blob,
            fileDisplayType
          }
        },
        errorHandler() {
          throw new Error(CodeBrowserError.NOT_FOUND)
        }
      }),
    {
      revalidateOnFocus: false,
      shouldRetryOnError: false
    }
  )

  const {
    data: repositoryGreps,
    isLoading: fetchingRepositoryGrep,
    error: repositoryGrepError
  } = useSWR(
    shouldFetchRepositoryGrep && searchQuery ? [activePath, searchQuery] : null,
    ([activePath, searchQuery]) => {
      const { repositorySpecifier } = resolveRepositoryInfoFromPath(activePath)
      return fetchRepositoryGrep(
        searchQuery,
        repositorySpecifier ? repoMap?.[repositorySpecifier] : undefined,
        activeEntryInfo.rev
      )
    },
    {
      revalidateOnFocus: false,
      shouldRetryOnError: false
    }
  )

  const fileBlob = rawFileResponse?.blob
  const contentLength = rawFileResponse?.contentLength
  const fileDisplayType = rawFileResponse?.fileDisplayType
  const viewAffectingError = error || rawFileError || entriesError

  const showErrorView = !!viewAffectingError

  const isTreeMode =
    activeEntryInfo?.viewMode === 'tree' || !activeEntryInfo?.viewMode

  const onPanelLayout = (sizes: number[]) => {
    if (sizes?.[2]) {
      setChatSideBarPanelSize(sizes[2])
    }
  }

  React.useEffect(() => {
    const init = async () => {
      if (initializing.current) return

      initializing.current = true
      const repos = await fetchAllRepositories()
      const redirect_filepath = searchParams.get('redirect_filepath')
      const redirect_git_url = searchParams.get('redirect_git_url')

      if (repos?.length && redirect_filepath && redirect_git_url) {
        const targetRepo = repos.find(repo => repo.gitUrl === redirect_git_url)
        if (targetRepo) {
          // use default rev
          const defaultRef = getDefaultRepoRef(targetRepo.refs)
          const refName = resolveRepoRef(defaultRef)?.name || ''

          const lineRangeInHash = parseLineNumberFromHash(window.location.hash)
          const isValidLineHash = !isNil(lineRangeInHash?.start)

          // for backward compatibility, redirecting param likes `?line=1`
          const startLineNumber = parseInt(
            searchParams.get('line')?.toString() ?? ''
          )

          const nextHash = isValidLineHash
            ? window.location.hash
            : formatLineHashForCodeBrowser({ start: startLineNumber })

          const detectedLanguage = redirect_filepath
            ? filename2prism(redirect_filepath)[0]
            : undefined
          const isMarkdown = detectedLanguage === 'markdown'

          updateActivePath(
            generateEntryPath(targetRepo, refName, redirect_filepath, 'file'),
            {
              replace: true,
              hash: nextHash,
              plain: isMarkdown && !!nextHash
            }
          )
          initializing.current = false
          return
        } else {
          // target repository not found
          setError(new Error(CodeBrowserError.REPOSITORY_NOT_FOUND))
        }
      }

      if (repos) setRepoMap(repositoryList2Map(repos))
      setInitialized(true)
    }

    if (!initialized && isPathInitialized) {
      init()
    }
  }, [activePath, initialized, isPathInitialized])

  React.useEffect(() => {
    if (!entriesResponse) return

    const { entries, requestPathname } = entriesResponse
    const { repositorySpecifier, viewMode, basename, rev } =
      resolveRepositoryInfoFromPath(requestPathname)
    const { repositorySpecifier: prevRepositorySpecifier, rev: prevRev } =
      resolveRepositoryInfoFromPath(prevActivePath.current)
    const expandedDirs = getDirectoriesFromBasename(
      basename,
      viewMode === 'tree'
    )
    const patchMap: TFileMap = {}
    if (entries.length) {
      for (const entry of entries) {
        const _basename = entry.basename
        patchMap[_basename] = {
          file: entry,
          name: resolveFileNameFromPath(_basename),
          // custom pathmane
          fullPath: _basename,
          treeExpanded: expandedDirs.includes(entry.basename)
        }
      }
    }

    const expandedKeys = expandedDirs.filter(Boolean)
    const shouldReplace =
      repositorySpecifier !== prevRepositorySpecifier || rev !== prevRev
    if (patchMap) {
      updateFileMap(patchMap, shouldReplace)
    }
    if (expandedKeys?.length) {
      if (shouldReplace) {
        setExpandedKeys(new Set(expandedKeys))
      } else {
        setExpandedKeys(keys => {
          const newSet = new Set(keys)
          for (const k of expandedKeys) {
            newSet.add(k)
          }
          return newSet
        })
      }
    }
  }, [entriesResponse])

  React.useEffect(() => {
    if (!initialized) return
    if (!progress && (fetchingRawFile || fetchingTreeEntries)) {
      setProgress(true)
    } else if (!fetchingRawFile && !fetchingTreeEntries) {
      setProgress(false)
    }
  }, [fetchingRawFile, fetchingTreeEntries])

  React.useEffect(() => {
    if (chatSideBarVisible) {
      chatSideBarPanelRef.current?.expand()
      chatSideBarPanelRef.current?.resize(chatSideBarPanelSize)
    } else {
      chatSideBarPanelRef.current?.collapse()
    }
  }, [chatSideBarVisible])

  React.useEffect(() => {
    if (!(fetchingRawFile || fetchingTreeEntries)) return

    const { repositorySpecifier, rev } = activeEntryInfo
    const { repositorySpecifier: prevRepositorySpecifier, rev: prevRev } =
      resolveRepositoryInfoFromPath(prevActivePath.current)
    if (repositorySpecifier !== prevRepositorySpecifier || rev !== prevRev) {
      // cleanup cache
      updateFileMap({}, true)
      setExpandedKeys(new Set())
    }
  }, [activeEntryInfo])

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
        <FileTreePanel fetchingTreeEntries={fetchingTreeEntries} />
      </ResizablePanel>
      <ResizableHandle className="hidden w-1 bg-border/40 hover:bg-border active:bg-blue-500 lg:block" />
      <ResizablePanel defaultSize={80} minSize={30}>
        <div className="flex h-full flex-col">
          <CodeSearchBar
            className={cn(
              'z-40',
              !!activeEntryInfo?.repositorySpecifier ? 'block' : 'hidden'
            )}
          />
          <div className="flex h-full flex-col overflow-y-auto px-4 pb-4">
            {(isTreeMode || isBlobMode) && (
              <FileDirectoryBreadcrumb
                className={cn('pb-4', {
                  'pt-4': !activeEntryInfo?.repositorySpecifier
                })}
              />
            )}
            {!initialized ? (
              <ListSkeleton className="rounded-lg border p-4" />
            ) : showErrorView ? (
              <ErrorView
                className={`rounded-lg border p-4`}
                error={viewAffectingError}
              />
            ) : (
              <>
                {isTreeMode && (
                  <TreeModeView
                    loading={fetchingTreeEntries}
                    initialized={initialized}
                    className={`rounded-lg border`}
                  />
                )}
                {isBlobMode && (
                  <BlobModeView
                    blob={fileBlob}
                    contentLength={contentLength}
                    fileDisplayType={fileDisplayType}
                    loading={fetchingRawFile || fetchingTreeEntries}
                  />
                )}
                {isSearchMode && (
                  <CodeSearchResultView
                    results={repositoryGreps?.files}
                    requestDuration={repositoryGreps?.elapsedMs}
                    loading={fetchingRepositoryGrep}
                  />
                )}
              </>
            )}
          </div>
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
): Promise<FileDisplayType | undefined> {
  if (!blob) return undefined

  const mimeType = blob?.type
  const detectedLanguage = filename2prism(path)?.[0]

  if (mimeType?.startsWith('image')) return 'image'
  if (detectedLanguage || mimeType?.startsWith('text')) return 'text'

  const isReadableText = await isReadableTextFile(blob)
  return isReadableText ? 'text' : 'raw'
}

async function fetchEntriesFromPath(
  path: string | undefined,
  repository: RepositoryListQuery['repositoryList'][0] | undefined
) {
  if (!path) return []
  if (!repository) {
    throw new Error(CodeBrowserError.REPOSITORY_NOT_FOUND)
  }
  if (isEmpty(repository.refs)) {
    throw new Error(CodeBrowserError.REPOSITORY_SYNC_FAILED)
  }

  const { basename, rev, viewMode } = resolveRepositoryInfoFromPath(path)

  if (!rev || !viewMode) throw new Error(CodeBrowserError.INVALID_URL)

  // array of dir basename that do not include the repo name.
  const directoryPaths = getDirectoriesFromBasename(
    basename,
    viewMode === 'tree'
  )
  // fetch all dirs from path
  const requests: Array<() => Promise<ResolveEntriesResponse>> =
    directoryPaths.map(
      dir => () =>
        fetcher(toEntryRequestUrl(repository, rev, dir) as string, {
          responseFormatter(response) {
            const contentType = response.headers.get('Content-Type')
            if (contentType !== ENTRY_CONTENT_TYPE) {
              throw new Error(CodeBrowserError.INVALID_URL)
            }
            return response.json()
          },
          errorHandler() {
            throw new Error(CodeBrowserError.NOT_FOUND)
          }
        })
    )
  const entries = await Promise.all(requests.map(fn => fn()))
  let result: TFile[] = []
  for (let entry of entries) {
    if (entry?.entries?.length) {
      result = [...result, ...entry.entries]
    }
  }
  return result
}

async function fetchRepositoryGrep(
  query: string,
  repository: RepositoryListQuery['repositoryList'][0] | undefined,
  rev: string | undefined
) {
  if (!repository) {
    throw new Error(CodeBrowserError.REPOSITORY_NOT_FOUND)
  }
  const result = client
    .query(repositoryGrepQuery, {
      id: repository.id,
      kind: repository.kind,
      query,
      rev,
      pause: !repository
    })
    .toPromise()

  return result?.then(res => {
    if (res?.error) {
      throw new Error(CodeBrowserError.FAILED_TO_FETCH)
    }

    return res?.data?.repositoryGrep
  })
}

export type { TFileMap, TFileMapItem }

export { SourceCodeBrowserContext, SourceCodeBrowser }
