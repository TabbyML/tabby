'use client'

import React, { PropsWithChildren } from 'react'
import filename2prism from 'filename2prism'
import { compact, findIndex, toNumber } from 'lodash-es'
import { toast } from 'sonner'
import { SWRResponse } from 'swr'
import useSWRImmutable from 'swr/immutable'

import useRouterStuff from '@/lib/hooks/use-router-stuff'
import fetcher from '@/lib/tabby/fetcher'
import type { ResolveEntriesResponse, TFile } from '@/lib/types'
import { cn } from '@/lib/utils'
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup
} from '@/components/ui/resizable'
import { useTopbarProgress } from '@/components/topbar-progress-indicator'

import { FileDirectoryBreadcrumb } from './file-directory-breadcrumb'
import { DirectoryView } from './file-directory-view'
import { mapToFileTree, sortFileTree, type TFileTreeNode } from './file-tree'
import { FileTreePanel } from './file-tree-panel'
import { RawFileView } from './raw-file-view'
import { TextFileView } from './text-file-view'
import {
  getDirectoriesFromPath,
  resolveBasenameFromPath,
  resolveFileNameFromPath,
  resolveRepoNameFromPath
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
}
type TFileMap = Record<string, TFileMapItem>

type SourceCodeBrowserContextValue = {
  activePath: string | undefined
  setActivePath: (path: string | undefined) => void
  fileMap: TFileMap
  updateFileMap: (map: TFileMap) => void
  expandedKeys: Set<string>
  setExpandedKeys: React.Dispatch<React.SetStateAction<Set<string>>>
  toggleExpandedKey: (key: string) => void
  currentFileRoutes: TFileMapItem[]
  initialized: boolean
  setInitialized: React.Dispatch<React.SetStateAction<boolean>>
  fileTreeData: TFileTreeNode[]
}

const SourceCodeBrowserContext =
  React.createContext<SourceCodeBrowserContextValue>(
    {} as SourceCodeBrowserContextValue
  )

const SourceCodeBrowserContextProvider: React.FC<PropsWithChildren> = ({
  children
}) => {
  const { searchParams, updateSearchParams } = useRouterStuff()

  const activePath = React.useMemo(() => {
    return searchParams.get('path')?.toString() ?? ''
  }, [searchParams])

  const setActivePath = (path: string | undefined) => {
    if (!path) {
      updateSearchParams({ del: ['path', 'plain'] })
    } else {
      updateSearchParams({ set: { path }, del: 'plain' })
    }
  }

  const [initialized, setInitialized] = React.useState(false)
  const [fileMap, setFileMap] = React.useState<TFileMap>({})
  const [expandedKeys, setExpandedKeys] = React.useState<Set<string>>(new Set())

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
    const pathSegments = activePath.split('/')
    for (let i = 0; i < pathSegments.length; i++) {
      const p = pathSegments.slice(0, i + 1).join('/')
      result.push(fileMap?.[p])
    }
    return compact(result)
  }, [activePath, fileMap])

  const fileTreeData: TFileTreeNode[] = React.useMemo(() => {
    return sortFileTree(mapToFileTree(fileMap))
  }, [fileMap])

  return (
    <SourceCodeBrowserContext.Provider
      value={{
        initialized,
        setInitialized,
        activePath,
        setActivePath,
        fileMap,
        updateFileMap,
        expandedKeys,
        setExpandedKeys,
        toggleExpandedKey,
        currentFileRoutes,
        fileTreeData
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
    updateFileMap,
    fileMap,
    initialized,
    setInitialized,
    setExpandedKeys
  } = React.useContext(SourceCodeBrowserContext)
  const { progress, setProgress } = useTopbarProgress()

  const activeRepoName = React.useMemo(() => {
    return resolveRepoNameFromPath(activePath)
  }, [activePath])

  const activeBasename = React.useMemo(() => {
    return resolveBasenameFromPath(activePath)
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
  const { data: rawFileResponse, isLoading: isRawFileLoading } =
    useSWRImmutable<{
      blob?: Blob
      contentLength?: number
    }>(
      isFileSelected
        ? `/repositories/${activeRepoName}/resolve/${activeBasename}`
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
        keepPreviousData: true,
        onError(err) {
          toast.error('Fail to fetch')
        }
      }
    )

  React.useEffect(() => {}, [activePath])

  React.useEffect(() => {
    if (isRawFileLoading) {
      setProgress(true)
    } else {
      setProgress(false)
    }
  }, [isRawFileLoading])

  const fileBlob = rawFileResponse?.blob
  const contentLength = rawFileResponse?.contentLength

  // fetch active file meta
  const { data: fileMeta } = useSWRImmutable(
    isFileSelected
      ? `/repositories/${activeRepoName}/meta/${activeBasename}`
      : null,
    fetcher
  )

  // fetch active dir
  const {
    data: subTree,
    isLoading: fetchingSubTree
  }: SWRResponse<ResolveEntriesResponse> = useSWRImmutable(
    shouldFetchSubDir
      ? `/repositories/${activeRepoName}/resolve/${activeBasename}`
      : null,
    fetcher
  )

  const showDirectoryView = activeEntry?.kind === 'dir' || activePath === ''
  const showTextFileView = isFileSelected && fileViewType === 'text'
  const showRawFileView =
    isFileSelected && (fileViewType === 'image' || fileViewType === 'raw')

  React.useEffect(() => {
    const init = async () => {
      const { patchMap, expandedKeys } = await initFileMap(activePath)
      if (patchMap) {
        updateFileMap(patchMap)
      }
      if (expandedKeys?.length) {
        setExpandedKeys(new Set(expandedKeys))
      }
      setInitialized(true)
    }

    init()
  }, [])

  React.useEffect(() => {
    const onFetchSubTree = () => {
      if (subTree?.entries?.length && activePath) {
        const repoName = resolveRepoNameFromPath(activePath)
        let patchMap: TFileMap = {}
        for (const entry of subTree.entries) {
          const path = `${repoName}/${entry.basename}`
          patchMap[path] = {
            file: entry,
            name: resolveFileNameFromPath(path),
            fullPath: path,
            treeExpanded: false
          }
        }
        updateFileMap(patchMap)
        const expandedKeysToAdd = getDirectoriesFromPath(activePath, true)
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

  return (
    <ResizablePanelGroup direction="horizontal" className={cn(className)}>
      <ResizablePanel defaultSize={20} minSize={20}>
        <FileTreePanel />
      </ResizablePanel>
      <ResizableHandle className="w-1 bg-border/40 hover:bg-border active:bg-border" />
      <ResizablePanel defaultSize={80} minSize={30}>
        <div className="flex h-full flex-col overflow-y-auto px-4 pb-4">
          <FileDirectoryBreadcrumb className="py-4" />
          <div>
            {showDirectoryView && (
              <DirectoryView
                loading={fetchingSubTree}
                initialized={initialized}
                className={`rounded-lg border`}
              />
            )}
            {showTextFileView && (
              <TextFileView
                blob={fileBlob}
                meta={fileMeta}
                contentLength={contentLength}
              />
            )}
            {showRawFileView && (
              <RawFileView
                blob={fileBlob}
                isImage={fileViewType === 'image'}
                contentLength={contentLength}
              />
            )}
          </div>
        </div>
      </ResizablePanel>
    </ResizablePanelGroup>
  )
}

const SourceCodeBrowser: React.FC<SourceCodeBrowserProps> = props => {
  return (
    <SourceCodeBrowserContextProvider>
      <SourceCodeBrowserRenderer className="source-code-browser" {...props} />
    </SourceCodeBrowserContextProvider>
  )
}

async function initFileMap(path?: string) {
  const defaultRepositoryName = resolveRepoNameFromPath(path)
  const defaultBasename = resolveBasenameFromPath(path)

  try {
    const repos = await fetchRepositories()
    const { defaultEntries, expandedDir } = await initDefaultEntries(repos)

    const patchMap: TFileMap = {}
    for (const repo of repos) {
      patchMap[repo.basename] = {
        file: repo,
        name: repo.basename,
        fullPath: repo.basename,
        treeExpanded: repo.basename === defaultRepositoryName
      }
    }
    for (const entry of defaultEntries) {
      const path = `${defaultRepositoryName}/${entry.basename}`
      patchMap[path] = {
        file: entry,
        name: resolveFileNameFromPath(path),
        fullPath: path,
        treeExpanded: expandedDir.includes(entry.basename)
      }
    }
    const expandedKeys = expandedDir.map(dir =>
      [defaultRepositoryName, dir].filter(Boolean).join('/')
    )

    return { patchMap, expandedKeys }
  } catch (e) {
    console.error(e)
    return {}
  }

  async function fetchRepositories(): Promise<TFile[]> {
    try {
      const repos: ResolveEntriesResponse = await fetcher(
        '/repositories/resolve/'
      )
      return repos?.entries
    } catch (e) {
      return []
    }
  }

  async function fetchDefaultEntries(data?: TFile[]) {
    try {
      // if (!accessToken) return undefined

      if (!defaultRepositoryName) return undefined
      // match default repository
      const repositoryIdx = findIndex(
        data,
        entry => entry.basename === defaultRepositoryName
      )
      if (repositoryIdx < 0) return undefined

      const directoryPaths = getDirectoriesFromPath(defaultBasename)
      // fetch default directories
      const requests: Array<() => Promise<ResolveEntriesResponse>> =
        directoryPaths.map(path => () => {
          return fetcher(
            `/repositories/${defaultRepositoryName}/resolve/${path}`
          )
        })
      const entries = await Promise.all(requests.map(fn => fn()))
      let result: TFile[] = []
      for (let entry of entries) {
        if (entry?.entries?.length) {
          result = [...result, ...entry.entries]
        }
      }
      return result
    } catch (e) {
      console.error(e)
    }
  }

  async function initDefaultEntries(data?: TFile[]) {
    let result: { defaultEntries: TFile[]; expandedDir: string[] } = {
      defaultEntries: [],
      expandedDir: []
    }
    try {
      if (defaultRepositoryName && data?.length) {
        const defaultEntries = await fetchDefaultEntries(data)
        const expandedDir = getDirectoriesFromPath(defaultBasename)
        if (defaultEntries?.length) {
          result.defaultEntries = defaultEntries
        }
        if (expandedDir?.length) {
          result.expandedDir = expandedDir
        }
      }
    } catch (e) {
      console.error(e)
    }
    return result
  }
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

export type { TFileMap, TFileMapItem }

export { SourceCodeBrowserContext, SourceCodeBrowser, getFileViewType }
