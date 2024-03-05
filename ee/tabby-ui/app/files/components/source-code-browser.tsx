'use client'

import React, { PropsWithChildren } from 'react'
import dynamic from 'next/dynamic'
import filename2prism from 'filename2prism'
import { compact, findIndex } from 'lodash-es'
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

import { FileDirectoryBreadcrumb } from './file-directory-breadcrumb'
import { DirectoryPanel } from './file-directory-panel'
import {
  mapToFileTree,
  FileTree as RepositoriesFileTree,
  sortFileTree,
  type TFileTreeNode
} from './file-tree'
import { RawContentPanel } from './raw-content-panel'
import {
  getDirectoriesFromPath,
  resolveBasenameFromPath,
  resolveFileNameFromPath,
  resolveRepoNameFromPath
} from './utils'

const SourceCodeEditor = dynamic(() => import('./source-code-editor'), {
  ssr: false
})

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

type SourceCodeBrowserProviderProps = {}

const SourceCodeBrowserContext =
  React.createContext<SourceCodeBrowserContextValue>(
    {} as SourceCodeBrowserContextValue
  )

const SourceCodeBrowserContextProvider: React.FC<
  PropsWithChildren<SourceCodeBrowserProviderProps>
> = ({ children }) => {
  const { searchParams, updateSearchParams } = useRouterStuff()

  const activePath = React.useMemo(() => {
    return searchParams.get('path')?.toString() ?? ''
  }, [searchParams])

  const setActivePath = (path: string | undefined) => {
    if (!path) {
      updateSearchParams({ del: 'path' })
    } else {
      updateSearchParams({ set: { path } })
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
    setActivePath,
    updateFileMap,
    fileMap,
    initialized,
    setInitialized,
    expandedKeys,
    toggleExpandedKey,
    setExpandedKeys,
    fileTreeData
  } = React.useContext(SourceCodeBrowserContext)

  const activeRepoName = React.useMemo(() => {
    return resolveRepoNameFromPath(activePath)
  }, [activePath])

  const activeBasename = React.useMemo(() => {
    return resolveBasenameFromPath(activePath)
  }, [activePath])
  const [fileViewType, setFileViewType] = React.useState<FileDisplayType>()

  const isFileActive =
    activePath && fileMap?.[activePath]?.file?.kind === 'file'
  const activeEntry = activePath ? fileMap?.[activePath]?.file : undefined

  const shouldFetchSubDir = React.useMemo(() => {
    if (!initialized) return false

    const isDir = activePath && fileMap?.[activePath]?.file?.kind === 'dir'
    return isDir && !fileMap?.[activePath]?.treeExpanded
  }, [activePath, fileMap, initialized])

  // fetch raw file
  const { data: fileBlob } = useSWRImmutable(
    isFileActive
      ? `/repositories/${activeRepoName}/resolve/${activeBasename}`
      : null,
    (url: string) =>
      fetcher(url, {
        format: 'blob'
      })
  )

  // fetch active file meta
  const { data: fileMeta } = useSWRImmutable(
    isFileActive
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

  const showDirectoryPanel = activeEntry?.kind === 'dir' || activePath === ''

  const showRawFilePanel = fileViewType === 'image' || fileViewType === 'raw'
  const showCodeEditor = fileViewType === 'text'

  const onSelectTreeNode = (treeNode: TFileTreeNode) => {
    setActivePath(treeNode.fullPath)
  }

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
    const afterFetchSubTree = () => {
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

    afterFetchSubTree()
  }, [subTree])

  React.useEffect(() => {
    const calculateViewType = async () => {
      const displayType = await getFileViewType(activePath ?? '', fileBlob)
      setFileViewType(displayType)
    }

    if (isFileActive) {
      calculateViewType()
    } else {
      setFileViewType('')
    }
  }, [activePath, isFileActive, fileBlob])

  return (
    <ResizablePanelGroup direction="horizontal" className={cn(className)}>
      <ResizablePanel defaultSize={20} minSize={20}>
        <div className="h-full overflow-hidden py-2">
          <RepositoriesFileTree
            className="h-full overflow-y-auto overflow-x-hidden px-4"
            onSelectTreeNode={onSelectTreeNode}
            activePath={activePath}
            fileMap={fileMap}
            updateFileMap={updateFileMap}
            expandedKeys={expandedKeys}
            toggleExpandedKey={toggleExpandedKey}
            initialized={initialized}
            fileTreeData={fileTreeData}
          />
        </div>
      </ResizablePanel>
      <ResizableHandle className="w-1 hover:bg-card active:bg-card" />
      <ResizablePanel defaultSize={80} minSize={30}>
        <div className="flex h-full flex-col overflow-y-auto px-4 pb-4">
          <FileDirectoryBreadcrumb className="sticky top-0 z-10 bg-background py-4" />
          <div className="flex-1">
            {/* {isFileActive && fileViewType === '' && <ListSkeleton />} */}
            {showDirectoryPanel && (
              <DirectoryPanel
                loading={fetchingSubTree}
                initialized={initialized}
                className={`rounded-lg border`}
              />
            )}
            {showCodeEditor && (
              <SourceCodeEditor
                className={`rounded-lg border py-2`}
                blob={fileBlob}
                meta={fileMeta}
                // key={activePath}
              />
            )}
            {showRawFilePanel && fileBlob && (
              <RawContentPanel
                className={`rounded-lg border py-2`}
                blob={fileBlob}
                isImage={fileViewType === 'image'}
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
        if (entry.entries?.length) {
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
        const nonPrintableRegex = /[\x00-\x08\x0E-\x1F\x7F-\x9F]/
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

    reader.readAsText(blobPart, 'UTF-8') // 假设文件是 UTF-8 编码
  })
}

export type { TFileMap, TFileMapItem }

export { SourceCodeBrowserContext, SourceCodeBrowser, getFileViewType }
