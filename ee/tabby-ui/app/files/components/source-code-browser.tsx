'use client'

import React, { PropsWithChildren, useState } from 'react'
import dynamic from 'next/dynamic'
import { compact, findIndex, has } from 'lodash-es'
import { SWRResponse } from 'swr'
import useSWRImmutable from 'swr/immutable'

import useRouterStuff from '@/lib/hooks/use-router-stuff'
import fetcher from '@/lib/tabby/fetcher'
import type { ResolveEntriesResponse, TFile, TFileMeta } from '@/lib/types'
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
import {
  getDirectoriesFromPath,
  resolveBasenameFromPath,
  resolveFileNameFromPath,
  resolveRepoNameFromPath
} from './utils'

const SourceCodeEditor = dynamic(() => import('./source-code-editor'), {
  ssr: false
})

type TCodeMap = Record<string, string>
type TFileMetaMap = Record<string, TFileMeta>
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
  codeMap: Record<string, string>
  setCodeMap: React.Dispatch<React.SetStateAction<TCodeMap>>
  fileMetaMap: TFileMetaMap
  setFileMetaMap: React.Dispatch<React.SetStateAction<TFileMetaMap>>
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
  const [codeMap, setCodeMap] = useState<TCodeMap>({})
  const [fileMetaMap, setFileMetaMap] = useState<TFileMetaMap>({})
  const [expandedKeys, setExpandedKeys] = React.useState<Set<string>>(new Set())

  const updateFileMap = (map: TFileMap) => {
    if (!map) return

    setFileMap({
      ...fileMap,
      ...map
    })
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
        codeMap,
        setCodeMap,
        fileMetaMap,
        setFileMetaMap,
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

const SourceCodeBrowserRenderer: React.FC<SourceCodeBrowserProps> = ({
  className
}) => {
  const {
    activePath,
    setActivePath,
    codeMap,
    setCodeMap,
    updateFileMap,
    fileMetaMap,
    setFileMetaMap,
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

  const shouldFetchRawFile = React.useMemo(() => {
    const isFile = activePath && fileMap?.[activePath]?.file?.kind === 'file'
    return isFile && !has(codeMap, activePath)
  }, [activePath, fileMap, codeMap])

  const shouldFetchFileMeta = React.useMemo(() => {
    const isFile = activePath && fileMap?.[activePath]?.file?.kind === 'file'
    return isFile && !has(fileMetaMap, activePath)
  }, [activePath, fileMap, codeMap])

  const shouldFetchSubDir = React.useMemo(() => {
    if (!initialized) return false

    const isDir = activePath && fileMap?.[activePath]?.file?.kind === 'dir'
    return isDir && !fileMap?.[activePath]?.treeExpanded
  }, [activePath, fileMap, initialized])

  // fetch raw file
  const { data: fileContent } = useSWRImmutable(
    shouldFetchRawFile
      ? `/repositories/${activeRepoName}/resolve/${activeBasename}`
      : null,
    (url: string) => fetcher(url, { format: 'text' })
  )

  // fetch active file meta
  const { data: fileMeta } = useSWRImmutable(
    shouldFetchFileMeta
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
    const afterFetchRawFile = () => {
      if (fileContent && activePath) {
        setCodeMap(map => ({
          ...map,
          [activePath]: fileContent
        }))
      }
    }

    afterFetchRawFile()
  }, [fileContent])

  React.useEffect(() => {
    const afterFetchMeta = () => {
      if (fileMeta && activePath) {
        setFileMetaMap(map => ({
          ...map,
          [activePath]: fileMeta
        }))
      }
    }

    afterFetchMeta()
  }, [fileMeta])

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

  const activeEntry = activePath ? fileMap?.[activePath]?.file : undefined
  const showEditor = activeEntry?.kind === 'file'
  const showDirectoryPanel = activeEntry?.kind === 'dir' || activePath === ''

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
            <DirectoryPanel
              loading={fetchingSubTree}
              initialized={initialized}
              className={`rounded-lg border ${
                showDirectoryPanel ? 'block' : 'hidden'
              }`}
            />
            <SourceCodeEditor
              className={`rounded-lg border py-2 ${
                showEditor ? 'block' : 'hidden'
              }`}
            />
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

export type { TFileMap, TFileMapItem }

export { SourceCodeBrowserContext, SourceCodeBrowser }
