'use client'

import React from 'react'
import { findIndex, isNil } from 'lodash-es'
import { SWRResponse } from 'swr'
import useSWRImmutable from 'swr/immutable'

import { useDebounce } from '@/lib/hooks/use-debounce'
import { useAuthenticatedApi, useSession } from '@/lib/tabby/auth'
import fetcher from '@/lib/tabby/fetcher'
import { cn } from '@/lib/utils'
import {
  IconChevronDown,
  IconChevronRight,
  IconDirectoryExpandSolid,
  IconDirectorySolid,
  IconFile,
  IconSpinner
} from '@/components/ui/icons'

type TFile = {
  kind: 'file' | 'dir'
  basename: string
}

/**
 * FileMap example
 * {
 *   'https_github.com_TabbyML_tabby.git/ee/tabby-ui/README.md': {
 *     file: {
 *      kind: 'file',
 *      basename: 'ee/tabby-ui/README.md'
 *     },
 *     treeExpanded: false
 *   },
 *   ...
 * }
 */
type TFileMap = Record<
  string,
  {
    file: TFile
    treeExpanded?: boolean
  }
>

interface FileTreeProps extends React.HTMLAttributes<HTMLDivElement> {
  onSelectTreeNode?: (treeNode: TFileTreeNode, repositoryName: string) => void
  repositoryName: string
  activePath?: string
  defaultEntries?: TFile[]
  defaultExpandedKeys?: string[]
}

interface RepositoriesFileTreeProps
  extends Omit<FileTreeProps, 'repositoryName'> {
  defaultRepository?: string
  defaultBasename?: string
}

type FileTreeProviderProps = React.PropsWithChildren<{
  onSelectTreeNode?: (treeNode: TFileTreeNode, repositoryName: string) => void
  repositoryName: string
  activePath?: string
  initialFileMap?: TFileMap
  defaultExpandedKeys?: string[]
}>

type TFileTreeNode = {
  name: string
  file: TFile
  children?: Array<TFileTreeNode>
}

type FileTreeContextValue = {
  repositoryName: string
  fileMap: TFileMap
  updateFileMap: (map: TFileMap) => void
  fileTree: TFileTreeNode
  onSelectTreeNode: FileTreeProps['onSelectTreeNode']
  expandedKeys: Set<string>
  toggleExpandedKey: (key: string) => void
  activePath: string | undefined
}

type DirectoryTreeNodeProps = {
  node: TFileTreeNode
  level: number
  root?: boolean
}
type FileTreeNodeProps = {
  node: TFileTreeNode
  level: number
}

interface FileTreeNodeViewProps extends React.HTMLAttributes<HTMLDivElement> {
  isActive?: boolean
  level: number
}

interface DirectoryTreeNodeViewProps
  extends React.HTMLAttributes<HTMLDivElement> {
  isActive?: boolean
  level: number
}

type ResolveEntriesResponse = { entries: TFile[] }

const FileTreeContext = React.createContext<FileTreeContextValue>(
  {} as FileTreeContextValue
)

const FileTreeProvider: React.FC<
  React.PropsWithChildren<FileTreeProviderProps>
> = ({
  onSelectTreeNode,
  children,
  repositoryName,
  initialFileMap,
  activePath,
  defaultExpandedKeys = []
}) => {
  const [fileMap, setFileMap] = React.useState<TFileMap>(initialFileMap ?? {})
  const [expandedKeys, setExpandedKeys] = React.useState<Set<string>>(
    new Set(defaultExpandedKeys)
  )

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

  const fileTreeData: TFileTreeNode = React.useMemo(() => {
    const rootTree = mapToFileTree(fileMap, repositoryName)
    sortFileTree(rootTree.children || [])

    return rootTree
  }, [fileMap, repositoryName])

  return (
    <FileTreeContext.Provider
      value={{
        fileMap,
        updateFileMap,
        onSelectTreeNode,
        fileTree: fileTreeData,
        repositoryName,
        expandedKeys,
        toggleExpandedKey,
        activePath
      }}
    >
      {children}
    </FileTreeContext.Provider>
  )
}

const GridArea = ({ level }: { level: number }) => {
  const items = React.useMemo(() => {
    return new Array(level).fill(1)
  }, [level])

  return (
    <div className="flex h-full shrink-0 items-stretch">
      {items.map((_item, index) => {
        return (
          <div
            key={index}
            className="flex h-8 w-2 border-r border-transparent transition-colors duration-300 group-focus-within:border-border group-hover:border-border group-focus-visible:border-border"
          />
        )
      })}
    </div>
  )
}

/**
 * Display FileTreeNode
 */
const FileTreeNodeView: React.FC<
  React.PropsWithChildren<FileTreeNodeViewProps>
> = ({ isActive, level, children, className, ...props }) => {
  return (
    <div
      className={cn(
        'flex cursor-pointer items-stretch rounded-sm hover:bg-accent focus:bg-accent focus:text-accent-foreground',
        isActive && 'bg-accent',
        className
      )}
      {...props}
    >
      <GridArea level={level} />
      <div className="flex flex-nowrap items-center gap-2 truncate whitespace-nowrap py-1">
        <div className="h-4 w-4 shrink-0"></div>
        {children}
      </div>
    </div>
  )
}

/**
 * Display DirectoryTreeNode
 */
const DirectoryTreeNodeView: React.FC<
  React.PropsWithChildren<DirectoryTreeNodeViewProps>
> = ({ children, level, className, ...props }) => {
  return (
    <div
      className={cn(
        'flex cursor-pointer items-stretch rounded-sm hover:bg-accent focus:bg-accent focus:text-accent-foreground',
        className
      )}
      {...props}
    >
      <GridArea level={level} />
      <div className="flex flex-nowrap items-center gap-2 truncate whitespace-nowrap py-1">
        {children}
      </div>
    </div>
  )
}

const FileTreeNode: React.FC<FileTreeNodeProps> = ({ node, level }) => {
  const { onSelectTreeNode, activePath, repositoryName } =
    React.useContext(FileTreeContext)
  const isFile = node.file.kind === 'file'
  const isActive = `${repositoryName}/${node.file.basename}` === activePath

  const handleSelect: React.MouseEventHandler<HTMLDivElement> = e => {
    if (isFile) {
      onSelectTreeNode?.(node, repositoryName)
    }
  }

  return (
    <FileTreeNodeView level={level} onClick={handleSelect} isActive={isActive}>
      <IconFile className="shrink-0" />
      <div className="truncate">{node?.name}</div>
    </FileTreeNodeView>
  )
}

const DirectoryTreeNode: React.FC<DirectoryTreeNodeProps> = ({
  node,
  level,
  root
}) => {
  const {
    repositoryName,
    fileMap,
    updateFileMap,
    expandedKeys,
    toggleExpandedKey
  } = React.useContext(FileTreeContext)

  const basename = root ? '' : node.file.basename
  const expanded = expandedKeys.has(basename)
  const shouldFetchChildren =
    node.file.kind === 'dir' && !fileMap?.[basename]?.treeExpanded && expanded

  const { data, isValidating }: SWRResponse<ResolveEntriesResponse> =
    useSWRImmutable(
      useAuthenticatedApi(
        shouldFetchChildren
          ? `/repositories/${repositoryName}/resolve/${basename}`
          : null
      ),
      fetcher,
      {
        revalidateIfStale: false
      }
    )

  React.useEffect(() => {
    if (data?.entries?.length) {
      const patchMap: TFileMap = data.entries.reduce((sum, cur) => {
        return {
          ...sum,
          [`${repositoryName}/${cur.basename}`]: {
            file: cur,
            treeExpanded: false
          }
        }
      }, {} as TFileMap)

      updateFileMap(patchMap)
    }
  }, [data])

  const loading = useDebounce(isValidating, 100)

  const existingChildren = !!node?.children?.length

  return (
    <>
      <DirectoryTreeNodeView
        level={level}
        onClick={e => toggleExpandedKey(basename)}
      >
        <div className="shrink-0">
          {loading ? (
            <IconSpinner />
          ) : expanded ? (
            <IconChevronDown />
          ) : (
            <IconChevronRight />
          )}
        </div>
        <div className="shrink-0" style={{ color: 'rgb(84, 174, 255)' }}>
          {expanded ? <IconDirectoryExpandSolid /> : <IconDirectorySolid />}
        </div>
        <div className="truncate">{node?.name}</div>
      </DirectoryTreeNodeView>
      <>
        {expanded && existingChildren ? (
          <>
            {node.children?.map(child => {
              const key = child.file.basename
              return child.file.kind === 'dir' ? (
                <DirectoryTreeNode key={key} node={child} level={level + 1} />
              ) : (
                <FileTreeNode key={key} node={child} level={level + 1} />
              )
            })}
          </>
        ) : null}
      </>
    </>
  )
}

const FileTreeRootRenderer: React.FC = () => {
  const { fileTree } = React.useContext(FileTreeContext)

  return <DirectoryTreeNode root level={0} node={fileTree} />
}

const FileTree: React.FC<FileTreeProps> = ({
  onSelectTreeNode,
  repositoryName,
  className,
  activePath,
  defaultEntries,
  defaultExpandedKeys,
  ...props
}) => {
  const initialFileMap: TFileMap = React.useMemo(() => {
    const map: TFileMap = {
      [repositoryName]: {
        file: {
          kind: 'dir',
          basename: repositoryName
        },
        treeExpanded: false
      }
    }
    if (!defaultEntries?.length) {
      return map
    } else {
      return buildFileMapFromEntries(
        defaultEntries,
        repositoryName,
        defaultExpandedKeys
      )
    }
  }, [])

  return (
    <div className={cn(className)} {...props}>
      <FileTreeProvider
        onSelectTreeNode={onSelectTreeNode}
        repositoryName={repositoryName}
        activePath={activePath}
        initialFileMap={initialFileMap}
        defaultExpandedKeys={defaultExpandedKeys}
      >
        <FileTreeRootRenderer />
      </FileTreeProvider>
    </div>
  )
}

const RepositoriesFileTree: React.FC<RepositoriesFileTreeProps> = ({
  onSelectTreeNode,
  activePath,
  className,
  defaultBasename,
  defaultRepository,
  ...props
}) => {
  const { data } = useSession()
  const accessToken = data?.accessToken
  const [initialized, setInitialized] = React.useState(false)
  const [defaultEntries, setDefaultEntries] = React.useState<TFile[]>()
  const [defaultExpandedKeys, setDefaultExpandedKeys] =
    React.useState<string[]>()

  const fetchDefaultEntries = async (data?: ResolveEntriesResponse) => {
    try {
      if (!accessToken) return undefined
      if (!defaultRepository || !defaultBasename) return undefined
      // match default repository
      const repositoryIdx = findIndex(
        data?.entries,
        entry => entry.basename === defaultRepository
      )
      if (repositoryIdx < 0) return undefined

      const directoryPaths = getDirectoriesFromBasename(defaultBasename)
      const requests: Array<() => Promise<ResolveEntriesResponse>> =
        directoryPaths.map(path => () => {
          return fetcher([
            `/repositories/${defaultRepository}/resolve/${path}`,
            accessToken
          ])
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

  const initDefaultEntries = async (data?: ResolveEntriesResponse) => {
    try {
      if (defaultBasename && defaultRepository && data) {
        const defaultEntriesRes = await fetchDefaultEntries(data)
        setDefaultEntries(defaultEntriesRes)

        const expandedKeys = getDirectoriesFromBasename(defaultBasename)
        setDefaultExpandedKeys(expandedKeys)
      }
    } catch (e) {
      console.error(e)
    }

    setTimeout(() => {
      setInitialized(true)
    })
  }

  const { data: repositories, error }: SWRResponse<ResolveEntriesResponse> =
    useSWRImmutable(useAuthenticatedApi(`/repositories/resolve/`), fetcher, {
      onSuccess: (data: ResolveEntriesResponse) => {
        initDefaultEntries(data)
      },
      onError: () => {
        initDefaultEntries()
      },
      revalidateOnMount: true,
      revalidateIfStale: false
    })

  if (!initialized) return <FileTreeSkeleton />

  if (error)
    return (
      <div className="mt-1 flex justify-center">error {error?.message}</div>
    )

  if (!repositories?.entries?.length) {
    return <div className="mt-1 flex justify-center">No Data</div>
  }

  return (
    <div className={cn('group', className)} {...props}>
      {repositories?.entries.map(entry => {
        return (
          <FileTree
            key={entry.basename}
            repositoryName={entry.basename}
            activePath={activePath}
            onSelectTreeNode={onSelectTreeNode}
            defaultEntries={
              entry.basename === defaultRepository ? defaultEntries : undefined
            }
            defaultExpandedKeys={
              entry.basename === defaultRepository
                ? defaultExpandedKeys
                : undefined
            }
          />
        )
      })}
    </div>
  )
}

function mapToFileTree(fileMap: TFileMap, rootBasename: string): TFileTreeNode {
  const tree: TFileTreeNode[] = []

  const fileKeys = Object.keys(fileMap)
  const childrenKeys = fileKeys.filter(k => k !== rootBasename)
  const rootNode = fileMap?.[rootBasename]

  for (const fileKey of childrenKeys) {
    const file = fileMap[fileKey]
    const pathSegments = file.file.basename.split('/')
    let currentNode = tree

    for (let i = 0; i < pathSegments.length; i++) {
      const segment = pathSegments[i]
      const existingNode = currentNode?.find(node => node.name === segment)

      if (existingNode) {
        currentNode = existingNode.children || []
      } else {
        const newNode: TFileTreeNode = {
          file: file.file,
          name: segment,
          children: []
        }
        currentNode.push(newNode)
        currentNode = newNode.children as TFileTreeNode[]
      }
    }
  }

  return {
    file: rootNode?.file || { kind: 'dir', basename: rootBasename },
    name: resolveRepositoryName(rootBasename),
    children: tree
  }
}

function buildFileMapFromEntries(
  entries: TFile[],
  repositoryName: string,
  defaultExpandedKeys?: string[]
): TFileMap {
  let map: TFileMap = {}
  for (const entry of entries) {
    map[`${repositoryName}/${entry.basename}`] = {
      file: entry,
      treeExpanded:
        entry.basename === repositoryName ||
        defaultExpandedKeys?.includes(entry.basename)
    }
  }
  return map
}

function FileTreeSkeleton() {
  return (
    <ul className="duration-600 animate-pulse space-y-3 p-2">
      <li className="h-4 rounded-md bg-gray-200 dark:bg-gray-700"></li>
      <li className="ml-4 h-4 rounded-md bg-gray-200 dark:bg-gray-700"></li>
      <li className="ml-4 h-4 rounded-md bg-gray-200 dark:bg-gray-700"></li>
      <li className="h-4 rounded-md bg-gray-200 dark:bg-gray-700"></li>
      <li className="ml-4 h-4 rounded-md bg-gray-200 dark:bg-gray-700"></li>
    </ul>
  )
}

function sortFileTree(tree: TFileTreeNode[]): TFileTreeNode[] {
  if (!tree.length) return []

  tree.sort((a, b) => {
    const aIsFile = a.file.kind === 'file' ? 1 : 0
    const bIsFile = b.file.kind === 'file' ? 1 : 0
    return aIsFile - bIsFile || a.name.localeCompare(b.name)
  })
  for (let item of tree) {
    if (item?.children) {
      sortFileTree(item.children)
    }
  }

  return tree
}

function resolveRepositoryName(name: string) {
  const repositoryName = /https_github.com_(\w+).git/.exec(name)?.[1]
  return repositoryName || name
}

function getDirectoriesFromBasename(basename: string): string[] {
  if (isNil(basename)) return []

  let result = ['']
  const pathSegments = basename.split('/')
  for (let i = 0; i < pathSegments.length - 1; i++) {
    result.push(pathSegments.slice(0, i + 1).join('/'))
  }
  return result
}

export { RepositoriesFileTree, type TFile, type TFileTreeNode }
