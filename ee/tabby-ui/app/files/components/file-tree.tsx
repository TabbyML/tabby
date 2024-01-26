'use client'

import React from 'react'
import { SWRResponse } from 'swr'
import useSWRImmutable from 'swr/immutable'

import { useDebounce } from '@/lib/hooks/use-debounce'
import fetcher from '@/lib/tabby/fetcher'
import type { ResolveEntriesResponse, TFile } from '@/lib/types'
import { cn } from '@/lib/utils'
import {
  IconChevronDown,
  IconChevronRight,
  IconDirectoryExpandSolid,
  IconDirectorySolid,
  IconFile,
  IconSpinner
} from '@/components/ui/icons'

import { TFileMap } from './source-code-browser'
import { resolveFileNameFromPath, resolveRepoNameFromPath } from './utils'

type TFileTreeNode = {
  name: string
  file: TFile
  fullPath: string
  children?: Array<TFileTreeNode>
}

interface FileTreeProps extends React.HTMLAttributes<HTMLDivElement> {
  onSelectTreeNode?: (treeNode: TFileTreeNode) => void
  activePath?: string
  fileMap: TFileMap
  updateFileMap: (map: TFileMap) => void
  expandedKeys: Set<string>
  toggleExpandedKey: (key: string) => void
  initialized: boolean
  fileTreeData: TFileTreeNode[]
}

interface FileTreeProviderProps extends FileTreeProps {}

type FileTreeContextValue = {
  fileMap: TFileMap
  updateFileMap: (map: TFileMap) => void
  fileTreeData: TFileTreeNode[]
  onSelectTreeNode: FileTreeProps['onSelectTreeNode']
  expandedKeys: Set<string>
  toggleExpandedKey: (key: string) => void
  activePath: string | undefined
  initialized: boolean
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

const FileTreeContext = React.createContext<FileTreeContextValue>(
  {} as FileTreeContextValue
)

const FileTreeProvider: React.FC<
  React.PropsWithChildren<FileTreeProviderProps>
> = ({
  onSelectTreeNode,
  children,
  activePath,
  fileMap,
  updateFileMap,
  expandedKeys,
  toggleExpandedKey,
  initialized,
  fileTreeData
}) => {
  return (
    <FileTreeContext.Provider
      value={{
        onSelectTreeNode,
        fileTreeData,
        expandedKeys,
        toggleExpandedKey,
        activePath,
        fileMap,
        updateFileMap,
        initialized
      }}
    >
      {children}
    </FileTreeContext.Provider>
  )
}

const GridArea: React.FC<{ level: number }> = ({ level }) => {
  const items = React.useMemo(() => {
    return new Array(level).fill(1)
  }, [level])

  return (
    <div className="flex h-full shrink-0 items-stretch">
      {items.map((_item, index) => {
        return (
          <div
            key={index}
            className="flex h-8 w-2 border-r border-transparent transition-colors duration-300 group-hover/filetree:border-border"
          />
        )
      })}
    </div>
  )
}

const ActiveViewBar = () => {
  return <div className="absolute -left-2 h-8 w-1 rounded-md bg-primary" />
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
        'relative flex cursor-pointer items-stretch rounded-sm hover:bg-accent focus:bg-accent focus:text-accent-foreground',
        isActive && 'bg-accent',
        className
      )}
      {...props}
    >
      {isActive && <ActiveViewBar />}
      <GridArea level={level} />
      <div className="flex flex-nowrap items-center gap-2 truncate whitespace-nowrap">
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
> = ({ children, level, isActive, className, ...props }) => {
  return (
    <div
      className={cn(
        'relative flex cursor-pointer items-stretch rounded-sm hover:bg-accent focus:bg-accent focus:text-accent-foreground',
        isActive ? 'bg-accent text-accent-foreground' : undefined,
        className
      )}
      {...props}
    >
      {isActive && <ActiveViewBar />}
      <GridArea level={level} />
      <div className="flex flex-nowrap items-center gap-2 truncate whitespace-nowrap">
        {children}
      </div>
    </div>
  )
}

const FileTreeNode: React.FC<FileTreeNodeProps> = ({ node, level }) => {
  const { onSelectTreeNode, activePath } = React.useContext(FileTreeContext)

  const isFile = node.file.kind === 'file'
  const isActive = node.fullPath === activePath

  const handleSelect: React.MouseEventHandler<HTMLDivElement> = e => {
    if (isFile) {
      onSelectTreeNode?.(node)
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
    fileMap,
    updateFileMap,
    expandedKeys,
    toggleExpandedKey,
    onSelectTreeNode,
    activePath
  } = React.useContext(FileTreeContext)

  const initialized = React.useRef(false)
  const repositoryName = React.useMemo(() => {
    return resolveRepoNameFromPath(node.fullPath)
  }, [node.fullPath])
  const basename = root ? '' : node.file.basename
  const expanded = expandedKeys.has(node.fullPath)
  const shouldFetchChildren =
    node.file.kind === 'dir' &&
    !fileMap?.[node.fullPath]?.treeExpanded &&
    expanded

  const { data, isValidating }: SWRResponse<ResolveEntriesResponse> =
    useSWRImmutable(
      shouldFetchChildren
        ? `/repositories/${repositoryName}/resolve/${basename}`
        : null,
      fetcher,
      {
        revalidateIfStale: false
      }
    )

  React.useEffect(() => {
    if (initialized.current) return

    if (data?.entries?.length) {
      const patchMap: TFileMap = data.entries.reduce((sum, cur) => {
        const path = `${repositoryName}/${cur.basename}`
        return {
          ...sum,
          [path]: {
            file: cur,
            name: resolveFileNameFromPath(path),
            fullPath: path,
            treeExpanded: false
          }
        }
      }, {} as TFileMap)

      updateFileMap(patchMap)
      initialized.current = true
    }
  }, [data])

  const onSelectDirectory: React.MouseEventHandler<HTMLDivElement> = e => {
    onSelectTreeNode?.(node)
  }

  const loading = useDebounce(isValidating, 100)

  const existingChildren = !!node?.children?.length

  return (
    <>
      <DirectoryTreeNodeView
        level={level}
        onClick={onSelectDirectory}
        isActive={activePath === node.fullPath}
      >
        <div
          className="flex h-8 shrink-0 items-center hover:bg-primary/10 hover:text-popover-foreground"
          onClick={e => {
            if (loading) return
            toggleExpandedKey(node.fullPath)
            e.stopPropagation()
          }}
        >
          {loading && !initialized.current ? (
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

const FileTreeRenderer: React.FC = () => {
  const { fileTreeData, initialized } = React.useContext(FileTreeContext)

  if (!initialized) return <FileTreeSkeleton />

  if (!fileTreeData?.length) return <div>No Data</div>

  return (
    <>
      {fileTreeData?.map(node => (
        <DirectoryTreeNode root level={0} node={node} key={node.fullPath} />
      ))}
    </>
  )
}

const FileTree: React.FC<FileTreeProps> = ({ className, ...props }) => {
  return (
    <div className={cn('group/filetree', className)}>
      <FileTreeProvider {...props}>
        <FileTreeRenderer />
      </FileTreeProvider>
    </div>
  )
}

function mapToFileTree(fileMap: TFileMap | undefined): TFileTreeNode[] {
  const tree: TFileTreeNode[] = []
  if (!fileMap) return tree

  const fileKeys = Object.keys(fileMap)
  for (const fileKey of fileKeys) {
    const file = fileMap[fileKey]
    const pathSegments = fileKey.split('/')
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
          fullPath: fileKey,
          children: []
        }
        currentNode.push(newNode)
        currentNode = newNode.children as TFileTreeNode[]
      }
    }
  }

  return tree
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

function sortFileTree(tree: TFileTreeNode[]) {
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

export type { TFileTreeNode }
export { FileTree, mapToFileTree, sortFileTree }
