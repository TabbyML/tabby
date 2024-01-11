'use client'

import React from 'react'
import { SWRResponse } from 'swr'
import useSWRImmutable from 'swr/immutable'

import { useDebounce } from '@/lib/hooks/use-debounce'
import { useAuthenticatedApi } from '@/lib/tabby/auth'
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

type TFileMap = Record<
  string,
  {
    file: TFile
    treeExpanded?: boolean
  }
>

interface FileTreeProps extends React.HTMLAttributes<HTMLDivElement> {
  onSelectTreeNode?: (treeNode: TFileTreeNode) => void
  repositoryName: string
  activePath?: string
}

type FileTreeProviderProps = React.PropsWithChildren<{
  onSelectTreeNode?: (treeNode: TFileTreeNode) => void
  repositoryName: string
  activePath?: string
  initialFileMap?: TFileMap
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
  activePath
}) => {
  const [fileMap, setFileMap] = React.useState<TFileMap>(initialFileMap ?? {})
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

type DirectoryTreeNodeProps = {
  node: TFileTreeNode
  level: number
  root?: boolean
}
type FileTreeNodeProps = {
  node: TFileTreeNode
}

interface FileTreeNodeViewProps extends React.HTMLAttributes<HTMLDivElement> {
  isActive?: boolean
}

const FileTreeNodeView: React.FC<
  React.PropsWithChildren<FileTreeNodeViewProps>
> = ({ isActive, children, style: styleFromProps, className, ...props }) => {
  return (
    <div
      className={cn(
        'focus:bg-accent focus:text-accent-foreground hover:bg-accent flex cursor-pointer flex-nowrap items-center gap-1 overflow-x-hidden whitespace-nowrap rounded-sm',
        isActive && 'bg-accent',
        className
      )}
      style={{
        ...styleFromProps,
        paddingLeft: 'calc(0.5rem * var(--level) + 1.5rem)'
      }}
      {...props}
    >
      {children}
    </div>
  )
}

const DirectoryTreeNodeView: React.FC<
  React.PropsWithChildren<FileTreeNodeViewProps>
> = ({ children, style: styleFromProps, className, ...props }) => {
  return (
    <div
      className={cn(
        'focus:bg-accent focus:text-accent-foreground hover:bg-accent flex cursor-pointer flex-nowrap items-center gap-1 truncate whitespace-nowrap rounded-sm',
        className
      )}
      style={{
        ...styleFromProps,
        paddingLeft: 'calc(0.25rem * var(--level) + 0.5rem)'
      }}
      {...props}
    >
      {children}
    </div>
  )
}

const FileTreeNode: React.FC<FileTreeNodeProps> = ({ node }) => {
  const { onSelectTreeNode, activePath } = React.useContext(FileTreeContext)
  const isFile = node.file.kind === 'file'
  const isActive = node.file.basename === activePath

  const handleSelect: React.MouseEventHandler<HTMLDivElement> = e => {
    if (isFile) {
      onSelectTreeNode?.(node)
    }
  }

  return (
    <FileTreeNodeView onClick={handleSelect} isActive={isActive}>
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

  const { data, isValidating }: SWRResponse<{ entries: TFile[] }> =
    useSWRImmutable(
      useAuthenticatedApi(
        shouldFetchChildren
          ? `/repositories/${repositoryName}/resolve/${basename}`
          : null
      ),
      fetcher
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

  const loading = useDebounce(isValidating, 150)

  const existingChildren = !!node?.children?.length
  const style = { '--level': level } as React.CSSProperties

  return (
    <div style={style}>
      <DirectoryTreeNodeView onClick={e => toggleExpandedKey(basename)}>
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
      <div>
        {expanded && existingChildren ? (
          <div>
            {node.children?.map(child => {
              const key = child.file.basename
              return child.file.kind === 'dir' ? (
                <DirectoryTreeNode key={key} node={child} level={level + 1} />
              ) : (
                <FileTreeNode key={key} node={child} />
              )
            })}
          </div>
        ) : null}
      </div>
    </div>
  )
}

const FileTreeRoot: React.FC = () => {
  const { fileTree } = React.useContext(FileTreeContext)

  return (
    <div style={{ '--level': 0 } as React.CSSProperties}>
      <DirectoryTreeNode root level={0} node={fileTree} />
    </div>
  )
}

const FileTree: React.FC<FileTreeProps> = ({
  onSelectTreeNode,
  repositoryName,
  className,
  activePath,
  ...props
}) => {
  const initialFileMap: TFileMap = React.useMemo(() => {
    return {
      [repositoryName]: {
        file: {
          kind: 'dir',
          basename: repositoryName
        },
        treeExpanded: false
      }
    }
  }, [])

  return (
    <div className={cn(className)} {...props}>
      <FileTreeProvider
        onSelectTreeNode={onSelectTreeNode}
        repositoryName={repositoryName}
        activePath={activePath}
        initialFileMap={initialFileMap}
      >
        <FileTreeRoot />
      </FileTreeProvider>
    </div>
  )
}

const RepositoriesFileTree: React.FC<Omit<FileTreeProps, 'repositoryName'>> = ({
  onSelectTreeNode,
  activePath,
  className,
  ...props
}) => {
  const [initialized, setInitialized] = React.useState(false)
  const { data: repositories, error }: SWRResponse<{ entries: TFile[] }> =
    useSWRImmutable(useAuthenticatedApi(`/repositories/resolve/`), fetcher, {
      onSuccess: () => {
        setInitialized(true)
      },
      onError: () => {
        setInitialized(true)
      }
    })

  if (!initialized) return <div className="flex justify-center">loading...</div>

  if (error) return <div>error {error?.message}</div>

  if (!repositories?.entries?.length) {
    return <div>No Data</div>
  }

  return (
    <div className={cn(className)} {...props}>
      {repositories?.entries.map(entry => {
        return (
          <FileTree
            key={entry.basename}
            repositoryName={entry.basename}
            activePath={activePath}
            onSelectTreeNode={onSelectTreeNode}
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

export { RepositoriesFileTree, type TFile, type TFileTreeNode }
