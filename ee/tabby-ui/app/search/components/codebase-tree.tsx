'use client'

import React, { useMemo } from 'react'
import { motion } from 'framer-motion'

import { cn } from '@/lib/utils'
import {
  IconChevronRight,
  IconDirectoryExpandSolid,
  IconDirectorySolid,
  IconFile
} from '@/components/ui/icons'

type TFileTreeNode = {
  name: string
  fullPath: string
  children?: Array<TFileTreeNode>
}

interface FileTreeProps extends React.HTMLAttributes<HTMLDivElement> {
  onSelectTreeNode?: (treeNode: TFileTreeNode) => void
  collapsedKeys: Set<string>
  toggleCollapsedKey: (key: string) => void
  fileList: string[]
}

interface FileTreeProviderProps extends FileTreeProps {}

type FileTreeContextValue = {
  fileTreeData: TFileTreeNode[]
  onSelectTreeNode: FileTreeProps['onSelectTreeNode']
  collapsedKeys: Set<string>
  toggleCollapsedKey: (key: string) => void
}

type DirectoryTreeNodeProps = {
  node: TFileTreeNode
  level: number
}
type FileTreeNodeProps = {
  node: TFileTreeNode
  level: number
}

interface FileTreeNodeViewProps extends React.HTMLAttributes<HTMLDivElement> {
  level: number
}

interface DirectoryTreeNodeViewProps
  extends React.HTMLAttributes<HTMLDivElement> {
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
  collapsedKeys,
  toggleCollapsedKey,
  fileList
}) => {
  const fileTreeData = useMemo(() => {
    return sortFileTree(listToFileTree(fileList))
  }, [fileList])

  return (
    <FileTreeContext.Provider
      value={{
        onSelectTreeNode,
        fileTreeData,
        collapsedKeys,
        toggleCollapsedKey
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
            // className="flex h-8 w-2 border-r border-transparent transition-colors duration-300 group-hover/filetree:border-border"
            className="flex h-8 w-2 border-r border-border transition-colors duration-300"
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
> = ({ level, children, className, ...props }) => {
  return (
    <div
      className={cn(
        'relative flex h-8 items-stretch rounded-sm pl-1.5 hover:bg-accent focus:bg-accent focus:text-accent-foreground',
        className
      )}
      {...props}
    >
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
> = ({ children, level, className, ...props }) => {
  return (
    <div
      className={cn(
        'relative flex cursor-pointer items-stretch rounded-sm pl-1.5 hover:bg-accent focus:bg-accent focus:text-accent-foreground',
        className
      )}
      {...props}
    >
      <GridArea level={level} />
      <div className="flex flex-nowrap items-center gap-2 truncate whitespace-nowrap">
        {children}
      </div>
    </div>
  )
}

const FileTreeNode: React.FC<FileTreeNodeProps> = ({ node, level }) => {
  return (
    <FileTreeNodeView level={level}>
      <IconFile className="shrink-0" />
      <div className="truncate">{node?.name}</div>
    </FileTreeNodeView>
  )
}

const DirectoryNode: React.FC<DirectoryTreeNodeProps> = ({ node, level }) => {
  const { collapsedKeys, toggleCollapsedKey, onSelectTreeNode } =
    React.useContext(FileTreeContext)

  const existingChildren = !!node?.children?.length
  const collapsed = collapsedKeys.has(node.fullPath)

  const onSelectDirectory: React.MouseEventHandler<HTMLDivElement> = e => {
    onSelectTreeNode?.(node)
  }

  return (
    <>
      <DirectoryTreeNodeView level={level} onClick={onSelectDirectory}>
        <div
          className="flex h-8 shrink-0 items-center"
          // onClick={e => {
          //   toggleCollapsedKey(node.fullPath)
          //   e.stopPropagation()
          // }}
        >
          <IconChevronRight
            className={cn('transition-transform ease-out', {
              'rotate-90': !collapsed
            })}
          />
        </div>
        <div className="shrink-0" style={{ color: 'rgb(84, 174, 255)' }}>
          {collapsed ? <IconDirectorySolid /> : <IconDirectoryExpandSolid />}
        </div>
        <div className="truncate">{node?.name}</div>
      </DirectoryTreeNodeView>
      <>
        {!collapsed && existingChildren ? (
          <motion.div
            initial={{ height: 0 }}
            animate={{ height: 'auto' }}
            transition={{
              duration: 0.1,
              ease: 'easeOut'
            }}
            className="overflow-hidden"
          >
            {node.children?.map(child => {
              const key = child.fullPath
              return child?.children?.length ? (
                <DirectoryNode key={key} node={child} level={level + 1} />
              ) : (
                <FileTreeNode key={key} node={child} level={level + 1} />
              )
            })}
          </motion.div>
        ) : null}
      </>
    </>
  )
}

const FileTreeRenderer: React.FC = () => {
  const { fileTreeData } = React.useContext(FileTreeContext)

  return (
    <>
      {fileTreeData?.map(node => {
        const isFile = !node.children?.length
        return isFile ? (
          <FileTreeNode level={0} node={node} key={node.fullPath} />
        ) : (
          <DirectoryNode level={0} node={node} key={node.fullPath} />
        )
      })}
    </>
  )
}

const CodebaseFileTree: React.FC<FileTreeProps> = ({ className, ...props }) => {
  return (
    <div className={cn('group/filetree select-none', className)}>
      <FileTreeProvider {...props}>
        <FileTreeRenderer />
      </FileTreeProvider>
    </div>
  )
}

function listToFileTree(fileList: string[]): TFileTreeNode[] {
  const tree: TFileTreeNode[] = []
  if (!fileList?.length) return tree

  for (const filePath of fileList) {
    const pathSegments = filePath.split('/')
    let currentNode = tree
    for (let i = 0; i < pathSegments.length; i++) {
      const p = pathSegments.slice(0, i + 1).join('/')
      const existingNode = currentNode?.find(node => node.fullPath === p)

      if (existingNode) {
        currentNode = existingNode.children || []
      } else {
        const newNode: TFileTreeNode = {
          name: resolveFileNameFromPath(filePath),
          fullPath: filePath,
          children: []
        }
        currentNode.push(newNode)
        currentNode = newNode.children as TFileTreeNode[]
      }
    }
  }

  return tree
}

function sortFileTree(tree: TFileTreeNode[]) {
  if (!tree.length) return []

  tree.sort((a, b) => {
    const aIsFile = isFile(a) ? 1 : 0
    const bIsFile = isFile(b) ? 1 : 0
    return aIsFile - bIsFile || a.name.localeCompare(b.name)
  })
  for (let item of tree) {
    if (item?.children) {
      sortFileTree(item.children)
    }
  }

  return tree
}

function isFile(node: TFileTreeNode) {
  return !node.children?.length
}

function resolveFileNameFromPath(path: string) {
  if (!path) return ''
  const pathSegments = path.split('/')
  return pathSegments[pathSegments.length - 1]
}

export { CodebaseFileTree }
