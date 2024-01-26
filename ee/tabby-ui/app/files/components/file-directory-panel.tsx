import React from 'react'
import { find, omit } from 'lodash-es'

import { useDebounce } from '@/lib/hooks/use-debounce'
import { cn } from '@/lib/utils'
import { IconDirectorySolid, IconFile } from '@/components/ui/icons'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/table'

import { TFileTreeNode } from './file-tree'
import { SourceCodeBrowserContext, TFileMapItem } from './source-code-browser'
import { resolveFileNameFromPath } from './utils'

interface DirectoryPanelProps extends React.HTMLAttributes<HTMLDivElement> {
  loading: boolean
}

const DirectoryPanel: React.FC<DirectoryPanelProps> = ({
  className,
  loading: propsLoading,
  ...props
}) => {
  const {
    fileMap,
    activePath,
    currentFileRoutes,
    setActivePath,
    fileTreeData
  } = React.useContext(SourceCodeBrowserContext)

  const loading = useDebounce(propsLoading, 300)

  const files = React.useMemo(() => {
    return getCurrentDirFromTree(fileTreeData, activePath)
  }, [fileTreeData, activePath])

  const showParentEntry = currentFileRoutes?.length > 0

  const onClickParent = () => {
    const parentPath =
      currentFileRoutes[currentFileRoutes?.length - 2]?.fullPath
    setActivePath(parentPath)
  }

  const onClickFile = (file: TFileMapItem) => {
    setActivePath(file.fullPath)
  }

  return (
    <div className={cn(className)}>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Name</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {showParentEntry && (
            <TableRow className="cursor-pointer" onClick={e => onClickParent()}>
              <TableCell className="p-1 px-4">
                <div className="flex items-center gap-2">
                  <div className="shrink-0">
                    <IconDirectorySolid
                      style={{ color: 'rgb(84, 174, 255)' }}
                    />
                  </div>
                  <span className="px-1 py-2">..</span>
                </div>
              </TableCell>
            </TableRow>
          )}
          <>
            {loading ? (
              <FileTreeSkeleton />
            ) : (
              <>
                {files.map(file => {
                  return (
                    <TableRow key={file.fullPath}>
                      <TableCell className="p-1 px-4 text-base">
                        <div className="flex items-center gap-2">
                          <div className="shrink-0">
                            {file.file.kind === 'dir' ? (
                              <IconDirectorySolid
                                style={{ color: 'rgb(84, 174, 255)' }}
                              />
                            ) : (
                              <IconFile />
                            )}
                          </div>
                          <span
                            onClick={e => onClickFile(file)}
                            className="cursor-pointer px-1 py-2 hover:text-primary hover:underline"
                          >
                            {resolveFileNameFromPath(file.fullPath)}
                          </span>
                        </div>
                      </TableCell>
                    </TableRow>
                  )
                })}
              </>
            )}
          </>
        </TableBody>
      </Table>
    </div>
  )
}

function FileTreeSkeleton() {
  return (
    <ul className="duration-600 animate-pulse space-y-3 p-2">
      <li className="h-4 rounded-md bg-gray-200 dark:bg-gray-700"></li>
      <li className="h-4 rounded-md bg-gray-200 dark:bg-gray-700"></li>
      <li className="h-4 rounded-md bg-gray-200 dark:bg-gray-700"></li>
    </ul>
  )
}

function getCurrentDirFromTree(
  treeData: TFileTreeNode[],
  path: string | undefined
): TFileTreeNode[] {
  // const regx = new RegExp(`${path}\/[\\w\.\-]+$`)
  if (!treeData?.length) return []
  if (!path) {
    const repos = treeData.map(x => omit(x, 'children')) || []
    return repos
  } else {
    let pathSegments = path.split('/')
    let currentNodes: TFileTreeNode[] = treeData
    while (pathSegments.length) {
      let name = pathSegments.shift()
      let node = find<TFileTreeNode>(currentNodes, t => t.name === name)
      if (node?.children) {
        currentNodes = node?.children
      } else {
        return []
      }
    }
    return currentNodes?.map(child => omit(child, 'children')) || []
  }
}

export { DirectoryPanel }
