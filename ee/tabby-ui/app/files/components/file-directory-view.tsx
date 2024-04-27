import React from 'react'
import { find, omit } from 'lodash-es'

import { useDebounceValue } from '@/lib/hooks/use-debounce'
import { cn } from '@/lib/utils'
import { IconDirectorySolid, IconFile } from '@/components/ui/icons'
import { Skeleton } from '@/components/ui/skeleton'
import { Table, TableBody, TableCell, TableRow } from '@/components/ui/table'

import { BlobHeader } from './blob-header'
import { TFileTreeNode } from './file-tree'
import { RepositoryKindIcon } from './repository-kind-icon'
import { SourceCodeBrowserContext, TFileMapItem } from './source-code-browser'

interface DirectoryViewProps extends React.HTMLAttributes<HTMLDivElement> {
  loading: boolean
  initialized: boolean
}

const DirectoryView: React.FC<DirectoryViewProps> = ({
  className,
  loading: propsLoading,
  initialized
}) => {
  const { activePath, currentFileRoutes, setActivePath, fileTreeData } =
    React.useContext(SourceCodeBrowserContext)

  const files = React.useMemo(() => {
    return getCurrentDirFromTree(fileTreeData, activePath)
  }, [fileTreeData, activePath])

  const [loading] = useDebounceValue(propsLoading, 300)

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
    <div className={cn('text-base', className)}>
      <BlobHeader blob={undefined} hideBlobActions className="border-0" />
      {loading || !initialized ? (
        <FileTreeSkeleton />
      ) : fileTreeData?.length ? (
        <Table>
          <TableBody>
            {showParentEntry && (
              <TableRow
                className="cursor-pointer"
                onClick={e => onClickParent()}
              >
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
              {files.map(file => {
                const isRepository = file.isRepository
                const repoKind = file.repository?.kind
                return (
                  <TableRow key={file.fullPath}>
                    <TableCell className="p-1 px-4 text-base">
                      <div className="flex items-center gap-2">
                        <div className="shrink-0">
                          {isRepository ? (
                            <RepositoryKindIcon
                              kind={repoKind}
                              fallback={
                                <IconDirectorySolid
                                  style={{ color: 'rgb(84, 174, 255)' }}
                                />
                              }
                            />
                          ) : file.file.kind === 'dir' ? (
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
                          {file.name}
                        </span>
                      </div>
                    </TableCell>
                  </TableRow>
                )
              })}
            </>
          </TableBody>
        </Table>
      ) : (
        <div className="flex justify-center py-8">
          No indexed repository yet
        </div>
      )}
    </div>
  )
}

function FileTreeSkeleton() {
  return (
    <ul className="space-y-3 p-2">
      <Skeleton />
      <Skeleton />
      <Skeleton />
      <Skeleton />
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
    for (let i = 1; i < pathSegments.length; i++) {
      const path = pathSegments.slice(0, i + 1).join('/')
      let node = find<TFileTreeNode>(currentNodes, t => t.fullPath === path)
      if (node?.children) {
        currentNodes = node?.children
      } else {
        return []
      }
    }
    return currentNodes?.map(child => omit(child, 'children')) || []
  }
}

export { DirectoryView }
