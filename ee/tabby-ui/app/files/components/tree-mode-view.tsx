import React from 'react'
import Link from 'next/link'
import { find, isEmpty, omit } from 'lodash-es'

import { useDebounceValue } from '@/lib/hooks/use-debounce'
import { cn } from '@/lib/utils'
import { buttonVariants } from '@/components/ui/button'
import {
  IconArrowRight,
  IconDirectorySolid,
  IconFile,
  IconFileSearch
} from '@/components/ui/icons'
import { Skeleton } from '@/components/ui/skeleton'
import { Table, TableBody, TableCell, TableRow } from '@/components/ui/table'

import { TFileTreeNode } from './file-tree'
import { RepositoryKindIcon } from './repository-kind-icon'
import { SourceCodeBrowserContext } from './source-code-browser'
import {
  generateEntryPath,
  getDefaultRepoRef,
  repositoryMap2List,
  resolveRepoRef,
  resolveRepositoryInfoFromPath
} from './utils'

interface TreeModeViewProps extends React.HTMLAttributes<HTMLDivElement> {
  loading: boolean
  initialized: boolean
}

const TreeModeView: React.FC<TreeModeViewProps> = ({
  className,
  loading: propsLoading,
  initialized
}) => {
  const {
    activePath,
    currentFileRoutes,
    fileTreeData,
    activeRepo,
    repoMap,
    activeEntryInfo
  } = React.useContext(SourceCodeBrowserContext)

  const files: TFileTreeNode[] = React.useMemo(() => {
    if (!isEmpty(repoMap) && !activeRepo) {
      return repositoryMap2List(repoMap).map(repo => {
        return {
          file: {
            basename: repo.name,
            kind: 'dir'
          },
          isRepository: true,
          repository: repo,
          fullPath: generateEntryPath(
            repo,
            resolveRepoRef(getDefaultRepoRef(repo.refs))?.name,
            '',
            'dir'
          ),
          name: repo.name
        }
      })
    }

    return getCurrentDirFromTree(fileTreeData, activePath)
  }, [fileTreeData, activePath, activeRepo, repoMap])

  const [loading] = useDebounceValue(propsLoading, 300)

  const showParentEntry = !!activeEntryInfo?.basename
  const parentNode = currentFileRoutes[currentFileRoutes?.length - 2]

  return (
    <div className={cn('text-base', className)}>
      {(loading && !files?.length) || !initialized ? (
        <FileTreeSkeleton />
      ) : files?.length ? (
        <Table>
          <TableBody>
            {showParentEntry && (
              <TableRow className="cursor-pointer">
                <TableCell className="p-1 px-4">
                  <Link
                    href={`/files/${generateEntryPath(
                      activeRepo,
                      activeEntryInfo.rev,
                      parentNode?.file?.basename,
                      parentNode?.file?.kind
                    )}`}
                  >
                    <div className="flex items-center gap-2">
                      <div className="shrink-0">
                        <IconDirectorySolid
                          style={{ color: 'rgb(84, 174, 255)' }}
                        />
                      </div>
                      <span className="px-1 py-2">..</span>
                    </div>
                  </Link>
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
                        <Link
                          href={
                            isRepository
                              ? `/files/${file.fullPath}`
                              : `/files/${generateEntryPath(
                                  activeRepo ?? file.repository,
                                  activeEntryInfo.rev,
                                  file.file.basename,
                                  file.file.kind
                                )}`
                          }
                          className="cursor-pointer px-1 py-2 hover:text-primary hover:underline"
                        >
                          {file.name}
                        </Link>
                      </div>
                    </TableCell>
                  </TableRow>
                )
              })}
            </>
          </TableBody>
        </Table>
      ) : isEmpty(repoMap) ? (
        <div className="flex min-h-[30vh] items-center justify-center">
          <div className="flex flex-col items-center gap-4">
            <div className="flex items-center gap-2">
              <IconFileSearch className="h-6 w-6" />
              <div className="text-xl font-semibold">No repositories</div>
            </div>
            <Link
              href="/settings/providers/git"
              className={cn(buttonVariants(), 'gap-2')}
            >
              <span>Connect</span>
              <IconArrowRight />
            </Link>
          </div>
        </div>
      ) : null}
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
  if (!treeData?.length) return []
  if (!path) {
    const repos = treeData.map(x => omit(x, 'children')) || []
    return repos
  } else {
    let { basename = '' } = resolveRepositoryInfoFromPath(path)

    if (!basename) return treeData
    const pathSegments = decodeURIComponent(basename).split('/')
    let currentNodes: TFileTreeNode[] = treeData
    for (let i = 0; i < pathSegments.length; i++) {
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

export { TreeModeView }
