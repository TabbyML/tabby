import React from 'react'

import { RepositoryKind } from '@/lib/gql/generates/graphql'
import { cn } from '@/lib/utils'
import { CopyButton } from '@/components/copy-button'

import { SourceCodeBrowserContext } from './source-code-browser'
import {
  generateEntryPath,
  resolveFileNameFromPath,
  resolveRepositoryInfoFromPath
} from './utils'

interface FileDirectoryBreadcrumbProps
  extends React.HTMLAttributes<HTMLDivElement> {}

const FileDirectoryBreadcrumb: React.FC<FileDirectoryBreadcrumbProps> = ({
  className
}) => {
  const {
    currentFileRoutes,
    updateActivePath,
    activePath,
    activeRepo,
    activeRepoRef,
    activeEntryInfo
  } = React.useContext(SourceCodeBrowserContext)
  const basename = React.useMemo(
    () => resolveRepositoryInfoFromPath(activePath)?.basename,
    [activePath]
  )

  const routes: Array<{
    name: string
    href: string
    kind?: RepositoryKind
  }> = React.useMemo(() => {
    const basename = activeEntryInfo?.basename
    let result = [
      {
        name: activeRepo?.name ?? '',
        href: generateEntryPath(activeRepo, activeRepoRef?.name, '', 'dir')
      }
    ]

    if (basename) {
      const pathSegments = basename?.split('/') || []
      for (let i = 0; i < pathSegments.length; i++) {
        const p = pathSegments.slice(0, i + 1).join('/')
        const name = resolveFileNameFromPath(p)
        result.push({
          name,
          href: generateEntryPath(activeRepo, activeRepoRef?.name, p, 'dir')
        })
      }
    }

    return result
  }, [activeEntryInfo, activeRepo, activeRepoRef])

  return (
    <div className={cn('flex flex-nowrap items-center gap-1', className)}>
      <div className="flex items-center gap-1 overflow-x-auto leading-8">
        <div
          className="cursor-pointer font-medium text-primary hover:underline"
          onClick={e => updateActivePath(undefined)}
        >
          Repositories
        </div>
        <div>/</div>
        {routes?.map((route, idx) => {
          const isRepo = idx === 0 && routes?.length > 1
          const isActiveFile = idx === routes.length - 1

          // todo use link
          return (
            <React.Fragment key={route.href}>
              <div
                className={cn(
                  'whitespace-nowrap',
                  isRepo || isActiveFile ? 'font-bold' : 'font-medium',
                  isActiveFile
                    ? ''
                    : 'cursor-pointer text-primary hover:underline',
                  isRepo ? 'hover:underline' : undefined
                )}
                onClick={e => {
                  if (isActiveFile) return
                  updateActivePath(route.href)
                }}
              >
                {route.name}
              </div>
              {!isActiveFile && <div>/</div>}
            </React.Fragment>
          )
        })}
      </div>
      {!!currentFileRoutes?.length && !!basename && (
        <CopyButton className="shrink-0" value={basename} />
      )}
    </div>
  )
}

export { FileDirectoryBreadcrumb }
