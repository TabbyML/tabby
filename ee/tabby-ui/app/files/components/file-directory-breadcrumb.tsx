import React from 'react'
import Link from 'next/link'

import { RepositoryKind } from '@/lib/gql/generates/graphql'
import { cn } from '@/lib/utils'
import { CopyButton } from '@/components/copy-button'

import { SourceCodeBrowserContext } from './source-code-browser'
import { generateEntryPath, resolveFileNameFromPath } from './utils'

interface FileDirectoryBreadcrumbProps
  extends React.HTMLAttributes<HTMLDivElement> {}

const FileDirectoryBreadcrumb: React.FC<FileDirectoryBreadcrumbProps> = ({
  className
}) => {
  const { currentFileRoutes, activeRepo, activeEntryInfo } = React.useContext(
    SourceCodeBrowserContext
  )
  const basename = activeEntryInfo?.basename
  const routes: Array<{
    name: string
    href: string
    kind?: RepositoryKind
  }> = React.useMemo(() => {
    const basename = activeEntryInfo?.basename
    let result = [
      {
        name: activeEntryInfo?.repositoryName ?? '',
        href: generateEntryPath(activeRepo, activeEntryInfo.rev, '', 'dir')
      }
    ]

    if (basename) {
      const pathSegments = decodeURIComponent(basename).split('/') || []
      for (let i = 0; i < pathSegments.length; i++) {
        const p = pathSegments.slice(0, i + 1).join('/')
        const name = resolveFileNameFromPath(p)
        result.push({
          name: decodeURIComponent(name),
          href: generateEntryPath(activeRepo, activeEntryInfo.rev, p, 'dir')
        })
      }
    }

    return result
  }, [activeEntryInfo, activeRepo])

  return (
    <div className={cn('flex flex-nowrap items-center gap-1', className)}>
      <div className="flex items-center gap-1 overflow-x-auto leading-8">
        <Link
          className="cursor-pointer font-medium text-primary hover:underline"
          href="/files"
        >
          Repositories
        </Link>
        <div>/</div>
        {routes?.map((route, idx) => {
          const isRepo = idx === 0 && routes?.length > 1
          const isActiveFile = idx === routes.length - 1
          const classname = cn(
            'whitespace-nowrap',
            isRepo || isActiveFile ? 'font-bold' : 'font-medium',
            isActiveFile ? '' : 'cursor-pointer text-primary hover:underline',
            isRepo ? 'hover:underline' : undefined
          )

          return (
            <React.Fragment key={route.href}>
              {isActiveFile ? (
                <div className={classname}>{route.name}</div>
              ) : (
                <Link className={classname} href={`/files/${route.href}`}>
                  {route.name}
                </Link>
              )}
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
