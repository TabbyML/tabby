import React from 'react'

import { cn } from '@/lib/utils'
import { CopyButton } from '@/components/copy-button'

import { SourceCodeBrowserContext } from './source-code-browser'
import { resolveBasenameFromPath } from './utils'

interface FileDirectoryBreadcrumbProps
  extends React.HTMLAttributes<HTMLDivElement> {}

const FileDirectoryBreadcrumb: React.FC<FileDirectoryBreadcrumbProps> = ({
  className
}) => {
  const { currentFileRoutes, setActivePath, activePath } = React.useContext(
    SourceCodeBrowserContext
  )
  const basename = React.useMemo(
    () => resolveBasenameFromPath(activePath),
    [activePath]
  )

  return (
    <div className={cn('flex flex-nowrap items-center gap-1', className)}>
      <div className="flex items-center gap-1 overflow-x-auto leading-8">
        <div
          className="cursor-pointer font-medium text-primary hover:underline"
          onClick={e => setActivePath(undefined)}
        >
          Repositories
        </div>
        <div>/</div>
        {currentFileRoutes?.map((route, idx) => {
          const isRepo = idx === 0 && currentFileRoutes?.length > 1
          const isActiveFile = idx === currentFileRoutes.length - 1

          return (
            <React.Fragment key={route.fullPath}>
              <div
                className={cn(
                  'whitespace-nowrap',
                  isRepo || isActiveFile ? 'font-bold' : 'font-medium',
                  isActiveFile
                    ? ''
                    : 'cursor-pointer text-primary hover:underline',
                  isRepo ? 'hover:underline' : undefined
                )}
                onClick={e => setActivePath(route.fullPath)}
              >
                {route.name}
              </div>
              {route.file.kind !== 'file' && <div>/</div>}
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
