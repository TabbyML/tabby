import { useWindowSize } from '@uidotdev/usehooks'

import { ContextInfo } from '@/lib/gql/generates/graphql'
import { cn } from '@/lib/utils'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import {
  IconCheck,
  IconChevronUpDown,
  IconFolderGit,
  IconRemove
} from '@/components/ui/icons'
import { Skeleton } from '@/components/ui/skeleton'
import LoadingWrapper from '@/components/loading-wrapper'

import { Badge } from '../ui/badge'
import { Button } from '../ui/button'

interface RepoSelectProps {
  repos: ContextInfo['sources']
  value: string | undefined
  onChange: (v: string | undefined) => void
  isInitializing?: boolean
  // sourceId
  workspaceRepoId?: string
}

export function RepoSelect({
  repos,
  value,
  onChange,
  isInitializing,
  workspaceRepoId
}: RepoSelectProps) {
  const { width } = useWindowSize()
  const isExtraSmallScreen = typeof width === 'number' && width < 240

  const onSelectRepo = (v: string) => {
    onChange(v)
  }

  const isWorkspaceRepo = !!workspaceRepoId && workspaceRepoId === value
  const selectedRepoName = repos?.find(
    repo => repo.sourceId === value
  )?.sourceName

  return (
    <LoadingWrapper
      loading={isInitializing}
      fallback={
        <div className="w-full pl-2">
          <Skeleton className="h-3 w-[20%]" />
        </div>
      }
    >
      <DropdownMenu>
        <Badge
          variant="outline"
          className="h-7 min-w-[8rem] items-center gap-1 overflow-hidden break-all rounded-md pr-0 text-sm font-semibold"
        >
          <DropdownMenuTrigger className="outline-none" asChild>
            <div className="flex flex-1 cursor-pointer items-center gap-1.5 overflow-hidden">
              <IconFolderGit className="shrink-0" />
              <div className="flex flex-1 items-center gap-1.5 truncate break-all">
                <span
                  className={cn('truncate', {
                    'text-muted-foreground': !selectedRepoName
                  })}
                >
                  {selectedRepoName || 'Select repo...'}
                </span>
                {isWorkspaceRepo && (
                  <span className="shrink-0 text-muted-foreground">
                    Context
                  </span>
                )}
              </div>
              {!value && <IconChevronUpDown className="shrink-0" />}
            </div>
          </DropdownMenuTrigger>
          {!!value && (
            <Button
              type="button"
              size="icon"
              variant="ghost"
              className="h-7 w-7 shrink-0 rounded-l-none"
              onClick={e => {
                e.stopPropagation()
                onChange(undefined)
              }}
            >
              <IconRemove />
            </Button>
          )}
        </Badge>
        <DropdownMenuContent
          side="top"
          align="start"
          className="dropdown-menu max-h-[30vh] min-w-[20rem] max-w-full overflow-y-auto overflow-x-hidden rounded-md border bg-popover p-2 text-popover-foreground shadow animate-in"
        >
          <DropdownMenuRadioGroup>
            {repos.map(repo => {
              const isSelected = repo.sourceId === value
              return (
                <DropdownMenuRadioItem
                  value={repo.sourceId}
                  key={repo.sourceId}
                  className="flex cursor-pointer items-center py-2 pl-3"
                  onSelect={() => onSelectRepo(repo.sourceId)}
                >
                  <div className="flex flex-1 items-center gap-2 truncate">
                    <IconCheck
                      className={cn(
                        'shrink-0',
                        repo.sourceId === value ? 'opacity-100' : 'opacity-0'
                      )}
                    />
                    <span
                      className={cn({
                        'font-medium': isSelected
                      })}
                    >
                      {repo.sourceName}
                    </span>
                  </div>
                  {repo.sourceId === workspaceRepoId && (
                    <span className="ml-1.5 shrink-0 text-muted-foreground">
                      {isExtraSmallScreen ? 'Workspace' : 'Repo in workspace'}
                    </span>
                  )}
                </DropdownMenuRadioItem>
              )
            })}
          </DropdownMenuRadioGroup>
        </DropdownMenuContent>
      </DropdownMenu>
    </LoadingWrapper>
  )
}
