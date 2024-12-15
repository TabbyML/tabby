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
  value: string | null | undefined
  onChange: (v: string | null) => void
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
  const onSelectRepo = (v: string) => {
    onChange(v)
    // todo input focus
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
          className="h-7 items-center gap-1 overflow-hidden rounded-md text-sm font-semibold pr-0 min-w-[8rem]"
        >
          <DropdownMenuTrigger className="outline-none" asChild>
            <div className="cursor-pointer flex-1 flex items-center gap-1.5 truncate break-all">
              <IconFolderGit className="shrink-0" />
              <div className="flex-1 flex items-center gap-1.5">
                <span
                  className={cn({
                    'text-muted-foreground': !selectedRepoName
                  })}
                >
                  {selectedRepoName || 'Select repo...'}
                </span>
                {isWorkspaceRepo && (
                  <span className="shrink-0 text-muted-foreground">
                    Repo in workspace
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
                console.log('eeeeeeee')
                e.preventDefault()
                e.stopPropagation()
                onChange(null)
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
                  className="cursor-pointer py-2 pl-3 flex items-center"
                  onSelect={() => onSelectRepo(repo.sourceId)}
                >
                  <div className="flex-1 truncate flex items-center gap-2">
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
                    <span className="text-muted-foreground ml-1.5 shrink-0">
                      Repo in workspace
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
