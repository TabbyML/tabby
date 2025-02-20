import { useRef, useState } from 'react'

import { ContextInfo } from '@/lib/gql/generates/graphql'
import { cn } from '@/lib/utils'
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator
} from '@/components/ui/command'
import { IconFolderGit } from '@/components/ui/icons'
import {
  Popover,
  PopoverContent,
  PopoverTrigger
} from '@/components/ui/popover'
import { SourceIcon } from '@/components/source-icon'

import LoadingWrapper from '../loading-wrapper'
import { Button } from '../ui/button'
import { IconCheck } from '../ui/icons'
import { Skeleton } from '../ui/skeleton'

interface RepoSelectProps {
  repos: ContextInfo['sources'] | undefined
  value: string | undefined
  onChange: (v: string | undefined) => void
  isInitializing?: boolean
}

export function RepoSelect({
  repos,
  value,
  onChange,
  isInitializing
}: RepoSelectProps) {
  const [open, setOpen] = useState(false)
  const commandListRef = useRef<HTMLDivElement>(null)

  const onSelectRepo = (v: string) => {
    onChange(v)
  }

  const scrollCommandListToTop = () => {
    requestAnimationFrame(() => {
      if (commandListRef.current) {
        commandListRef.current.scrollTop = 0
      }
    })
  }

  const onSearchChange = () => {
    scrollCommandListToTop()
  }

  const selectedRepo = value
    ? repos?.find(repo => repo.sourceId === value)
    : undefined
  const selectedRepoName = selectedRepo?.sourceName

  // if there's no repo, hide the repo select
  if (!isInitializing && !repos?.length) return null

  return (
    <LoadingWrapper
      loading={isInitializing}
      fallback={
        <div className="w-full pl-2">
          <Skeleton className="h-3 w-[10rem]" />
        </div>
      }
    >
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger className="outline-none" asChild>
          <Button
            variant="ghost"
            className="gap-2 px-1.5 py-1 font-normal text-foreground/90"
            title={selectedRepoName || 'select codebase'}
          >
            {selectedRepo ? (
              <SourceIcon
                kind={selectedRepo.sourceKind}
                className="h-3.5 w-3.5 shrink-0"
              />
            ) : (
              <IconFolderGit className="shrink-0" />
            )}
            <div className="flex flex-1 items-center gap-1.5 truncate break-all">
              <span
                className={cn('truncate', {
                  'text-muted-foreground': !selectedRepoName
                })}
              >
                {selectedRepoName || 'Codebase'}
              </span>
            </div>
          </Button>
        </PopoverTrigger>
        <PopoverContent
          side="bottom"
          align="start"
          className="max-h-[50vh] min-w-[20vw] max-w-[80vw] overflow-x-hidden rounded-md border bg-popover p-2 pb-0 text-popover-foreground shadow animate-in"
        >
          <Command>
            <CommandInput
              placeholder="Select repository..."
              onValueChange={onSearchChange}
            />
            <CommandList className="max-h-[30vh]" ref={commandListRef}>
              <CommandEmpty>No repository found</CommandEmpty>
              <CommandGroup>
                {repos?.map(repo => {
                  const isSelected = repo.sourceId === value

                  return (
                    <CommandItem
                      key={repo.sourceId}
                      onSelect={() => {
                        onSelectRepo(repo.sourceId)
                        setOpen(false)
                      }}
                      title={repo.sourceName}
                    >
                      <IconCheck
                        className={cn(
                          'mr-1 shrink-0',
                          repo.sourceId === value ? 'opacity-100' : 'opacity-0'
                        )}
                      />
                      <div className="flex flex-1 items-center gap-1 overflow-x-hidden">
                        <SourceIcon
                          kind={repo.sourceKind}
                          className="shrink-0"
                        />
                        <div
                          className={cn('truncate', {
                            'font-semibold': isSelected
                          })}
                        >
                          {repo.sourceName}
                        </div>
                      </div>
                    </CommandItem>
                  )
                })}
              </CommandGroup>
            </CommandList>
            <CommandSeparator />
            <CommandGroup>
              <CommandItem
                disabled={!value}
                className="flex justify-center"
                onSelect={() => {
                  onChange(undefined)
                  setOpen(false)
                }}
              >
                Clear
              </CommandItem>
            </CommandGroup>
          </Command>
        </PopoverContent>
      </Popover>
    </LoadingWrapper>
  )
}
