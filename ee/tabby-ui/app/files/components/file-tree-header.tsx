'use client'

import React, { useContext } from 'react'
import { isNil } from 'lodash-es'

import { GitReference } from '@/lib/gql/generates/graphql'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList
} from '@/components/ui/command'
import {
  IconCheck,
  IconFolderGit,
  IconGitFork,
  IconTag
} from '@/components/ui/icons'
import {
  Popover,
  PopoverContent,
  PopoverTrigger
} from '@/components/ui/popover'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'

import { RepositoryKindIcon } from './repository-kind-icon'
import { SourceCodeBrowserContext } from './source-code-browser'
import { RepositoryRefKind } from './types'
import {
  generateEntryPath,
  getDefaultRepoRef,
  repositoryMap2List,
  resolveRepoRef,
  resolveRepoSpecifierFromRepoInfo
} from './utils'

interface FileTreeHeaderProps extends React.HTMLAttributes<HTMLDivElement> {}

const FileTreeHeader: React.FC<FileTreeHeaderProps> = ({
  className,
  ...props
}) => {
  const {
    updateActivePath,
    initialized,
    activeRepo,
    activeRepoRef,
    fileMap,
    repoMap,
    activeEntryInfo
  } = useContext(SourceCodeBrowserContext)
  const repoList = React.useMemo(() => {
    return repositoryMap2List(repoMap).map(repo => {
      const repoSpecifier = resolveRepoSpecifierFromRepoInfo(repo) as string
      return {
        repo,
        repoSpecifier
      }
    })
  }, [repoMap])
  const [refSelectVisible, setRefSelectVisible] = React.useState(false)
  const [activeRefKind, setActiveRefKind] = React.useState<RepositoryRefKind>(
    activeRepoRef?.kind ?? 'branch'
  )
  const { repositoryKind, repositoryName, repositorySpecifier } =
    activeEntryInfo
  const refs = activeRepo?.refs
  const formattedRefs = React.useMemo(() => {
    if (!refs?.length) return []
    return refs.map(ref => resolveRepoRef(ref))
  }, [refs])

  const branches = formattedRefs.filter(o => o.kind === 'branch')
  const tags = formattedRefs.filter(o => o.kind === 'tag')
  const commandOptions = activeRefKind === 'tag' ? tags : branches
  const noIndexedRepo = initialized && !repoList?.length

  const onSelectRef = (ref: GitReference | undefined) => {
    if (isNil(ref)) return
    const nextRev = resolveRepoRef(ref)?.name ?? ''
    const { basename = '' } = activeEntryInfo
    const kind = fileMap?.[basename]?.file?.kind ?? 'dir'

    updateActivePath(generateEntryPath(activeRepo, nextRev, basename, kind))
  }

  const onSelectRepo = (repoSpecifier: string) => {
    const repo = repoList.find(o => o.repoSpecifier === repoSpecifier)?.repo
    if (repo) {
      const path = `${repoSpecifier}/-/tree/${
        resolveRepoRef(getDefaultRepoRef(repo.refs)).name
      }`
      updateActivePath(path)
    }
  }

  const onClickFilesTitle = () => {
    if (!activeRepo) return
    updateActivePath(
      generateEntryPath(activeRepo, activeEntryInfo?.rev, '', 'dir')
    )
  }

  return (
    <div className={cn(className)} {...props}>
      <div className="py-4 font-bold leading-8">
        <span
          className={cn('py-1', {
            'hover:underline cursor-pointer': !!activeRepo
          })}
          onClick={onClickFilesTitle}
        >
          Files
        </span>
      </div>
      <div className="space-y-3">
        {/* Repository select */}
        <Select
          disabled={!initialized}
          onValueChange={onSelectRepo}
          value={repositorySpecifier}
        >
          <SelectTrigger>
            <SelectValue asChild>
              <div className="flex items-center gap-2 overflow-hidden">
                <div className="shrink-0">
                  <RepositoryKindIcon
                    kind={repositoryKind}
                    fallback={<IconFolderGit />}
                  />
                </div>
                <span
                  className={cn(
                    'truncate',
                    !repositoryName && 'text-muted-foreground'
                  )}
                >
                  {repositoryName || 'Pick a repository'}
                </span>
              </div>
            </SelectValue>
          </SelectTrigger>
          <SelectContent className="max-h-[50vh] overflow-y-auto">
            {noIndexedRepo ? (
              <SelectItem isPlaceHolder value="" disabled>
                No repositories
              </SelectItem>
            ) : (
              <>
                {repoList?.map(repo => {
                  return (
                    <SelectItem
                      key={repo.repoSpecifier}
                      value={repo.repoSpecifier}
                    >
                      <div className="flex items-center gap-1">
                        <RepositoryKindIcon
                          kind={repo.repo.kind}
                          fallback={<IconFolderGit />}
                        />
                        {repo.repo.name}
                      </div>
                    </SelectItem>
                  )
                })}
              </>
            )}
          </SelectContent>
        </Select>
        {!!activeRepo && (
          <>
            {/* branch select */}
            <Popover open={refSelectVisible} onOpenChange={setRefSelectVisible}>
              <PopoverTrigger asChild>
                <Button
                  className="w-full justify-start gap-2 px-3"
                  variant="outline"
                >
                  {!!activeRepoRef && (
                    <>
                      {activeRepoRef.kind === 'tag' ? (
                        <IconTag className="shrink-0" />
                      ) : (
                        <IconGitFork className="shrink-0" />
                      )}
                      <span className="truncate" title={activeRepoRef.name}>
                        {activeRepoRef.kind === 'commit'
                          ? activeRepoRef.ref?.commit?.substring(0, 7)
                          : activeRepoRef.name}
                      </span>
                    </>
                  )}
                </Button>
              </PopoverTrigger>
              <PopoverContent
                className="w-[var(--radix-popover-trigger-width)] p-0"
                align="start"
                side="bottom"
                sideOffset={-8}
              >
                <Command>
                  <CommandInput
                    placeholder={
                      activeRefKind === 'tag' ? 'Find a tag' : 'Find a branch'
                    }
                  />
                  <Tabs
                    className="my-1 border-b"
                    value={activeRefKind}
                    onValueChange={v =>
                      setActiveRefKind(v as RepositoryRefKind)
                    }
                  >
                    <TabsList className="bg-popover py-0">
                      <TabsTrigger value="branch">Branches</TabsTrigger>
                      <TabsTrigger value="tag">Tags</TabsTrigger>
                    </TabsList>
                  </Tabs>
                  <CommandList className="max-h-[30vh]">
                    <CommandEmpty>Nothing to show</CommandEmpty>
                    <CommandGroup>
                      {commandOptions.map((ref, refIndex) => (
                        <CommandItem
                          key={ref?.ref?.name ?? refIndex}
                          onSelect={() => {
                            setRefSelectVisible(false)
                            onSelectRef(ref.ref)
                          }}
                        >
                          <IconCheck
                            className={cn(
                              'mr-2 shrink-0',
                              !!ref?.name && ref.name === activeRepoRef?.name
                                ? 'opacity-100'
                                : 'opacity-0'
                            )}
                          />
                          <span className="truncate" title={ref.name}>
                            {ref.name ?? ''}
                          </span>
                        </CommandItem>
                      ))}
                    </CommandGroup>
                  </CommandList>
                </Command>
              </PopoverContent>
            </Popover>
          </>
        )}
      </div>
    </div>
  )
}

export { FileTreeHeader }
