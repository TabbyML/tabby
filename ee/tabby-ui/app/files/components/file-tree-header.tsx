'use client'

import React, { useContext } from 'react'
import { isNil } from 'lodash-es'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import { RepositoryKind } from '@/lib/gql/generates/graphql'
import { useDebounceCallback } from '@/lib/hooks/use-debounce'
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
  IconClose,
  IconDirectorySolid,
  IconFile,
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
import {
  SearchableSelect,
  SearchableSelectAnchor,
  SearchableSelectContent,
  SearchableSelectInput,
  SearchableSelectOption
} from '@/components/searchable-select'

import { RepositoryKindIcon } from './repository-kind-icon'
import { SourceCodeBrowserContext } from './source-code-browser'
import {
  generateEntryPath,
  getDefaultRepoRef,
  repositoryMap2List,
  resolveRepoRef,
  resolveRepoSpecifierFromRepoInfo
} from './utils'

interface FileTreeHeaderProps extends React.HTMLAttributes<HTMLDivElement> {}

type SearchOption = {
  path: string
  type: string
  indices: number[]
  id: string
}

type RepositoryRefKind = 'branch' | 'tag'

const repositorySearch = graphql(/* GraphQL */ `
  query RepositorySearch(
    $kind: RepositoryKind!
    $id: ID!
    $rev: String
    $pattern: String!
  ) {
    repositorySearch(kind: $kind, id: $id, rev: $rev, pattern: $pattern) {
      type
      path
      indices
    }
  }
`)

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
  const repoId = activeRepo?.id
  const refs = activeRepo?.refs
  const formattedRefs = React.useMemo(() => {
    if (!refs?.length) return []
    return refs.map(ref => resolveRepoRef(ref))
  }, [refs])

  const branches = formattedRefs.filter(o => o.kind === 'branch')
  const tags = formattedRefs.filter(o => o.kind === 'tag')
  const commandOptions = activeRefKind === 'tag' ? tags : branches

  const inputRef = React.useRef<HTMLInputElement>(null)
  const [input, setInput] = React.useState<string>()
  const [repositorySearchPattern, setRepositorySearchPattern] =
    React.useState<string>()
  const [options, setOptions] = React.useState<Array<SearchOption>>()
  const [optionsVisible, setOptionsVisible] = React.useState(false)

  const noIndexedRepo = initialized && !repoList?.length

  const [{ data: repositorySearchData }] = useQuery({
    query: repositorySearch,
    variables: {
      kind: repositoryKind as RepositoryKind,
      id: repoId as string,
      pattern: repositorySearchPattern ?? '',
      rev: activeRepoRef?.name
    },
    pause: !repoId || !repositoryKind || !repositorySearchPattern
  })

  const onSelectRef = (ref: string) => {
    if (isNil(ref)) return
    const nextRev = resolveRepoRef(ref)?.name ?? ''
    const { basename = '' } = activeEntryInfo
    const kind = fileMap?.[basename]?.file?.kind ?? 'dir'

    // clear repository search
    setInput(undefined)
    updateActivePath(generateEntryPath(activeRepo, nextRev, basename, kind))
  }

  React.useEffect(() => {
    const _options =
      repositorySearchData?.repositorySearch?.map(option => ({
        ...option,
        id: option.path
      })) ?? []
    setOptions(_options)
    setOptionsVisible(!!repositorySearchPattern)
  }, [repositorySearchData?.repositorySearch])

  const onSelectRepo = (repoSpecifier: string) => {
    const repo = repoList.find(o => o.repoSpecifier === repoSpecifier)?.repo
    if (repo) {
      const path = `${repoSpecifier}/-/tree/${
        resolveRepoRef(getDefaultRepoRef(repo.refs)).name
      }`
      // clear repository search
      setInput(undefined)
      updateActivePath(path)
    }
  }

  const onInputValueChange = useDebounceCallback((v: string | undefined) => {
    if (!v) {
      setRepositorySearchPattern('')
      setOptionsVisible(false)
      setOptions([])
    } else {
      setRepositorySearchPattern(v)
    }
  }, 500)

  const onClearInput = () => {
    onInputValueChange.run('')
    onInputValueChange.flush()
  }

  const onSelectFile = async (value: SearchOption) => {
    if (!value.path) return
    const path = generateEntryPath(
      activeRepo,
      activeRepoRef?.name,
      value.path,
      value.type as any
    )
    updateActivePath(path)
  }

  // shortcut 't'
  React.useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const target = event.target as Element
      const tagName = target?.tagName?.toLowerCase()
      if (
        tagName === 'input' ||
        tagName === 'textarea' ||
        tagName === 'select'
      ) {
        return
      }

      if (event.key === 't') {
        event.preventDefault()
        inputRef.current?.focus()
      }
    }

    window.addEventListener('keydown', handleKeyDown)

    return () => {
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [])

  const onClickFilesTitle = () => {
    if (!activeRepo) return
    updateActivePath(
      generateEntryPath(activeRepo, activeEntryInfo?.rev, '', 'dir')
    )
  }

  return (
    <div className={cn(className)} {...props}>
      <div className="py-4 font-bold leading-8" onClick={onClickFilesTitle}>
        <span
          className={cn('py-1', {
            'hover:underline cursor-pointer': !!activeRepo
          })}
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
                      {activeRepoRef.kind === 'branch' ? (
                        <IconGitFork className="shrink-0" />
                      ) : (
                        <IconTag className="shrink-0" />
                      )}
                      <span className="truncate" title={activeRepoRef.name}>
                        {activeRepoRef.name ?? ''}
                      </span>
                    </>
                  )}
                </Button>
              </PopoverTrigger>
              <PopoverContent
                className="w-[var(--radix-popover-trigger-width)] p-0"
                align="start"
                side="bottom"
              >
                <Command className="transition-all">
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
                          key={ref.ref ?? refIndex}
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
            {/* Go to file */}
            <SearchableSelect
              stayOpenOnInputClick
              options={options}
              onSelect={onSelectFile}
              open={optionsVisible}
              onOpenChange={v => {
                if ((input || options?.length) && v) {
                  setOptionsVisible(true)
                } else {
                  setOptionsVisible(false)
                }
              }}
            >
              {({ highlightedIndex }) => {
                return (
                  <>
                    <SearchableSelectAnchor>
                      <div className="relative">
                        <SearchableSelectInput
                          className="pr-8"
                          placeholder="Go to file"
                          spellCheck={false}
                          value={input}
                          ref={inputRef}
                          disabled={!repositoryName}
                          onClick={e => {
                            if (repositorySearchPattern && !optionsVisible) {
                              setOptionsVisible(true)
                            }
                          }}
                          onChange={e => {
                            let value = e.target.value
                            setInput(value)
                            if (!value) {
                              onClearInput()
                            } else {
                              onInputValueChange.run(value)
                            }
                          }}
                        />
                        <div className="absolute right-2 top-0 flex h-full items-center">
                          {input ? (
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-6 w-6 cursor-pointer"
                              onClick={e => {
                                setInput('')
                                onClearInput()
                                inputRef.current?.focus()
                              }}
                            >
                              <IconClose />
                            </Button>
                          ) : (
                            <kbd
                              className="rounded-md border bg-secondary/50 px-1.5 text-xs leading-4 text-muted-foreground shadow-[inset_-0.5px_-1.5px_0_hsl(var(--muted))]"
                              onClick={e => {
                                inputRef.current?.focus()
                              }}
                            >
                              t
                            </kbd>
                          )}
                        </div>
                      </div>
                    </SearchableSelectAnchor>
                    <SearchableSelectContent
                      align="start"
                      side="bottom"
                      onOpenAutoFocus={e => e.preventDefault()}
                      style={{ width: '50vw', maxWidth: 700 }}
                      className="max-h-[50vh] overflow-y-auto"
                    >
                      <>
                        {options?.length ? (
                          options?.map((item, index) => (
                            <SearchableSelectOption
                              item={item}
                              index={index}
                              key={item?.id}
                              className="flex w-full items-center gap-2 overflow-x-hidden"
                            >
                              <div className="shrink-0">
                                {item.type === 'dir' ? (
                                  <IconDirectorySolid
                                    style={{ color: 'rgb(84, 174, 255)' }}
                                  />
                                ) : (
                                  <IconFile />
                                )}
                              </div>
                              <div className="flex-1 break-all">
                                <HighlightMatches
                                  text={item.path}
                                  indices={item.indices}
                                />
                              </div>
                              {highlightedIndex === index && (
                                <div className="shrink-0`">
                                  {item.type === 'dir'
                                    ? 'Go to folder'
                                    : 'Go to file'}
                                </div>
                              )}
                            </SearchableSelectOption>
                          ))
                        ) : (
                          <div className="flex h-24 items-center justify-center">
                            No matches found
                          </div>
                        )}
                      </>
                    </SearchableSelectContent>
                  </>
                )
              }}
            </SearchableSelect>
          </>
        )}
      </div>
    </div>
  )
}

const HighlightMatches = ({
  text,
  indices
}: {
  text: string
  indices: number[]
}) => {
  const indicesSet = React.useMemo(() => {
    return new Set(indices)
  }, [indices])

  return (
    <p className="text-muted-foreground">
      {text.split('').map((char, index) => {
        return indicesSet.has(index) ? (
          <span
            className="font-semibold text-foreground"
            key={`${char}_${index}`}
          >
            {char}
          </span>
        ) : (
          char
        )
      })}
    </p>
  )
}

export { FileTreeHeader }
