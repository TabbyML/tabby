'use client'

import React, { useContext } from 'react'
import { isEmpty } from 'lodash-es'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import { useDebounceCallback } from '@/lib/hooks/use-debounce'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  IconClose,
  IconDirectorySolid,
  IconFile,
  IconFolderGit
} from '@/components/ui/icons'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'
import {
  SearchableSelect,
  SearchableSelectAnchor,
  SearchableSelectContent,
  SearchableSelectInput,
  SearchableSelectOption
} from '@/components/searchable-select'
import { useTopbarProgress } from '@/components/topbar-progress-indicator'

import { RepositoryKindIcon } from './repository-kind-icon'
import { SourceCodeBrowserContext, TFileMap } from './source-code-browser'
import {
  fetchEntriesFromPath,
  getDirectoriesFromBasename,
  key2RepositoryKind,
  resolveFileNameFromPath,
  resolveRepoIdFromPath,
  resolveRepoKindFromPath,
  resolveRepoSpecifierFromPath
} from './utils'

interface FileTreeHeaderProps extends React.HTMLAttributes<HTMLDivElement> { }

type SearchOption = {
  path: string
  type: string
  indices: number[]
  id: string
}

const repositorySearch = graphql(/* GraphQL */ `
  query RepositorySearch($kind: RepositoryKind!, $id: ID!, $pattern: String!) {
    repositorySearch(kind: $kind, id: $id, pattern: $pattern) {
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
    activePath,
    fileTreeData,
    setActivePath,
    initialized,
    updateFileMap,
    setExpandedKeys,
    fileMap
  } = useContext(SourceCodeBrowserContext)
  const { setProgress } = useTopbarProgress()

  const repoSpecifier = resolveRepoSpecifierFromPath(activePath)
  const repoKind = resolveRepoKindFromPath(activePath)
  const repoId = resolveRepoIdFromPath(activePath)
  const currentRepoKind = React.useMemo(() => {
    if (!repoSpecifier || isEmpty(fileMap)) return undefined
    return fileMap?.[repoSpecifier]?.repository?.kind
  }, [repoSpecifier, fileMap])
  const currentRepoName = React.useMemo(() => {
    if (!repoSpecifier || isEmpty(fileMap)) return undefined
    return fileMap?.[repoSpecifier]?.name
  }, [repoSpecifier, fileMap])

  const inputRef = React.useRef<HTMLInputElement>(null)
  const [input, setInput] = React.useState<string>()
  const [repositorySearchPattern, setRepositorySearchPattern] =
    React.useState<string>()
  const [options, setOptions] = React.useState<Array<SearchOption>>()
  const [optionsVisible, setOptionsVisible] = React.useState(false)

  const noIndexedRepo = initialized && !fileTreeData?.length

  const [{ data: repositorySearchData }] = useQuery({
    query: repositorySearch,
    variables: {
      kind: key2RepositoryKind(repoKind),
      id: repoId,
      pattern: repositorySearchPattern ?? ''
    },
    pause: !repoSpecifier || !repositorySearchPattern
  })

  React.useEffect(() => {
    const _options =
      repositorySearchData?.repositorySearch?.map(option => ({
        ...option,
        id: option.path
      })) ?? []
    setOptions(_options)
    setOptionsVisible(!!repositorySearchPattern)
  }, [repositorySearchData?.repositorySearch])

  const onSelectRepo = (name: string) => {
    setActivePath(name)
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
    const path = value.path
    if (!path) return

    const fullPath = `${repoSpecifier}/${path}`
    try {
      setProgress(true)
      const entries = await fetchEntriesFromPath(fullPath)
      const initialExpandedDirs = getDirectoriesFromBasename(path)

      const patchMap: TFileMap = {}
      // fetch dirs
      for (const entry of entries) {
        const path = `${repoSpecifier}/${entry.basename}`
        patchMap[path] = {
          file: entry,
          name: resolveFileNameFromPath(path),
          fullPath: path,
          treeExpanded: initialExpandedDirs.includes(entry.basename)
        }
      }
      const expandedKeys = initialExpandedDirs.map(dir =>
        [repoSpecifier, dir].filter(Boolean).join('/')
      )
      if (patchMap) {
        updateFileMap(patchMap)
      }
      if (expandedKeys?.length) {
        setExpandedKeys(prevKeys => {
          const newSet = new Set(prevKeys)
          for (const k of expandedKeys) {
            newSet.add(k)
          }
          return newSet
        })
      }
    } catch (e) {
    } finally {
      setActivePath(fullPath)
      setProgress(false)
    }
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

  return (
    <div className={cn(className)} {...props}>
      <div className="py-4 font-bold leading-8">Files</div>
      <div className="space-y-3">
        <Select
          disabled={!initialized}
          onValueChange={onSelectRepo}
          value={repoSpecifier}
        >
          <SelectTrigger>
            <SelectValue>
              <div className="flex items-center gap-2">
                <RepositoryKindIcon
                  kind={currentRepoKind}
                  fallback={<IconFolderGit />}
                />
                <span
                  className={currentRepoName ? '' : 'text-muted-foreground'}
                >
                  {currentRepoName || 'Pick a repository'}
                </span>
              </div>
            </SelectValue>
          </SelectTrigger>
          <SelectContent className="max-h-[50vh] overflow-y-auto">
            {noIndexedRepo ? (
              <SelectItem isPlaceHolder value="" disabled>
                No indexed repository
              </SelectItem>
            ) : (
              <>
                {fileTreeData?.map(repo => {
                  return (
                    <SelectItem key={repo.fullPath} value={repo.fullPath}>
                      <div className='flex items-center gap-1'>
                        <RepositoryKindIcon
                          kind={repo?.repository?.kind}
                          fallback={<IconFolderGit />}
                        />
                        {repo.name}
                      </div>
                    </SelectItem>
                  )
                })}
              </>
            )}
          </SelectContent>
        </Select>
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
                      disabled={!currentRepoName}
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
          <span className="font-semibold text-foreground">{char}</span>
        ) : (
          char
        )
      })}
    </p>
  )
}

export { FileTreeHeader }
