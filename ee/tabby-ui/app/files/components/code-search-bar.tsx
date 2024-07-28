import React, { ReactNode } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { useCombobox } from 'downshift'
import { trim } from 'lodash-es'
import { useQuery } from 'urql'

import {
  RepositoryKind,
  RepositorySearchQuery
} from '@/lib/gql/generates/graphql'
import { useDebounceValue } from '@/lib/hooks/use-debounce'
import { repositorySearch } from '@/lib/tabby/query'
import { ArrayElementType } from '@/lib/types'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  IconClose,
  IconDirectorySolid,
  IconFile,
  IconSearch
} from '@/components/ui/icons'
import { Input } from '@/components/ui/input'
import { Separator } from '@/components/ui/separator'

import { SourceCodeBrowserContext } from './source-code-browser'
import { generateEntryPath } from './utils'

interface CodeSearchBarProps extends React.OlHTMLAttributes<HTMLDivElement> {}

type RepositorySearchItem = ArrayElementType<
  RepositorySearchQuery['repositorySearch']
>
type OptionItem = {
  label: string | ReactNode
  value: string
  disabled?: boolean
  type: 'file' | 'pattern' | 'tips'
  repositorySearch?: RepositorySearchItem
}

export const CodeSearchBar: React.FC<CodeSearchBarProps> = ({ className }) => {
  const router = useRouter()
  const searchParams = useSearchParams()
  const { activeEntryInfo, activeRepo, activeRepoRef, updateActivePath } =
    React.useContext(SourceCodeBrowserContext)
  const [query, setQuery] = React.useState(searchParams.get('q')?.toString())
  const [debouncedQuery] = useDebounceValue(query, 300)
  const inputRef = React.useRef<HTMLInputElement>(null)
  const clearInput = () => {
    setQuery('')
    inputRef.current?.focus()
  }

  const repositoryKind = activeRepo?.kind
  const repoId = activeRepo?.id

  const repositorySearchPattern = React.useMemo(() => {
    const regex = /^f:(.+)/
    const matches = trim(debouncedQuery).match(regex)
    if (matches) return matches[1] || undefined
    return debouncedQuery
  }, [debouncedQuery])

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

  const repositorySearchOptions: Array<OptionItem> = React.useMemo(() => {
    if (!repositorySearchPattern) return []
    return (
      repositorySearchData?.repositorySearch?.map(option => ({
        repositorySearch: option,
        value: option.path,
        label: option.path,
        type: 'file'
      })) ?? []
    )
  }, [repositorySearchData?.repositorySearch, repositorySearchPattern])

  const {
    isOpen,
    getMenuProps,
    getInputProps,
    highlightedIndex,
    getItemProps,
    openMenu
  } = useCombobox({
    items: repositorySearchOptions,
    onSelectedItemChange({ selectedItem }) {
      if (selectedItem?.type === 'file' && selectedItem.repositorySearch) {
        const path = generateEntryPath(
          activeRepo,
          activeRepoRef?.name,
          selectedItem.repositorySearch.path,
          selectedItem.repositorySearch.type as 'file' | 'dir'
        )
        updateActivePath(path)
        return
      }

      onSubmit(selectedItem?.value)
    },
    stateReducer(_state, actionAndChanges) {
      const { type, changes } = actionAndChanges
      switch (type) {
        case useCombobox.stateChangeTypes.InputClick:
          return {
            ...changes,
            highlightedIndex: undefined,
            isOpen: true
          }
        default:
          return changes
      }
    }
  })

  const onInputValueChange = (val: string) => {
    if (!isOpen) {
      openMenu()
    }
    setQuery(val)
  }

  // shortcut '[/]'
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

      if (event.key === '/') {
        event.preventDefault()
        inputRef.current?.focus()
        openMenu()
      }
    }

    window.addEventListener('keydown', handleKeyDown)

    return () => {
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [])

  const onSubmit = (pattern: string | undefined) => {
    if (!pattern) return

    const pathname = generateEntryPath(
      activeRepo,
      activeEntryInfo?.rev,
      '',
      'search'
    )

    router.push(`/files/${pathname}?q=${encodeURIComponent(pattern)}`)
  }

  return (
    <div
      className={cn(
        'flex w-full shrink-0 items-center bg-background px-4 py-3.5 transition duration-500 ease-in-out',
        className
      )}
    >
      <div className={cn('relative w-full')}>
        <div className="h-9">
          <div
            className={cn({
              'absolute z-10 bg-white dark:bg-popover dark:text-secondary-foreground inset-0':
                isOpen
            })}
          >
            <Input
              placeholder="Type [/] to search"
              className="w-full"
              // autoComplete="off"
              {...getInputProps({
                onKeyDown: e => {
                  if (e.key === 'Enter' && !e.nativeEvent.isComposing) {
                    e.preventDefault()
                    onSubmit(query)
                  }
                },
                ref: inputRef
              })}
              value={query}
              onChange={e => onInputValueChange(e.target.value)}
            />
          </div>
        </div>
        <div className="absolute right-2 top-0 z-20 flex h-full items-center">
          {query ? (
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6 cursor-pointer"
              onClick={clearInput}
            >
              <IconClose />
            </Button>
          ) : null}
          <div className="z-20 ml-2 flex items-center border-l border-l-border pl-2">
            <Button
              variant="ghost"
              className="h-6 w-6 "
              size="icon"
              onClick={() => onSubmit(query)}
            >
              <IconSearch />
            </Button>
          </div>
        </div>
        {isOpen && (
          <div
            className="absolute -inset-x-3 -top-2 flex max-h-[60vh] flex-col overflow-hidden rounded-lg border bg-white p-4 shadow-2xl dark:bg-popover dark:text-secondary-foreground"
            {...getMenuProps()}
          >
            <div className="h-12 shrink-0" />
            <div className="flex-1 overflow-y-auto">
              {!!repositorySearchOptions?.length && (
                <>
                  <div className="text-md mb-1 pl-2 font-semibold">Code</div>
                  {repositorySearchOptions.map((item, index) => {
                    const repositorySearch =
                      item.repositorySearch as RepositorySearchItem
                    const highlighted = highlightedIndex === index
                    return (
                      <div
                        key={item.repositorySearch?.path}
                        className={cn(
                          'relative flex cursor-default select-none items-center gap-1 rounded-sm px-2 py-1.5 text-sm outline-none',
                          highlighted &&
                            'cursor-pointer bg-accent text-accent-foreground'
                        )}
                        {...getItemProps({
                          item,
                          index,
                          onMouseLeave: e => e.preventDefault(),
                          onMouseOut: e => e.preventDefault()
                        })}
                      >
                        <div className="shrink-0">
                          {item?.repositorySearch?.type === 'dir' ? (
                            <IconDirectorySolid
                              style={{ color: 'rgb(84, 174, 255)' }}
                            />
                          ) : (
                            <IconFile />
                          )}
                        </div>
                        <div className="flex-1 break-all">
                          <HighlightMatches
                            text={repositorySearch.path}
                            indices={repositorySearch.indices}
                          />
                        </div>
                        <div className="shrink-0 text-sm text-muted-foreground">
                          Jump to
                        </div>
                      </div>
                    )
                  })}
                  <Separator className="my-2" />
                </>
              )}
              <div className="text-md mb-1 pl-2 font-semibold">
                Narrow your search
              </div>
              <div className="flex items-center gap-1 px-2 text-sm">
                <div className="text-secondary-foreground">
                  <span className="mr-0.5 bg-secondary px-1 py-0.5 text-secondary-foreground">
                    f:
                  </span>
                  Include only results from file path matching the given search
                  pattern.
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function HighlightMatches({
  text,
  indices
}: {
  text: string
  indices: number[]
}) {
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
