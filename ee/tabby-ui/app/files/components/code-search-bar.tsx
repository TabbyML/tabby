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
  IconFilter,
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
    if (!debouncedQuery) return undefined

    const prefixRegex = /-?(f|lang):\S+\s?/g
    return trim(debouncedQuery.replace(prefixRegex, ''))
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
    const _options = repositorySearchData?.repositorySearch?.slice(0, 5)
    return (
      _options?.map(option => ({
        repositorySearch: option,
        value: option.path,
        label: option.path,
        type: 'file'
      })) ?? []
    )
  }, [repositorySearchData?.repositorySearch, repositorySearchPattern])

  const filerOptions: Array<OptionItem> = React.useMemo(() => {
    const _options: Array<OptionItem> = [
      {
        label:
          'Include only results from file path matching the given search pattern.',
        value: 'f',
        type: 'tips'
      },
      {
        label:
          'Exclude results from file path matching the given search pattern.',
        value: '-f',
        type: 'tips'
      },
      {
        label: 'Include only results from the given language.',
        value: 'lang',
        type: 'tips'
      },
      {
        label: 'Exclude results from the given language.',
        value: '-lang',
        type: 'tips'
      }
    ]
    if (!query) return [_options[0], _options[2]]
    let negativeRuleRegex = /(^|\s)-$/
    let fileRegx = /(^|\s)-?f$/
    let langRegx = /(^|\s)-?l(a(n(g)?)?)?$/

    const negativeRuleMatches = query.match(negativeRuleRegex)
    const fileRegxMatches = query.match(fileRegx)
    const langRegxMatches = query.match(langRegx)

    if (negativeRuleMatches) return [_options[1], _options[3]]
    if (!fileRegxMatches && !langRegxMatches) return []
    if (fileRegxMatches) {
      return _options.slice(0, 2)
    }
    if (langRegxMatches) {
      return _options.slice(2)
    }
    return []
  }, [query])

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
        case useCombobox.stateChangeTypes.InputKeyDownArrowDown: {
          if (!repositorySearchOptions?.length || !_state.isOpen) return changes
          const isLastItemHighlighted =
            _state.highlightedIndex === repositorySearchOptions.length - 1
          return {
            ...changes,
            highlightedIndex: isLastItemHighlighted
              ? undefined
              : changes.highlightedIndex
          }
        }
        case useCombobox.stateChangeTypes.InputKeyDownArrowUp: {
          if (!repositorySearchOptions?.length || !_state.isOpen) return changes
          const isFirstItemHighlighted = _state.highlightedIndex === 0
          return {
            ...changes,
            highlightedIndex: isFirstItemHighlighted
              ? undefined
              : changes.highlightedIndex
          }
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

  // shortcut '/'
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
        return
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

  const noOptions = !filerOptions?.length && !repositorySearchOptions?.length

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
              'absolute z-10 inset-0': isOpen
            })}
          >
            <Input
              className="w-full"
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
          {!query && (
            <div
              className="absolute left-3 top-1.5 cursor-text select-none text-muted-foreground"
              onClick={e => {
                e.preventDefault()
                inputRef.current?.focus()
                openMenu()
              }}
            >
              Type{' '}
              <kbd className="rounded border border-muted-foreground px-0.5">
                /
              </kbd>{' '}
              to search
            </div>
          )}
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
            className={cn(
              'absolute -inset-x-3 -top-2.5 flex max-h-[60vh] flex-col overflow-hidden rounded-lg border bg-background p-4 shadow-2xl dark:border-2 dark:border-[#33363c] dark:bg-[hsl(0,0,13.5%)]',
              {
                'pb-0.5': noOptions
              }
            )}
            {...getMenuProps({
              suppressRefError: true
            })}
          >
            <div className={cn('shrink-0', noOptions ? 'h-9' : 'h-12')} />
            <div className="flex-1 overflow-y-auto">
              {!!filerOptions?.length && (
                <>
                  <div className="text-md mb-2 pl-2 font-semibold">
                    Narrow your search
                  </div>
                  <div className="space-y-2">
                    {filerOptions.map(item => {
                      return <NarrowSearchItem data={item} key={item.value} />
                    })}
                  </div>
                </>
              )}
              {!!repositorySearchOptions?.length && (
                <>
                  {!!filerOptions?.length && <Separator className="my-2" />}
                  <div className="text-md mb-1 pl-2 font-semibold">Code</div>
                  {repositorySearchOptions.map((item, index) => {
                    const repositorySearch =
                      item.repositorySearch as RepositorySearchItem
                    const highlighted = highlightedIndex === index
                    return (
                      <div
                        key={item.repositorySearch?.path}
                        className={cn(
                          'relative flex cursor-default select-none items-center gap-1 rounded-sm px-2 py-1.5 text-sm',
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
                </>
              )}
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

function NarrowSearchItem({ data }: { data: OptionItem }) {
  const { label, value } = data
  return (
    <div className="flex items-center gap-1 px-2 text-sm">
      <IconFilter className="shrink-0" />
      <div className="text-secondary-foreground">
        <span className="mr-0.5 rounded bg-secondary px-1 py-0.5 text-secondary-foreground">
          {value}:
        </span>
        <span>{label}</span>
      </div>
    </div>
  )
}
