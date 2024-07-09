'use client'

import { useEffect, useMemo, useState } from 'react'
import Link from 'next/link'
import TextareaAutosize from 'react-textarea-autosize'
import { useQuery } from 'urql'

import { Repository } from '@/lib/gql/generates/graphql'
import { useCurrentTheme } from '@/lib/hooks/use-current-theme'
import { repositoryListQuery } from '@/lib/tabby/query'
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
import {
  Popover,
  PopoverContent,
  PopoverPortal,
  PopoverTrigger
} from '@/components/ui/popover'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'

import { buttonVariants } from './ui/button'
import { IconArrowRight, IconCheck, IconCode, IconSpinner } from './ui/icons'

export default function TextAreaSearch({
  onSearch,
  className,
  placeholder,
  showBetaBadge,
  isLoading,
  autoFocus,
  loadingWithSpinning,
  cleanAfterSearch = true,
  isFollowup
}: {
  onSearch: (value: string) => void
  className?: string
  placeholder?: string
  showBetaBadge?: boolean
  isLoading?: boolean
  autoFocus?: boolean
  loadingWithSpinning?: boolean
  cleanAfterSearch?: boolean
  isFollowup?: boolean
}) {
  const [isShow, setIsShow] = useState(false)
  const [isFocus, setIsFocus] = useState(false)
  const [value, setValue] = useState('')
  const [selectedRepo, setSelectRepo] = useState<RepoItem | undefined>()
  const { theme } = useCurrentTheme()

  useEffect(() => {
    // Ensure the textarea height remains consistent during rendering
    setIsShow(true)
  }, [])

  const onSearchKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) return e.preventDefault()
  }

  const onSearchKeyUp = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      return search()
    }
  }

  const search = () => {
    if (!value || isLoading) return
    onSearch(value)
    if (cleanAfterSearch) setValue('')
  }

  return (
    <div
      className={cn(
        'relative overflow-hidden rounded-lg border border-muted-foreground bg-background px-4 transition-all hover:border-muted-foreground/60',
        {
          'flex-col gap-1 w-full': !isFollowup,
          'flex w-full items-center ': isFollowup,
          '!border-zinc-400': isFocus && isFollowup && theme !== 'dark',
          '!border-primary': isFocus && (!isFollowup || theme === 'dark'),
          'py-0': showBetaBadge,
          'border-2 dark:border border-zinc-400 hover:border-zinc-400/60 dark:border-muted-foreground dark:hover:border-muted-foreground/60':
            isFollowup
        },
        className
      )}
    >
      {showBetaBadge && (
        <span
          className="absolute -right-8 top-1 mr-3 rotate-45 rounded-none border-none py-0.5 pl-6 pr-5 text-xs text-primary"
          style={{ background: theme === 'dark' ? '#333' : '#e8e1d3' }}
        >
          Beta
        </span>
      )}
      <TextareaAutosize
        className={cn(
          'text-area-autosize flex-1 resize-none rounded-lg !border-none bg-transparent !shadow-none !outline-none !ring-0 !ring-offset-0',
          {
            'w-full': !isFollowup,
            '!h-[48px]': !isShow,
            'pt-4': !showBetaBadge,
            'pt-5': showBetaBadge,
            'pb-4': isFollowup && !showBetaBadge,
            'pb-5': isFollowup && showBetaBadge
          }
        )}
        placeholder={placeholder || 'Ask anything...'}
        maxRows={5}
        onKeyDown={onSearchKeyDown}
        onKeyUp={onSearchKeyUp}
        onFocus={() => setIsFocus(true)}
        onBlur={() => setIsFocus(false)}
        onChange={e => setValue(e.target.value)}
        value={value}
        autoFocus={autoFocus}
        minRows={isFollowup ? 1 : 2}
      />
      <div
        className={cn('flex items-center justify-between gap-2', {
          'pb-2': !isFollowup
        })}
      >
        {!isFollowup && (
          <RepoSelect
            className="overflow-hidden"
            value={selectedRepo}
            onChange={setSelectRepo}
          />
        )}
        <div
          className={cn(
            'flex items-center justify-center rounded-lg p-1 transition-all',
            {
              'bg-primary text-primary-foreground cursor-pointer':
                value.length > 0,
              '!bg-muted !text-primary !cursor-default':
                isLoading || value.length === 0,
              'mr-1.5': !showBetaBadge,
              'h-6 w-6': !isFollowup
              // 'mr-6': showBetaBadge,
            }
          )}
          onClick={search}
        >
          {loadingWithSpinning && isLoading && (
            <IconSpinner className="h-3.5 w-3.5" />
          )}
          {(!loadingWithSpinning || !isLoading) && (
            <IconArrowRight className="h-3.5 w-3.5" />
          )}
        </div>
      </div>
    </div>
  )
}

type RepoItem = Pick<Partial<Repository>, 'id' | 'kind' | 'name'> & {
  allCode?: boolean
}

interface RepoSelectProps {
  value: RepoItem | undefined
  onChange: (val: RepoItem | undefined) => void
  className?: string
}
function RepoSelect({ value, onChange, className }: RepoSelectProps) {
  const [commandVisible, setCommandVisible] = useState(false)
  const [{ data, fetching }] = useQuery({
    query: repositoryListQuery
  })
  const repos = data?.repositoryList

  const emptyText = useMemo(() => {
    if (!repos?.length)
      return (
        <div className="space-y-4 py-2">
          <p className="font-semibold">No indexed repositories</p>
          <Link
            href="/settings/providers/git"
            className={cn(buttonVariants({ size: 'sm' }), 'gap-1')}
          >
            Add repositories
            <IconArrowRight />
          </Link>
        </div>
      )

    return 'No results found'
  }, [repos])

  const isAllSelected = !!repos?.length && value?.allCode

  return (
    <Tooltip delayDuration={0}>
      <Popover open={commandVisible} onOpenChange={e => setCommandVisible(e)}>
        <PopoverTrigger asChild>
          <TooltipTrigger asChild>
            <div
              className={cn(
                buttonVariants({ variant: 'ghost' }),
                '-ml-2 cursor-pointer rounded-full px-2',
                className
              )}
            >
              <div className="flex items-center gap-2 overflow-hidden">
                <IconCode
                  className={cn(
                    'shrink-0',
                    value ? 'text-foreground/70' : 'text-foreground/50'
                  )}
                />
                <span
                  className={cn(
                    'flex-1 truncate',
                    value ? 'text-foreground/70' : 'text-foreground/50'
                  )}
                >
                  {isAllSelected
                    ? 'All Repositories'
                    : value
                    ? value.name
                    : 'Select repository'}
                </span>
              </div>
            </div>
          </TooltipTrigger>
        </PopoverTrigger>
        <PopoverPortal>
          <PopoverContent className="min-w-[300px]" align="start" side="bottom">
            <Command>
              <CommandInput placeholder="Search" />
              <CommandList className="max-h-[200px]">
                <CommandEmpty>
                  {fetching ? (
                    <div className="flex justify-center">
                      <IconSpinner className="h-6 w-6" />
                    </div>
                  ) : (
                    emptyText
                  )}
                </CommandEmpty>
                <CommandGroup>
                  {!!repos?.length && (
                    <>
                      <CommandItem
                        key="all"
                        onSelect={e => {
                          if (isAllSelected) {
                            onChange(undefined)
                          } else {
                            onChange({ allCode: true })
                          }
                          setCommandVisible(false)
                        }}
                        className="flex items-center gap-2"
                      >
                        <div className="w-4 shrink-0">
                          {isAllSelected && <IconCheck className="shrink-0" />}
                        </div>
                        <span>All indexed repositories</span>
                      </CommandItem>
                      <CommandSeparator className="my-2" />
                    </>
                  )}
                  {repos?.map(repo => {
                    const isSelected = repo.id === value?.id
                    return (
                      <CommandItem
                        key={repo.id}
                        onSelect={() => {
                          onChange({ ...repo, allCode: false })
                          setCommandVisible(false)
                        }}
                        className="flex cursor-pointer items-center gap-2"
                      >
                        <div className="h-4 w-4 shrink-0">
                          {isSelected && <IconCheck className="shrink-0" />}
                        </div>
                        <span>{repo.name}</span>
                      </CommandItem>
                    )
                  })}
                </CommandGroup>
              </CommandList>
              {!!value && (
                <>
                  <CommandSeparator />
                  <CommandItem
                    onSelect={() => {
                      onChange(undefined)
                      setCommandVisible(false)
                    }}
                    className="!pointer-events-auto mt-1 cursor-pointer justify-center text-center !opacity-100"
                  >
                    Clear
                  </CommandItem>
                </>
              )}
            </Command>
          </PopoverContent>
        </PopoverPortal>
      </Popover>
      <TooltipContent>
    Effortlessly interact with your repositories for contextualized search and assistance.
      </TooltipContent>
    </Tooltip>
  )
}
