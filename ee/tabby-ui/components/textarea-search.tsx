'use client'

import { MouseEvent, useEffect, useMemo, useRef, useState } from 'react'
import Link from 'next/link'
import { omit } from 'lodash-es'
import TextareaAutosize from 'react-textarea-autosize'
import { useQuery } from 'urql'

import { SESSION_STORAGE_KEY } from '@/lib/constants'
import { Repository } from '@/lib/gql/generates/graphql'
import { useCurrentTheme } from '@/lib/hooks/use-current-theme'
import { repositoryListQuery } from '@/lib/tabby/query'
import { AnswerEngineExtraContext } from '@/lib/types'
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
  isFollowup,
  extraContext
}: {
  onSearch: (value: string, extraContext: AnswerEngineExtraContext) => void
  className?: string
  placeholder?: string
  showBetaBadge?: boolean
  isLoading?: boolean
  autoFocus?: boolean
  loadingWithSpinning?: boolean
  cleanAfterSearch?: boolean
  isFollowup?: boolean
  extraContext?: AnswerEngineExtraContext
}) {
  const [isShow, setIsShow] = useState(false)
  const [isFocus, setIsFocus] = useState(false)
  const [value, setValue] = useState('')
  const [selectedRepo, setSelectedRepo] = useState<
    AnswerEngineExtraContext['repository'] | undefined
  >()
  const { theme } = useCurrentTheme()
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const getPreviouslySelectedRepo = () => {
    try {
      const selectedRepoValue = sessionStorage.getItem(
        SESSION_STORAGE_KEY.SEARCH_SELECTED_REPO
      )
      if (selectedRepoValue) {
        const temp = JSON.parse(selectedRepoValue)
        if (temp) {
          setSelectedRepo(temp)
        }
      }
    } catch (e) {}
  }

  useEffect(() => {
    // Ensure the textarea height remains consistent during rendering
    setIsShow(true)
    // try to load cached repo selection
    getPreviouslySelectedRepo()
  }, [])

  const onSearchKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) return e.preventDefault()
  }

  const onSearchKeyUp = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      return search()
    }
  }

  const search = (e?: MouseEvent<HTMLDivElement>) => {
    if (!value || isLoading) return
    e?.stopPropagation()
    onSearch(value, { repository: selectedRepo })
    if (cleanAfterSearch) setValue('')
  }

  const onWrapperClick = () => {
    if (isFollowup) {
      textareaRef.current?.focus()
    }
  }

  useEffect(() => {
    if (extraContext?.repository) {
      setSelectedRepo(extraContext?.repository)
    }
  }, [extraContext])

  const showRepoSelect = !isFollowup || !!extraContext?.repository

  return (
    <div
      className={cn(
        'relative overflow-hidden rounded-lg border border-muted-foreground bg-background px-4 transition-all hover:border-muted-foreground/60',
        {
          'flex-col gap-1 w-full': showRepoSelect,
          'flex w-full items-center ': !showRepoSelect,
          '!border-zinc-400': isFocus && isFollowup && theme !== 'dark',
          '!border-primary': isFocus && (!isFollowup || theme === 'dark'),
          'py-0': showBetaBadge,
          'border-2 dark:border border-zinc-400 hover:border-zinc-400/60 dark:border-muted-foreground dark:hover:border-muted-foreground/60':
            isFollowup
        },
        className
      )}
      onClick={onWrapperClick}
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
          'text-area-autosize mr-1 w-full flex-1 resize-none rounded-lg !border-none bg-transparent !shadow-none !outline-none !ring-0 !ring-offset-0',
          {
            '!h-[48px]': !isShow,
            'pt-4': !showBetaBadge,
            'pt-5': showBetaBadge,
            'pb-4': !showRepoSelect && !showBetaBadge,
            'pb-5': !showRepoSelect && showBetaBadge
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
        ref={textareaRef}
      />
      <div
        className={cn('flex items-center justify-between gap-2', {
          'pb-2': showRepoSelect
        })}
      >
        {showRepoSelect && (
          <RepoSelect
            className="overflow-hidden"
            value={selectedRepo}
            onChange={setSelectedRepo}
            disabled={isFollowup}
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
interface RepoSelectProps {
  value: AnswerEngineExtraContext['repository'] | undefined
  onChange: (val: AnswerEngineExtraContext['repository'] | undefined) => void
  className?: string
  disabled?: boolean
}
function RepoSelect({ value, onChange, className, disabled }: RepoSelectProps) {
  const [commandVisible, setCommandVisible] = useState(false)
  const [{ data, fetching }] = useQuery({
    query: repositoryListQuery
  })
  const repos = data?.repositoryList

  const emptyText = useMemo(() => {
    if (!repos?.length)
      return (
        <div className="space-y-4 py-2">
          <p className="font-semibold">No repositories</p>
          <Link
            href="/settings/providers/git"
            className={cn(buttonVariants({ size: 'sm' }), 'gap-1')}
          >
            Connect
            <IconArrowRight />
          </Link>
        </div>
      )

    return 'No results found'
  }, [repos])

  const onSelectRepo = (repo: Repository) => {
    onChange(repo)
    setCommandVisible(false)
    sessionStorage.setItem(
      SESSION_STORAGE_KEY.SEARCH_SELECTED_REPO,
      JSON.stringify(omit(repo, 'refs'))
    )
  }

  return (
    <Tooltip delayDuration={0}>
      <Popover
        open={commandVisible}
        onOpenChange={e => {
          if (disabled) return
          setCommandVisible(e)
        }}
      >
        <PopoverTrigger asChild>
          <TooltipTrigger asChild>
            <div
              className={cn(
                buttonVariants({ variant: 'ghost' }),
                '-ml-2 cursor-pointer rounded-full px-2',
                {
                  'cursor-default hover:bg-transparent': disabled
                },
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
                  {value?.name ?? 'Select repository'}
                </span>
              </div>
            </div>
          </TooltipTrigger>
        </PopoverTrigger>
        <PopoverPortal>
          <PopoverContent
            className="min-w-[300px] lg:max-w-[60vw]"
            align="start"
            side="bottom"
          >
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
                  {repos?.map(repo => {
                    const isSelected =
                      !!value?.id &&
                      `${repo.kind}_${repo.id}` === `${value.kind}_${value.id}`
                    return (
                      <CommandItem
                        value={`${repo.kind}_${repo.id}`}
                        key={`${repo.kind}_${repo.id}`}
                        onSelect={() => onSelectRepo(repo)}
                        className="flex cursor-pointer items-center gap-2 overflow-hidden"
                      >
                        <div className="h-4 w-4 shrink-0">
                          {isSelected && <IconCheck className="shrink-0" />}
                        </div>
                        <span className="truncate">{repo.name}</span>
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
                      sessionStorage.removeItem(
                        SESSION_STORAGE_KEY.SEARCH_SELECTED_REPO
                      )
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
        Effortlessly interact with your repositories for contextualized search
        and assistance.
      </TooltipContent>
    </Tooltip>
  )
}
