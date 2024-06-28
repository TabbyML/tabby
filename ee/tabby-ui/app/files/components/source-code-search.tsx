'use client'

import React, { KeyboardEvent as ReactKeyboardEvent } from 'react'

import { Button } from '@/components/ui/button'
import { IconClose, IconSearch } from '@/components/ui/icons'
import { Input } from '@/components/ui/input'

import { SourceCodeBrowserContext } from './source-code-browser'
import { generateEntryPath } from './utils'

interface SourceCodeSearchProps {}

const GLOBAL_SEARCH_SHORTCUT = 's'

export const SourceCodeSearch = ({ ...props }: SourceCodeSearchProps) => {
  const ctx = React.useContext(SourceCodeBrowserContext)

  const inputRef = React.useRef<HTMLInputElement>(null)

  const [query, setQuery] = React.useState<string>('')

  // TODO: Merge with file tree code
  React.useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const target = event.target as Element
      const tagName = target?.tagName?.toLowerCase()
      if (
        tagName === 'input' ||
        tagName === 'textarea' ||
        tagName === 'select'
      ) {
        if (event.key === 'Enter' || event.key === 'Escape') {
          return
        }
      } else if (event.key === GLOBAL_SEARCH_SHORTCUT) {
        event.preventDefault()
        inputRef.current?.focus()
      }
    }

    window.addEventListener('keydown', handleKeyDown)

    return () => {
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [inputRef])

  /**
   *
   */
  const onSubmit = async () => {
    const searchPath = generateEntryPath(
      ctx.activeRepo,
      ctx.activeRepoRef?.name,
      undefined,
      'search',
      query
    )

    await ctx.updateActivePath(searchPath)
  }

  const onInput = (event: React.FormEvent<HTMLInputElement>) => {
    setQuery(event.currentTarget.value)
  }

  const clearInput = () => {
    setQuery('')

    // Focus after the state is updated
    // TODO: Improve this. Is there a more idiomatic way to do this?
    setTimeout(() => {
      inputRef.current?.focus()
    }, 0)
  }

  const onFocus = () => {
    if (query) {
      // Temporarily disabled until the router kinks are worked out
      // onSubmit()
    }
  }

  const onKeydown = (event: ReactKeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Escape') {
      clearInput()
    }

    if (event.key === 'Enter') {
      event.preventDefault()
      onSubmit()
    }
  }

  return (
    <form
      onSubmit={onSubmit}
      className="w-full  bg-background py-3.5 px-4 shrink-0 flex items-center"
    >
      <div className="relative w-full">
        <Input
          type="search"
          ref={inputRef}
          placeholder="Search repository..."
          value={query}
          onInput={onInput}
          onFocus={onFocus}
          onKeyDown={onKeydown}
          className="w-full "
        />
        <div className="absolute right-2 top-0 flex h-full items-center">
          {query ? (
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6 cursor-pointer"
              onClick={clearInput}
            >
              <IconClose />
            </Button>
          ) : (
            <kbd
              className="rounded-md border bg-secondary/50 px-1.5 text-xs leading-4 text-muted-foreground shadow-[inset_-0.5px_-1.5px_0_hsl(var(--muted))]"
              onClick={() => {
                inputRef.current?.focus()
              }}
            >
              {GLOBAL_SEARCH_SHORTCUT}
            </kbd>
          )}
          <div className="border-l-border border-l flex items-center ml-2 pl-2">
            <Button
              variant="ghost"
              className="h-6 w-6 "
              size="icon"
              type="submit"
            >
              <IconSearch />
            </Button>
          </div>
        </div>
      </div>
    </form>
  )
}
