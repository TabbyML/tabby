'use client'

import React, {
  Dispatch,
  KeyboardEvent as ReactKeyboardEvent,
  Ref,
  RefAttributes,
  SetStateAction,
  useEffect
} from 'react'

import { Button } from '@/components/ui/button'
import { IconClose, IconSearch } from '@/components/ui/icons'
import { Input } from '@/components/ui/input'

interface GlobalSearchProps {
  query: string
  onFocus: () => void
  onInput: (e: React.ChangeEvent<HTMLInputElement>) => void
  onSubmit: (
    e: React.FormEvent<HTMLFormElement> | ReactKeyboardEvent<HTMLInputElement>
  ) => void
  inputRef: HTMLInputElement | undefined
  setInputRef: Dispatch<SetStateAction<HTMLInputElement | undefined>>
  clearInput: () => void
}

const GLOBAL_SEARCH_SHORTCUT = 's'

export const GlobalSearch = ({ ...props }: GlobalSearchProps) => {
  // TODO: Merge this with the file tree code
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const target = event.target as Element
      const tagName = target?.tagName?.toLowerCase()
      if (
        tagName === 'input' ||
        tagName === 'textarea' ||
        tagName === 'select'
      ) {
        if (event.key === 'Enter' || event.key === 'Escape') {
          console.log('return')
          return
        }
      }

      if (event.key === GLOBAL_SEARCH_SHORTCUT) {
        event.preventDefault()
        props.inputRef?.focus()
      }
    }

    // FIXME: this is broken
    window.addEventListener('keydown', handleKeyDown)

    return () => {
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [props.inputRef])

  return (
    <form onSubmit={props.onSubmit} className="w-full flex items-center h-14">
      <div className="relative w-full">
        <Input
          type="search"
          placeholder="Search repository..."
          value={props.query}
          onInput={props.onInput}
          onKeyDown={e => {
            if (e.key === 'Escape') {
              props.clearInput()
            }

            if (e.key === 'Enter') {
              e.preventDefault()
              props.onSubmit(e)
            }
          }}
          onFocus={props.onFocus}
          ref={ref => {
            props.setInputRef(ref as HTMLInputElement)
          }}
          className="w-full "
        />
        <div className="absolute right-2 top-0 flex h-full items-center">
          {props.query ? (
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6 cursor-pointer"
              onClick={props.clearInput}
            >
              <IconClose />
            </Button>
          ) : (
            <kbd
              className="rounded-md border bg-secondary/50 px-1.5 text-xs leading-4 text-muted-foreground shadow-[inset_-0.5px_-1.5px_0_hsl(var(--muted))]"
              onClick={() => {
                props.inputRef?.focus()
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
